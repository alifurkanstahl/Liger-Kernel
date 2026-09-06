import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import infer_device
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops.qwen4_exp import LigerGroupRMSNormFusedFunction
from liger_kernel.ops.qwen4_exp import LigerGroupRMSNormWrite4Function
from liger_kernel.ops.qwen4_exp import LigerQwen4ExpGRWriteFunction
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.utils import is_hip
from liger_kernel.transformers.functional import liger_qwen4_exp_gr_write
from liger_kernel.transformers.functional import liger_qwen4_exp_hyper_connection_pre
from liger_kernel.transformers.functional import liger_qwen4_exp_ngram_hash
device = infer_device()
pytestmark = pytest.mark.skipif(
    device != "cuda" or is_hip(), reason="Qwen4Exp Triton kernels require an NVIDIA CUDA GPU"
)


def qwen4_exp_ngram_hash_ref(shifted_ids, multipliers, vocab_sizes, offsets):
    ngram_size = shifted_ids.shape[-1]
    n_heads = vocab_sizes.numel()
    heads_per_ngram = n_heads // (ngram_size - 1)
    blocks = []
    for current_ngram_size in range(2, ngram_size + 1):
        mixed_ids = shifted_ids[..., 0] * multipliers[0]
        for position in range(1, current_ngram_size):
            mixed_ids = torch.bitwise_xor(mixed_ids, shifted_ids[..., position] * multipliers[position])
        start = (current_ngram_size - 2) * heads_per_ngram
        end = start + heads_per_ngram
        blocks.append(mixed_ids.unsqueeze(-1).remainder(vocab_sizes[start:end]) + offsets[start:end])
    return torch.cat(blocks, dim=-1)


def qwen4_exp_eos_aware_ngram_hash_ref(previous_context, input_ids, multipliers, vocab_sizes, offsets, eos_token_id):
    token_history = torch.cat([previous_context, input_ids], dim=-1)
    positions = torch.arange(token_history.shape[1], device=token_history.device, dtype=torch.long)
    eos_positions = torch.where(token_history == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((token_history.shape[0], 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
    shifted_tokens = []
    for shift in range(previous_context.shape[1] + 1):
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(token_history.shape[0], -1)
        shifted = token_history.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        shifted_tokens.append(torch.where(valid, shifted, eos_token_id))
    shifted_tokens = torch.stack(shifted_tokens, dim=-1)[:, -input_ids.shape[1] :]
    return qwen4_exp_ngram_hash_ref(shifted_tokens, multipliers, vocab_sizes, offsets)


def qwen4_exp_hyper_connection_pre_ref(mix_logits, normalized_input, n_groups):
    hidden_size = normalized_input.shape[-1] // n_groups
    mix = torch.sigmoid(mix_logits).unflatten(-1, (n_groups, hidden_size))
    values = normalized_input.unflatten(-1, (n_groups, hidden_size))
    return (mix * values).mean(dim=-2)


def qwen4_exp_gr_write_ref(block_output, residual, write_logits):
    write_scale = 2 * torch.sigmoid(write_logits)
    injection = block_output.unsqueeze(-2) * write_scale.unsqueeze(-1)
    return residual + injection.flatten(-2)


def qwen4_exp_group_rms_norm_ref(hidden_states, weight, eps, offset, casting_mode, n_groups):
    """Simpler exact baseline using the shared generic grouped-RMS primitive."""
    return LigerRMSNormFunction.apply(
        hidden_states,
        weight,
        eps,
        offset,
        casting_mode,
        False,
        None,
        n_groups,
    )


def qwen4_exp_sum_three_grads(grad0, grad1, grad2):
    """Materialize the explicit left-associated low-precision reference sum."""
    return (grad0 + grad1) + grad2
@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((128, 4, 1024), torch.float32),
        pytest.param(
            (2048, 4, 1024),
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
            id="profile-bf16",
        ),
        pytest.param(
            (512, 4, 2560),
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
            id="production-bf16",
        ),
    ],
)
def test_qwen4_exp_group_rms_norm_add3_backward(shape, dtype):
    set_seed(42)
    rows, n_groups, group_size = shape
    dim = n_groups * group_size
    x = torch.randn((rows, dim), device=device, dtype=dtype, requires_grad=True)
    weight = (torch.randn(dim, device=device, dtype=dtype) * 0.02).requires_grad_(True)
    grads = [torch.randn_like(x) for _ in range(3)]

    outputs = LigerGroupRMSNormFusedFunction.apply(x, weight, 1e-6, 1.0, "gemma", n_groups)
    assert outputs[0].data_ptr() == outputs[1].data_ptr() == outputs[2].data_ptr()
    torch.autograd.backward(outputs, grads)
    actual_dx = x.grad.detach().clone()
    actual_dw = weight.grad.detach().clone()

    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    reference = LigerRMSNormFunction.apply(ref_x, ref_weight, 1e-6, 1.0, "gemma", False, None, n_groups)
    reference.backward(qwen4_exp_sum_three_grads(*grads))

    assert_verbose_allclose(outputs[0], reference, atol=0.0, rtol=0.0)
    assert_verbose_allclose(actual_dx, ref_x.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(actual_dw, ref_weight.grad, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "rows, used_consumers",
    [
        pytest.param(32, (1,), id="only-middle"),
        pytest.param(32, (1, 2), id="missing-first"),
        pytest.param(512, (0, 2), id="missing-middle-512"),
        pytest.param(2048, (0, 2), id="missing-middle-2048"),
        pytest.param(32, (0, 1), id="missing-last"),
    ],
)
def test_qwen4_exp_group_rms_norm_sparse_consumers_match_explicit_zeros(monkeypatch, rows, used_consumers):
    if not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    import liger_kernel.ops.rms_norm as rms_norm_ops

    set_seed(42)
    n_groups, group_size = 4, 2048
    dim = n_groups * group_size
    x = torch.randn((rows, dim), device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    grads = [torch.randn_like(x) for _ in range(3)]

    calls = 0
    original_add2 = rms_norm_ops.rms_group_norm_backward_add2

    def record_add2(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_add2(*args, **kwargs)

    monkeypatch.setattr(rms_norm_ops, "rms_group_norm_backward_add2", record_add2)
    outputs = LigerGroupRMSNormFusedFunction.apply(x, weight, 1e-6, 1.0, "gemma", n_groups)
    torch.autograd.backward(
        tuple(outputs[index] for index in used_consumers),
        tuple(grads[index] for index in used_consumers),
    )
    actual_dx = x.grad.detach().clone()
    actual_dw = weight.grad.detach().clone()
    assert calls == (1 if len(used_consumers) == 2 else 0)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    reference = LigerGroupRMSNormFusedFunction.apply(ref_x, ref_weight, 1e-6, 1.0, "gemma", n_groups)
    explicit_grads = [
        grad if index in used_consumers else torch.zeros_like(grads[0]) for index, grad in enumerate(grads)
    ]
    torch.autograd.backward(reference, tuple(explicit_grads))

    assert outputs[0].data_ptr() == outputs[1].data_ptr() == outputs[2].data_ptr()
    assert_verbose_allclose(outputs[0], reference[0], atol=0.0, rtol=0.0)
    assert_verbose_allclose(actual_dx, ref_x.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(actual_dw, ref_weight.grad, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "shape",
    [
        (2048, 4, 1024),
        (512, 4, 2560),
        (2048, 4, 2560),
    ],
)
def test_qwen4_exp_group_rms_norm_write4_backward(shape):
    if not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    set_seed(42)
    rows, n_groups, group_size = shape
    dim = n_groups * group_size
    x = torch.randn((rows, dim), device=device, dtype=torch.bfloat16, requires_grad=True)
    rms_weight = (torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    write_weight = (torch.randn((n_groups, dim), device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    grad_down = torch.randn_like(x)
    grad_pre = torch.randn_like(x)
    grad_write = torch.randn((rows, n_groups), device=device, dtype=torch.bfloat16)

    norm_down, norm_pre, write_logits = LigerGroupRMSNormWrite4Function.apply(
        x, rms_weight, write_weight, 1e-6, 1.0, "gemma", n_groups
    )
    torch.autograd.backward((norm_down, norm_pre, write_logits), (grad_down, grad_pre, grad_write))
    actual_grads = (x.grad.detach().clone(), rms_weight.grad.detach().clone(), write_weight.grad.detach().clone())

    ref_x = x.detach().clone().requires_grad_(True)
    ref_rms_weight = rms_weight.detach().clone().requires_grad_(True)
    ref_write_weight = write_weight.detach().clone().requires_grad_(True)
    ref_down, ref_write, ref_pre = LigerGroupRMSNormFusedFunction.apply(
        ref_x, ref_rms_weight, 1e-6, 1.0, "gemma", n_groups
    )
    ref_write_logits = torch.mm(ref_write.view(rows, dim), ref_write_weight.transpose(0, 1)) / n_groups
    torch.autograd.backward((ref_down, ref_pre, ref_write_logits), (grad_down, grad_pre, grad_write))

    assert norm_down.data_ptr() == norm_pre.data_ptr()
    assert_verbose_allclose(norm_down, ref_down, atol=0.0, rtol=0.0)
    assert_verbose_allclose(write_logits, ref_write_logits, atol=0.0, rtol=0.0)
    for actual, expected in zip(actual_grads, (ref_x.grad, ref_rms_weight.grad, ref_write_weight.grad)):
        assert_verbose_allclose(actual, expected, atol=4e-3, rtol=0.0)


@pytest.mark.parametrize("rows", [512, 2048])
def test_qwen4_exp_group_rms_norm_benchmark_reference_parity(rows):
    if not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    set_seed(42)
    n_groups, hidden_size = 4, 2048
    dim = n_groups * hidden_size
    x = torch.randn((rows, dim), device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    grads = [torch.randn_like(x) for _ in range(3)]

    outputs = LigerGroupRMSNormFusedFunction.apply(x, weight, 1e-6, 1.0, "gemma", n_groups)
    torch.autograd.backward(outputs, grads)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    reference = qwen4_exp_group_rms_norm_ref(ref_x, ref_weight, 1e-6, 1.0, "gemma", n_groups)
    reference.backward(qwen4_exp_sum_three_grads(*grads))

    assert_verbose_allclose(outputs[0], reference, atol=0.0, rtol=0.0)
    assert_verbose_allclose(x.grad, ref_x.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(weight.grad, ref_weight.grad, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("rows", [512, 2048])
def test_qwen4_exp_group_rms_norm_write4_benchmark_reference_parity(rows):
    if not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    set_seed(42)
    n_groups, hidden_size = 4, 2048
    dim = n_groups * hidden_size
    x = torch.randn((rows, dim), device=device, dtype=torch.bfloat16, requires_grad=True)
    rms_weight = (torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    write_weight = (torch.randn((n_groups, dim), device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    grads = [
        torch.randn_like(x),
        torch.randn_like(x),
        torch.randn((rows, n_groups), device=device, dtype=torch.bfloat16),
    ]

    outputs = LigerGroupRMSNormWrite4Function.apply(x, rms_weight, write_weight, 1e-6, 1.0, "gemma", n_groups)
    torch.autograd.backward(outputs, grads)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_rms_weight = rms_weight.detach().clone().requires_grad_(True)
    ref_write_weight = write_weight.detach().clone().requires_grad_(True)
    ref_down, ref_for_write, ref_pre = LigerGroupRMSNormFusedFunction.apply(
        ref_x, ref_rms_weight, 1e-6, 1.0, "gemma", n_groups
    )
    ref_write_logits = torch.mm(ref_for_write, ref_write_weight.transpose(0, 1)) / n_groups
    torch.autograd.backward((ref_down, ref_pre, ref_write_logits), grads)

    assert_verbose_allclose(outputs[0], ref_down, atol=0.0, rtol=0.0)
    assert_verbose_allclose(outputs[2], ref_write_logits, atol=0.0, rtol=0.0)
    for actual, expected in zip(
        (x.grad, rms_weight.grad, write_weight.grad),
        (ref_x.grad, ref_rms_weight.grad, ref_write_weight.grad),
    ):
        assert_verbose_allclose(actual, expected, atol=4e-3, rtol=0.0)


@pytest.mark.parametrize(
    "used_consumers",
    [
        pytest.param((0,), id="only-down"),
        pytest.param((1,), id="only-pre"),
        pytest.param((2,), id="only-write"),
        pytest.param((0, 1), id="missing-write"),
        pytest.param((0, 2), id="missing-pre"),
        pytest.param((1, 2), id="missing-down"),
    ],
)
def test_qwen4_exp_group_rms_norm_write4_sparse_consumers(monkeypatch, used_consumers):
    if not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    import liger_kernel.ops.qwen4_exp as qwen4_exp_ops

    fused_calls = 0
    original_fused_backward = qwen4_exp_ops._qwen4_exp_group_rms_norm_backward_write4

    def record_fused_backward(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return original_fused_backward(*args, **kwargs)

    monkeypatch.setattr(qwen4_exp_ops, "_qwen4_exp_group_rms_norm_backward_write4", record_fused_backward)
    set_seed(42)
    rows, n_groups, group_size = 32, 4, 64
    dim = n_groups * group_size
    inputs = [
        torch.randn((rows, dim), device=device, dtype=torch.bfloat16, requires_grad=True),
        (torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True),
        (torch.randn((n_groups, dim), device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True),
    ]
    grads = [
        torch.randn_like(inputs[0]),
        torch.randn_like(inputs[0]),
        torch.randn((rows, n_groups), device=device, dtype=torch.bfloat16),
    ]
    outputs = LigerGroupRMSNormWrite4Function.apply(*inputs, 1e-6, 1.0, "gemma", n_groups)
    torch.autograd.backward(
        tuple(outputs[index] for index in used_consumers),
        tuple(grads[index] for index in used_consumers),
    )
    actual = [None if tensor.grad is None else tensor.grad.detach().clone() for tensor in inputs]
    assert fused_calls == 0

    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    ref_down, ref_write, ref_pre = LigerGroupRMSNormFusedFunction.apply(
        reference_inputs[0], reference_inputs[1], 1e-6, 1.0, "gemma", n_groups
    )
    ref_write_logits = torch.mm(ref_write, reference_inputs[2].transpose(0, 1)) / n_groups
    reference_outputs = (ref_down, ref_pre, ref_write_logits)
    torch.autograd.backward(
        tuple(reference_outputs[index] for index in used_consumers),
        tuple(grads[index] for index in used_consumers),
    )
    for value, expected in zip(actual, reference_inputs):
        if expected.grad is None:
            assert value is None
        else:
            assert_verbose_allclose(value, expected.grad, atol=4e-3, rtol=0.0)


@pytest.mark.parametrize(
    "used_consumers", [(3,), (0, 3), (1, 3), (2, 3), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 2, 3)]
)
def test_qwen4_exp_write4_residual_consumers(used_consumers):
    """An unused residual stays undefined; residual-only use leaves Parameter grads undefined."""
    set_seed(43)
    inputs = [
        torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        for shape in ((32, 256), (256,), (4, 256))
    ]
    reference_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    actual = LigerGroupRMSNormWrite4Function.apply(*inputs, 1e-6, 1.0, "gemma", 4, True)
    reference = (*LigerGroupRMSNormWrite4Function.apply(*reference_inputs, 1e-6, 1.0, "gemma", 4), reference_inputs[0])
    gradients = [torch.randn_like(value) for value in reference]
    saved_gradients = [value.clone() for value in gradients]
    for values in (actual, reference):
        torch.autograd.backward(tuple(values[i] for i in used_consumers), tuple(gradients[i] for i in used_consumers))
    for value, expected in zip(actual, reference):
        assert torch.equal(value, expected)
    for value, expected in zip(inputs, reference_inputs):
        assert (value.grad is None) == (expected.grad is None)
        if value.grad is not None:
            assert torch.equal(value.grad, expected.grad)
    for value, expected in zip(gradients, saved_gradients):
        assert torch.equal(value, expected)
def test_qwen4_exp_residual_join_rounds_rms_gradient_before_add():
    """Cancellation must happen after BF16 rounding, not against the FP32 RMS result."""
    set_seed(45)
    x = torch.randn(32, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, device=device, dtype=torch.bfloat16, requires_grad=True)
    write_weight = torch.randn(4, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    outputs = LigerGroupRMSNormWrite4Function.apply(x, weight, write_weight, 1e-6, 1.0, "gemma", 4)
    gradients = tuple(torch.randn_like(value) for value in outputs)
    expected = torch.autograd.grad(outputs, (x, weight, write_weight), gradients)
    residual_gradient = -expected[0]
    fused = LigerGroupRMSNormWrite4Function.apply(x, weight, write_weight, 1e-6, 1.0, "gemma", 4, True)
    actual = torch.autograd.grad(fused, (x, weight, write_weight), (*gradients, residual_gradient))
    assert torch.count_nonzero(expected[0]) > 0
    assert torch.count_nonzero(actual[0]) == 0
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
@pytest.mark.parametrize("shape, ngram_size, heads_per_ngram", [((2, 17), 2, 4), ((3, 11), 4, 3)])
def test_qwen4_exp_ngram_hash(shape, ngram_size, heads_per_ngram):
    set_seed(42)
    n_heads = (ngram_size - 1) * heads_per_ngram
    input_ids = torch.randint(0, 248320, shape, device=device, dtype=torch.long)
    previous_context = torch.randint(0, 248320, (shape[0], ngram_size - 1), device=device, dtype=torch.long)
    eos_token_id = 2
    input_ids[:, ::5] = eos_token_id
    previous_context[:, 0] = eos_token_id
    multipliers = torch.tensor(
        [922337203685477 // (index + 1) | 1 for index in range(ngram_size)], device=device, dtype=torch.long
    )
    vocab_sizes = torch.arange(1009, 1009 + n_heads, device=device, dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), vocab_sizes.cumsum(0)[:-1]])

    output = liger_qwen4_exp_ngram_hash(
        previous_context,
        input_ids,
        multipliers,
        vocab_sizes,
        offsets,
        eos_token_id,
    )
    reference = qwen4_exp_eos_aware_ngram_hash_ref(
        previous_context,
        input_ids,
        multipliers,
        vocab_sizes,
        offsets,
        eos_token_id,
    )
    assert output.dtype == torch.long
    assert torch.equal(output, reference)


def test_qwen4_exp_ngram_hash_rejects_rocm_explicitly(monkeypatch):
    import liger_kernel.ops.qwen4_exp as qwen4_exp_ops

    monkeypatch.setattr(qwen4_exp_ops, "is_hip", lambda: True)
    token_ids = torch.tensor([[1, 2]], device=device, dtype=torch.long)
    metadata = torch.tensor([3, 5], device=device, dtype=torch.long)
    with pytest.raises(ValueError, match="ROCm is not supported"):
        qwen4_exp_ops.qwen4_exp_ngram_hash(
            token_ids[:, :1],
            token_ids,
            metadata,
            metadata,
            metadata,
            eos_token_id=2,
        )


def test_qwen4_exp_ngram_hash_default_fullgraph_and_cuda_graph():
    set_seed(42)
    shape = (2, 17)
    ngram_size = 4
    n_heads = 9
    input_ids = torch.randint(0, 248320, shape, device=device, dtype=torch.long)
    previous_context = torch.randint(0, 248320, (shape[0], ngram_size - 1), device=device, dtype=torch.long)
    input_ids[:, ::5] = 2
    previous_context[:, 0] = 2
    multipliers = torch.tensor(
        [922337203685477 // (index + 1) | 1 for index in range(ngram_size)], device=device, dtype=torch.long
    )
    vocab_sizes = torch.arange(1009, 1009 + n_heads, device=device, dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), vocab_sizes.cumsum(0)[:-1]])
    args = (previous_context, input_ids, multipliers, vocab_sizes, offsets, 2)
    expected = qwen4_exp_eos_aware_ngram_hash_ref(*args)

    torch.compiler.reset()
    compiled = torch.compile(liger_qwen4_exp_ngram_hash, fullgraph=True)
    assert torch.equal(compiled(*args), expected)
    torch.compiler.reset()

    static_args = tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in args)
    liger_qwen4_exp_ngram_hash(*static_args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = liger_qwen4_exp_ngram_hash(*static_args)

    replay_previous = torch.randint_like(previous_context, low=0, high=248320)
    replay_input = torch.randint_like(input_ids, low=0, high=248320)
    replay_previous[:, 1] = 2
    replay_input[:, ::4] = 2
    replay_expected = qwen4_exp_eos_aware_ngram_hash_ref(
        replay_previous,
        replay_input,
        multipliers,
        vocab_sizes,
        offsets,
        2,
    )
    static_args[0].copy_(replay_previous)
    static_args[1].copy_(replay_input)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, replay_expected)
@pytest.mark.parametrize("shape, n_groups", [((2, 13, 256), 4), ((3, 5, 154), 2)])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        torch.float32,
    ],
)
def test_qwen4_exp_hyper_connection_pre_forward_backward(shape, n_groups, dtype):
    set_seed(42)
    mix_logits = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    normalized_input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn((*shape[:-1], shape[-1] // n_groups), device=device, dtype=dtype)

    output = liger_qwen4_exp_hyper_connection_pre(mix_logits, normalized_input, n_groups)
    output.backward(grad_output)
    actual_grads = [tensor.grad.detach().float().clone() for tensor in (mix_logits, normalized_input)]

    ref_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (mix_logits, normalized_input)]
    reference = qwen4_exp_hyper_connection_pre_ref(*ref_inputs, n_groups)
    reference.backward(grad_output)
    reference_grads = [tensor.grad.detach().float() for tensor in ref_inputs]

    tolerance = (8e-3, 2e-3) if dtype == torch.bfloat16 else (2e-5, 2e-5)
    assert_verbose_allclose(output.float(), reference.float(), atol=tolerance[0], rtol=tolerance[1])
    for actual, expected in zip(actual_grads, reference_grads):
        assert_verbose_allclose(actual, expected, atol=tolerance[0], rtol=tolerance[1])


@pytest.mark.parametrize(
    "shape, n_groups",
    [
        ((2, 13, 64), 4),
        ((3, 5, 77), 2),
        ((2, 11, 65), 3),
        ((2, 7, 63), 8),
        pytest.param((1, 2048, 2048), 4, id="production-shape"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        torch.float32,
    ],
)
def test_qwen4_exp_gr_write_forward_backward(shape, n_groups, dtype):
    set_seed(42)
    block_output = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    residual = torch.randn(*shape[:-1], n_groups * shape[-1], device=device, dtype=dtype, requires_grad=True)
    write_logits = torch.randn(*shape[:-1], n_groups, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn_like(residual)

    output = liger_qwen4_exp_gr_write(block_output, residual, write_logits)
    output.backward(grad_output)
    actual_grads = [tensor.grad.detach().float().clone() for tensor in (block_output, residual, write_logits)]

    ref_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (block_output, residual, write_logits)]
    reference = qwen4_exp_gr_write_ref(*ref_inputs)
    reference.backward(grad_output)
    reference_grads = [tensor.grad.detach().float() for tensor in ref_inputs]

    if dtype == torch.bfloat16:
        tolerances = ((2e-2, 2e-2), (1e-3, 1e-3), (0.0, 0.0), (2e-2, 2e-2))
    else:
        tolerances = ((2e-5, 2e-5),) * 4
    for actual, expected, (atol, rtol) in zip(
        (output.float(), *actual_grads), (reference.float(), *reference_grads), tolerances
    ):
        assert_verbose_allclose(actual, expected, atol=atol, rtol=rtol)
