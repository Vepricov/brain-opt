"""Focused CPU tests for causal per-update KL-matched SOAP."""

import copy
import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = Path(__file__).parents[2] / "verl" / "utils" / "kl_matched_soap.py"
SPEC = importlib.util.spec_from_file_location("kl_matched_soap_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
kl_matched_soap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kl_matched_soap
SPEC.loader.exec_module(kl_matched_soap)
KLMatchedSOAP = kl_matched_soap.KLMatchedSOAP
categorical_fisher_quadratic = kl_matched_soap.categorical_fisher_quadratic
exact_logits_jvp = kl_matched_soap.exact_logits_jvp
ExactLogitsJVPFisher = kl_matched_soap.ExactLogitsJVPFisher
matched_alpha = kl_matched_soap.matched_alpha
partition_actor_parameters = kl_matched_soap.partition_actor_parameters


def _named_parameters():
    return [
        ("model.layers.0.self_attn.q_proj.weight", torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))),
        ("model.layers.0.mlp.down_proj.weight", torch.nn.Parameter(torch.tensor([[2.0, 0.0], [-1.0, 1.0]]))),
        ("model.layers.0.input_layernorm.weight", torch.nn.Parameter(torch.tensor([1.0, 1.0]))),
        ("model.embed_tokens.weight", torch.nn.Parameter(torch.tensor([[0.2, 0.3], [0.4, 0.5]]))),
    ]


def _optimizer(named=None, **kwargs):
    named = _named_parameters() if named is None else named
    optimizer = KLMatchedSOAP(
        named,
        lr=0.01,
        weight_decay=0.1,
        eps=1e-5,
        soap_precondition_frequency=2,
        soap_max_precond_dim=8,
        fisher_prompt_indices=range(16),
        **kwargs,
    )
    optimizer.bind_fisher_evaluator(
        lambda directions: 0.5 * sum(float(direction.double().square().sum()) for direction in directions.values()),
        {"policy": "test", "indices": list(range(16)), "sha256": "fixed"},
    )
    return optimizer, named


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            _assert_nested_equal(lhs, rhs)
    else:
        assert left == right


def test_full_vocabulary_fisher_quadratic_matches_explicit_categorical_formula():
    logits = torch.tensor([[[1.2, -0.4, 0.7], [4.0, 1.0, -3.0]]], dtype=torch.float64)
    tangent = torch.tensor([[[0.3, -0.8, 1.1], [99.0, 99.0, 99.0]]], dtype=torch.float64)
    mask = torch.tensor([[True, False]])
    probabilities = logits[0, 0].softmax(dim=-1)
    expected = 0.5 * ((probabilities * tangent[0, 0].square()).sum() - (probabilities * tangent[0, 0]).sum().square())
    actual = categorical_fisher_quadratic(logits, tangent, mask)
    assert actual.item() == pytest.approx(expected.item(), rel=2e-6)
    # A constant shift of every vocabulary logit is in the categorical Fisher nullspace.
    assert categorical_fisher_quadratic(logits[:, :1], torch.ones_like(tangent[:, :1]), mask[:, :1]).item() == 0


def test_functional_forward_mode_logits_jvp_is_exact_and_nonmutating():
    module = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.3, -0.2]], dtype=torch.float64))
    direction = torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.7, 0.8]], dtype=torch.float64)
    inputs = torch.tensor([[2.0, -3.0]], dtype=torch.float64)
    weight_before = module.weight.detach().clone()

    logits, tangent = exact_logits_jvp(module, ("weight",), (module.weight,), (direction,), {"input": inputs})

    assert torch.equal(logits, inputs @ module.weight.T)
    assert torch.equal(tangent, inputs @ direction.T)
    assert not logits.requires_grad
    assert not tangent.requires_grad
    assert torch.equal(module.weight, weight_before)


def test_exact_logits_jvp_fisher_runs_through_real_one_rank_fsdp_on_cpu():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    class ToyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Linear(2, 3, bias=False)

        def forward(self, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            values = torch.nn.functional.one_hot(input_ids, num_classes=2).float()
            return SimpleNamespace(logits=self.self_attn(values))

    descriptor, rendezvous = tempfile.mkstemp(prefix="kl-matched-soap-fsdp-")
    os.close(descriptor)
    owns_group = not torch.distributed.is_initialized()
    try:
        if owns_group:
            torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
        model = ToyCausalLM()
        with torch.no_grad():
            model.self_attn.weight.copy_(torch.tensor([[1.0, -0.5], [0.2, 0.7], [-0.3, 0.4]]))
        wrapped = FSDP(model, use_orig_params=True, device_id=torch.device("cpu"))
        parameter = next(wrapped.parameters())
        direction = torch.tensor([[0.1, 0.2], [-0.4, 0.3], [0.5, -0.6]])
        inputs = torch.tensor([[0, 1]])
        mask = torch.tensor([[True, True]])
        evaluator = ExactLogitsJVPFisher(wrapped, inputs, torch.ones_like(inputs), mask)

        actual = evaluator({parameter: direction})

        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
        tangent = torch.nn.functional.one_hot(inputs, num_classes=2).float() @ direction.T
        expected = categorical_fisher_quadratic(logits, tangent, mask).item()
        assert actual == pytest.approx(expected, rel=1e-6)
    finally:
        if owns_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if os.path.exists(rendezvous):
            os.remove(rendezvous)


def test_exact_logits_jvp_fisher_runs_through_nested_one_rank_fsdp_on_cpu(monkeypatch):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    class ToyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Linear(2, 3, bias=False)

        def forward(self, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            values = torch.nn.functional.one_hot(input_ids, num_classes=2).float()
            return SimpleNamespace(logits=self.self_attn(values))

    descriptor, rendezvous = tempfile.mkstemp(prefix="kl-matched-soap-nested-fsdp-")
    os.close(descriptor)
    owns_group = not torch.distributed.is_initialized()
    try:
        if owns_group:
            torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
        model = ToyCausalLM()
        with torch.no_grad():
            model.self_attn.weight.copy_(torch.tensor([[1.0, -0.5], [0.2, 0.7], [-0.3, 0.4]]))
        model.self_attn = FSDP(model.self_attn, use_orig_params=True, device_id=torch.device("cpu"))
        nested_fsdp = model.self_attn
        nested_unwrapped = nested_fsdp._fsdp_wrapped_module
        wrapped = FSDP(model, use_orig_params=True, device_id=torch.device("cpu"))
        parameter = next(wrapped.parameters())
        parameter_before = parameter.detach().clone()
        module_tree_before = tuple((name, id(child)) for name, child in wrapped.named_modules())
        direction = torch.tensor([[0.1, 0.2], [-0.4, 0.3], [0.5, -0.6]])
        inputs = torch.tensor([[0, 1]])
        mask = torch.tensor([[True, True]])
        evaluator = ExactLogitsJVPFisher(wrapped, inputs, torch.ones_like(inputs), mask)
        observed_jvps = []

        def record_exact_jvp(*args, **kwargs):
            result = exact_logits_jvp(*args, **kwargs)
            observed_jvps.append(result[1])
            return result

        monkeypatch.setattr(kl_matched_soap, "exact_logits_jvp", record_exact_jvp)

        failing_evaluator = ExactLogitsJVPFisher(wrapped, inputs, torch.ones_like(inputs), mask[:, :1])
        with pytest.raises(ValueError, match="Fisher mask"):
            failing_evaluator({parameter: direction})
        assert wrapped._fsdp_wrapped_module.self_attn is nested_fsdp
        assert nested_fsdp._fsdp_wrapped_module is nested_unwrapped
        assert tuple((name, id(child)) for name, child in wrapped.named_modules()) == module_tree_before

        actual = evaluator({parameter: direction})

        with FSDP.summon_full_params(wrapped, recurse=True, writeback=False):
            logits = wrapped(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
        tangent = torch.nn.functional.one_hot(inputs, num_classes=2).float() @ direction.T
        expected = categorical_fisher_quadratic(logits, tangent, mask).item()
        assert torch.equal(observed_jvps[-1], tangent)
        assert actual == pytest.approx(expected, rel=1e-6)
        assert torch.equal(parameter, parameter_before)
        assert wrapped._fsdp_wrapped_module.self_attn is nested_fsdp
        assert tuple((name, id(child)) for name, child in wrapped.named_modules()) == module_tree_before
    finally:
        if owns_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if os.path.exists(rendezvous):
            os.remove(rendezvous)


def test_proposal_does_not_mutate_parameters_gradients_or_live_optimizer_state():
    optimizer, named = _optimizer()
    for index, (_, parameter) in enumerate(named):
        parameter.grad = torch.full_like(parameter, 0.2 + index * 0.1)
    parameters_before = [parameter.detach().clone() for _, parameter in named]
    assert all(parameter.grad is not None for _, parameter in named)
    gradients_before = [parameter.grad.clone() for _, parameter in named if parameter.grad is not None]
    state_before = copy.deepcopy(optimizer.state_dict())

    proposal = optimizer.propose()

    assert proposal.generation == 0
    for (_, parameter), value, gradient in zip(named, parameters_before, gradients_before, strict=True):
        assert torch.equal(parameter, value)
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, gradient)
    _assert_nested_equal(optimizer.state_dict(), state_before)


def test_commit_advances_both_proposal_states_exactly_once_and_rejects_reuse():
    optimizer, named = _optimizer()
    for _, parameter in named:
        parameter.grad = torch.full_like(parameter, 0.25)
    proposal = optimizer.propose()
    soap_parameter = named[0][1]
    expected = soap_parameter.detach() + 1.5 * proposal.soap_directions[soap_parameter]
    optimizer.commit(proposal, alpha=1.5)
    assert torch.allclose(soap_parameter, expected)
    assert optimizer.state[soap_parameter]["soap_step"] == 1
    assert optimizer.state[soap_parameter]["adamw_step"] == 1
    with pytest.raises(RuntimeError, match="stale|already-committed"):
        optimizer.commit(proposal, alpha=1.5)
    assert optimizer.state[soap_parameter]["soap_step"] == 1
    assert optimizer.state[soap_parameter]["adamw_step"] == 1


def test_step_checkpoint_restore_preserves_shadow_soap_alpha_and_prompt_identity():
    optimizer, named = _optimizer()
    for index, (_, parameter) in enumerate(named):
        parameter.grad = torch.full_like(parameter, 0.1 + index * 0.05)
    optimizer.step()
    assert all(parameter.grad is None for _, parameter in named)
    checkpoint = copy.deepcopy(optimizer.state_dict())

    restored_named = [(name, torch.nn.Parameter(parameter.detach().clone())) for name, parameter in named]
    restored, _ = _optimizer(restored_named)
    restored.load_state_dict(checkpoint)

    assert restored._update_generation == 1
    assert restored.latest_telemetry == optimizer.latest_telemetry
    assert restored.state_dict()["kl_matched_soap"]["prompt_identity"]["sha256"] == "fixed"
    for source, target in zip(optimizer.state.values(), restored.state.values(), strict=True):
        _assert_nested_equal(source, target)

    for (_, source), (_, target) in zip(named, restored_named, strict=True):
        source.grad = torch.full_like(source, 0.37)
        target.grad = source.grad.clone()
    source_proposal = optimizer.propose()
    restored_proposal = restored.propose()
    for source, target in zip(
        source_proposal.soap_directions.values(), restored_proposal.soap_directions.values(), strict=True
    ):
        assert torch.equal(source, target)
    for source, target in zip(
        source_proposal.adamw_directions.values(), restored_proposal.adamw_directions.values(), strict=True
    ):
        assert torch.equal(source, target)


def test_actor_auxiliary_update_is_exact_adamw_and_is_not_alpha_scaled():
    optimizer, named = _optimizer()
    auxiliary = named[2][1]
    reference = torch.nn.Parameter(auxiliary.detach().clone())
    reference_optimizer = torch.optim.AdamW([reference], lr=0.01, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-5)
    for _, parameter in named:
        parameter.grad = torch.full_like(parameter, 0.25)
    assert auxiliary.grad is not None
    reference.grad = auxiliary.grad.clone()
    proposal = optimizer.propose()
    optimizer.commit(proposal, alpha=7.0)
    reference_optimizer.step()
    assert torch.allclose(auxiliary, reference, rtol=1e-6, atol=1e-7)


def test_restore_fails_closed_when_pinned_prompt_identity_changes():
    optimizer, _ = _optimizer()
    checkpoint = optimizer.state_dict()
    restored, _ = _optimizer()
    restored._prompt_identity = {"policy": "test", "indices": list(range(16)), "sha256": "different"}
    with pytest.raises(RuntimeError, match="prompt identity"):
        restored.load_state_dict(checkpoint)


def test_actor_ownership_is_disjoint_and_exhaustive():
    named = _named_parameters()
    soap, auxiliary = partition_actor_parameters(named)
    soap_names = {name for name, _ in soap}
    auxiliary_names = {name for name, _ in auxiliary}
    assert soap_names.isdisjoint(auxiliary_names)
    assert soap_names | auxiliary_names == {name for name, _ in named}
    assert soap_names == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }


@pytest.mark.parametrize("q_adamw,q_soap", [(0.0, 1.0), (1.0, 0.0), (math.inf, 1.0), (1.0, math.nan)])
def test_alpha_fails_closed_for_zero_or_nonfinite_quadratics(q_adamw, q_soap):
    with pytest.raises(FloatingPointError):
        matched_alpha(q_adamw, q_soap, minimum=0.5, maximum=2.0, clamp=True)


def test_alpha_clamps_explicitly_or_fails_closed_when_disabled():
    assert matched_alpha(100.0, 1.0, minimum=0.5, maximum=2.0, clamp=True) == (2.0, 10.0)
    with pytest.raises(FloatingPointError, match="clamping disabled"):
        matched_alpha(100.0, 1.0, minimum=0.5, maximum=2.0, clamp=False)
