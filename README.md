# brain_opt

Drop-in PyTorch optimizer collection. **Pure Python**, single runtime
dependency (`torch>=1.10`), no `torch.distributed` requirements, no
`@torch.compile`, no custom CUDA kernels — drops cleanly into any base
image as `site-packages`.

| name     | class               | category                                                          |
|----------|---------------------|-------------------------------------------------------------------|
| SGD      | `brain_opt.SGD`     | re-export of `torch.optim.SGD`                                    |
| AdamW    | `brain_opt.AdamW`   | re-export of `torch.optim.AdamW`                                  |
| SignSGD  | `brain_opt.SignSGD` | sign-based SGD / Signum. **Memory-efficient for LLM / MLLM.**     |
| Lion     | `brain_opt.Lion`    | EvoLved Sign Momentum, Chen et al. 2023. **Memory-efficient for LLM / MLLM.** |
| Muon     | `brain_opt.Muon`    | D-Muon (Liu et al. 2025): Newton-Schulz orthogonalised momentum + decoupled WD, with AdamW fallback for 1-D params. **Matrix method.** |
| Shampoo  | `brain_opt.Shampoo` | canonical Algorithm 2 from Gupta-Koren-Singer 2018. **Matrix method.** |
| SOAP     | `brain_opt.SOAP`    | Adam in Shampoo's eigenbasis, Vyas et al. 2024. **Matrix method.** |

## Install

```bash
pip install brain_opt                                  # once released
pip install /path/to/brain_opt-0.1.0-py3-none-any.whl  # offline / base image
pip install -e .                                       # from a checkout
```

The built wheel is `brain_opt-0.1.0-py3-none-any.whl` — a universal pure
Python 3 wheel. You can copy it into your base image and `pip
install` it, or unpack the `brain_opt/` directory directly into your
project's `site-packages`. No compilation, no native extensions, no
optional deps. Works with any torch ≥ 1.10 (the API surface used is
`torch.optim.Optimizer`, `torch.linalg.{svd,eigh,qr}` and standard tensor
ops, all stable across modern releases).

## Usage — drop-in style

```python
import brain_opt

opt = brain_opt.Lion(model.parameters(), lr=3e-4, weight_decay=0.1)
opt.zero_grad()
loss.backward()
opt.step()
```

Same call shape as `torch.optim.AdamW(...)`. Hyperparameters keep their
conventional names (`lr`, `betas`, `weight_decay`, `eps`, ...). **Direct
constructors do *not* rescale `lr`** — they forward whatever you pass,
matching the `torch.optim` convention.

## Usage — factory with unified `lr=1e-3`

```python
from brain_opt import get_optimizer

# One reference lr that works across every method:
opt = get_optimizer(model.parameters(), name="Lion",    lr=1e-3)  # -> Lion lr=1e-4
opt = get_optimizer(model.parameters(), name="Muon",    lr=1e-3)  # -> Muon lr=2e-2
opt = get_optimizer(model.parameters(), name="Shampoo", lr=1e-3)  # -> Shampoo lr=1e-1
opt = get_optimizer(model.parameters(), name="SOAP",    lr=1e-3)  # -> SOAP lr=3e-3

# Opt out:
opt = get_optimizer(model.parameters(), name="Lion",
                    lr=5e-4, auto_scale_lr=False)                 # -> Lion lr=5e-4
```

`get_optimizer` accepts case- and separator-insensitive names: `SGD`,
`AdamW` (alias `Adam`), `SignSGD` (alias `Signum`), `Lion`, `Muon`,
`Shampoo`, `SOAP`. Unknown names raise `KeyError`.

### Why the rescaling?

Optimal `lr` varies by 2–3 orders of magnitude across optimizer
families. To keep a single user-facing knob, the factory multiplies your
`lr` by the per-method calibration in `brain_opt.LR_MULTIPLIERS`:

| method   | multiplier | resulting lr for user `1e-3` |
|----------|-----------:|-----------------------------:|
| SGD      |       10.0 | 1e-2                         |
| AdamW    |        1.0 | 1e-3                         |
| SignSGD  |        1.0 | 1e-3                         |
| Lion     |        1.0 | 1e-3                         |
| Muon     |       10.0 | 1e-2                         |
| Shampoo  |      100.0 | 1e-1                         |
| SOAP     |        3.0 | 3e-3                         |

> **Lion note.** Chen et al. (2023) report that on *large* Transformer
> training Lion's optimum is 3–10× smaller than AdamW (~1e-4 for AdamW's
> 1e-3). On small models / short runs Lion converges much faster at lr
> matching AdamW, so the default multiplier here is `1.0`. For LLM-scale
> training pass `auto_scale_lr=False, lr=1e-4..3e-4` explicitly, or
> override at runtime: `brain_opt.LR_MULTIPLIERS["Lion"] = 0.1`.

The scaling function is also exported directly:

```python
from brain_opt import scale_lr
scale_lr("Lion", 1e-3)     # 0.0001
scale_lr("Shampoo", 1e-3)  # 0.1
```

## Reference constructor defaults (no scaling applied)

```python
brain_opt.SGD(params, lr=..., momentum=0.0, weight_decay=0.0, nesterov=False)
brain_opt.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, ...)
brain_opt.SignSGD(params, lr=1e-3, momentum=0.0, weight_decay=0.0, nesterov=False)
brain_opt.Lion(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0)
brain_opt.Muon(params, lr=0.02, weight_decay=0.0, momentum=0.95, nesterov=True,
               ns_steps=5, adamw_betas=(0.9, 0.95), adamw_eps=1e-8,
               adamw_lr_ratio=1.0)
brain_opt.Shampoo(params, lr=1e-1, momentum=0.0, weight_decay=0.0,
                  epsilon=1e-4, update_freq=1)
brain_opt.SOAP(params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=-1.0,
               eps=1e-8, weight_decay=0.01,
               precondition_frequency=10, max_precond_dim=10000,
               merge_dims=False, precondition_1d=False,
               normalize_grads=False, data_format="channels_first",
               correct_bias=True)
```

## Memory-efficient methods for LLM / MLLM

`SignSGD` and `Lion` are the recommended drop-in replacements for AdamW
when memory is the bottleneck:

* AdamW stores 2 moments per parameter (~ 2× param memory)
* Lion stores 1 moment per parameter (~ 1× param memory)
* SignSGD with `momentum=0` stores nothing (~ 0× param memory)

## Matrix methods

`Muon`, `Shampoo` and `SOAP` precondition matrix-shaped parameters with
second-order information. They are more expensive per step but converge
in fewer steps on language and vision models.

For Muon, 1-D parameters (layer-norms, biases) and embedding / lm-head
matrices automatically fall back to AdamW. You can override the
classification by setting `state[p]["use_muon"]` after construction, or
by passing `"use_muon": False` inside a parameter group.

## Sources / credit

* Lion — https://github.com/google/automl/tree/master/lion
* Muon — https://github.com/KellerJordan/Muon, D-Muon details from
  Liu et al., "Muon is Scalable for LLM Training", arXiv:2502.16982
  (https://github.com/MoonshotAI/Moonlight)
* Shampoo — https://github.com/moskomule/shampoo.pytorch (port shipped by
  `jettify/pytorch-optimizer`), Algorithm 2 of arXiv:1802.09568
* SOAP — https://github.com/nikhilvyas/SOAP, arXiv:2409.11321

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

15 smoke tests cover convergence of every method through the factory at
`lr=1e-3`, the `scale_lr` mapping, factory name resolution and the
public API.

## License

Apache-2.0.
