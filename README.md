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
| SoftSignum | `brain_opt.SoftSignum` | Signum with a temperature-controlled `tanh` instead of `sign` (BrainLab). |
| SoftMuon | `brain_opt.SoftMuon` | Muon with soft saturation of singular values (BrainLab). **Matrix method.** |

Federated / distributed optimization simulators are available under the
same pure-Python package:

| method            | function                         | purpose |
|-------------------|----------------------------------|---------|
| FedAvg            | `brain_opt.run_fedavg`           | synchronous federated averaging with configurable client sampling |
| AsyncSGD          | `brain_opt.run_async_sgd`        | delayed client gradients with server-side staleness weighting |
| Async-Local SGD   | `brain_opt.run_async_local_sgd`  | delayed local-SGD deltas with simulated latency/staleness |
| QSGD / MQSGD      | `brain_opt.run_qsgd`             | compressed SGD with Rand-m or Markovian (BanLast, KAWASAKI) coordinate selection |
| AMQSGD            | `brain_opt.run_amqsgd`           | accelerated compressed SGD with the same compressors |

Parameter-efficient language model training (full fine-tuning, LoRA,
SimplexLoRA, WeightLoRA) is available as `brain_opt.training` and the
`brain-opt train` command; it needs the `training` extra.

## Install

```bash
# From GitHub (recommended for now — works in any base image)
pip install git+https://github.com/Vepricov/brain-opt.git

# From a built wheel (drop into a base image without network)
pip install /path/to/brain_opt-0.3.0-py3-none-any.whl

# From a checkout
pip install -e .
```

The built wheel is `brain_opt-0.3.0-py3-none-any.whl` — a universal pure
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

## Federated optimization simulators

`brain_opt.federated` covers the distributed-optimization part of the
package without requiring a real cluster. Clients are local PyTorch
tensors; latency and staleness are simulated in one Python process. This
is intended for reproducible method comparisons before wiring the methods
into a real scheduler.

```python
import torch
import torch.nn as nn

from brain_opt import (
    AsyncConfig,
    FedAvgConfig,
    FederatedClient,
    run_async_local_sgd,
    run_async_sgd,
    run_fedavg,
)

clients = [
    FederatedClient(x=torch.randn(32, 8), y=torch.randn(32, 1), name=f"c{i}")
    for i in range(20)
]
model = nn.Linear(8, 1)

# FedAvg with 5 active clients per round.
fedavg = run_fedavg(
    model,
    clients,
    FedAvgConfig(rounds=20, clients_per_round=5, local_steps=3, lr=0.05),
)

# Asynchronous updates with simulated delay and staleness weighting.
async_sgd = run_async_sgd(
    nn.Linear(8, 1),
    clients,
    AsyncConfig(updates=100, lr=0.02, max_pending=8, max_delay=5),
)
async_local = run_async_local_sgd(
    nn.Linear(8, 1),
    clients,
    AsyncConfig(updates=100, local_steps=3, lr=0.05, max_pending=8),
)
```

Every result exposes `result.model` and a list of metric dictionaries in
`result.history`. Async histories include `time`, `staleness` and
`staleness_weight`, which are the main quantities to plot against loss or
accuracy in a simulation study.

## BrainLab methods

The package also ships methods developed at BrainLab.

### SoftSignum and SoftMuon

`SoftSignum` replaces `sign(m)` in Signum with `tanh(tau * m)`. The
temperature is chosen every step so that a given share of coordinates
stays in the linear regime; `alpha_sign` sets the fraction of training
spent in the pure sign regime. `SoftMuon` applies the same idea to the
singular values of the Muon momentum and is computed with Newton-Schulz
iterations, without an SVD.

```python
opt = brain_opt.SoftSignum(model.parameters(), lr=1e-3, total_steps=10_000)
opt = brain_opt.get_optimizer(model.parameters(), name="SoftMuon", lr=1e-3)
```

Code and experiments: https://github.com/brain-lab-research/softsign

### Markovian compression (MQSGD, AMQSGD)

Clients send a share `ratio` of gradient coordinates per round.
`compressor="randm"` picks them uniformly at random (QSGD with Rand-m).
`"banlast"` forbids coordinates sent during the last `history` rounds,
`"kawasaki"` divides the probability of a coordinate by `forgetting` for
every recent send. Both Markovian compressors remain unbiased in the
long run.

```python
from brain_opt import CompressedConfig, run_amqsgd, run_qsgd

cfg = CompressedConfig(rounds=100, lr=0.1, ratio=0.05,
                       compressor="kawasaki", history=3, forgetting=2.0)
mqsgd = run_qsgd(nn.Linear(8, 1), clients, cfg)
amqsgd = run_amqsgd(nn.Linear(8, 1), clients, cfg)
```

### SimplexLoRA and WeightLoRA

`TrainingConfig(mode=...)` accepts `"full"`, `"lora"`, `"simplexlora"`
and `"weightlora"`. SimplexLoRA learns adapter importance weights on a
simplex during a short search and redistributes the LoRA rank budget
between layers. WeightLoRA keeps only the `keep` most important adapters
after `selection_steps` steps and trains them as plain LoRA.

```bash
pip install -e ".[training]"
brain-opt train config.json
```

`config.json` holds `base_checkpoint`, `train_text`, `validation_text`,
`output_dir` and a `config` object with the `TrainingConfig` fields; the
nested `simplex` or `weight_lora` objects take the fields of
`SimplexLoRAConfig` and `WeightLoRAConfig`.

Code and experiments: https://github.com/brain-lab-research/SimplexLoRA,
https://github.com/brain-lab-research/WLoRA

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

The tests cover convergence of every optimizer through the factory at
`lr=1e-3`, the `scale_lr` mapping, factory name resolution, the public API,
the federated and compressed simulators, SoftSignum and SoftMuon, and an
offline LoRA / SimplexLoRA / WeightLoRA training round trip on a tiny
random model.

## License

Apache-2.0.
