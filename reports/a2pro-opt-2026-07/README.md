# A2.Pro optimization report

Date: 2026-07-12

This report covers OPT-1, OPT-2, OPT-3 and OPT-4. The goal is to show what is implemented, what was demonstrated, and which points of the technical requirements are already closed.

The report is GitHub-ready. All images and data files are stored under `artifacts/`, and all image links are relative.

## Summary

OPT-1 and OPT-2 are implemented in `brain_opt` as a Python optimizer library. There is external evidence from Dmitry Yudin's team on a robotics training task, but the raw logs and exact task description are not in this repository yet.

OPT-3 is implemented in `brain_opt==0.2.0` as a single-process federated optimization simulator. It has a reproducible demo with CSV, JSON, Markdown summary and plots.

OPT-4 is implemented as the `a2_kvant` Python library and as the A2.Pro `quantize` stage. It has two demonstrations: a Qwen3-8B benchmark with resource and quality metrics, and a completed A2.Pro stand run that writes a quantized checkpoint and metrics artifact.

Main caveat: OPT-3 is currently a simulator, not a real multi-device distributed runtime. It is enough for algorithm demonstration, but not enough if acceptance requires actual training across several devices.

## Technical requirements checked

The mapping below is based on these documents:

- `ТЗ 4 Платформа Сгибнев.pdf`, point 3.2.3.7: `A2.Оптимизация` must implement at least one method for optimization, distributed optimization modifications, and efficient storage of large models.
- `3 Дополнение к ТЗ Безносиков`, point 2.5: efficient adaptive optimization, efficient distributed optimization, specialized optimization tasks, and benchmark evaluation.
- `3 Дополнение к ТЗ Безносиков`, points 3.4 and 3.5: experimental code must be delivered as a component, module or library, with configs, logs, metrics and result tables.
- `3 Дополнение к ТЗ Безносиков`, point 3.5.2.1: distributed training components target synchronous or asynchronous training on one or more clusters with more than one device.
- `3 Дополнение к ТЗ Безносиков`, point 3.5.2.6: components must work inside A2.Pro `Оптимизация`.
- `3 Дополнение к ТЗ Безносиков`, point 3.7.2: tests should include baseline comparison, quality metrics and resource metrics where applicable.
- `ТЗ 3 Безносиков.pdf`, point 3.2: quantization and pruning methods based on optimization formulations.
- `ТЗ 3 Безносиков.pdf`, point 3.3: training procedures with quantization.

## Base image integration

The integration design is aligned with the request to call optimization methods from the base image.

The base image `plibs:jaguar-a2pro-optlibs` includes:

- `brain_opt==0.2.0`
- `a2_kvant==0.1.0`

The A2.Pro solution image uses `QUANT_LIB_SOURCE=base` by default. In that mode the image removes local `/app/src/a2_kvant` and checks that imports resolve from the base image:

```python
import a2_kvant
import brain_opt
from brain_opt import get_optimizer, run_fedavg
```

This is the right integration path: the solution code contains the PlatformAPI wrapper, while the optimization methods live in the base image.

Current limitation: after upgrading `brain_opt` to `0.2.0`, the package was rebuilt on `vv_h200`, but the A2.Pro stand package was not re-uploaded. The completed stand run proves OPT-4 and the base-library path, but it does not prove OPT-3 import in the stand UI environment.

## OPT-1: memory-efficient optimizers

Current mapping: OPT-1 is the memory-efficient and adaptive optimizer block.

Implemented in `brain_opt`:

- `SGD`, baseline re-export
- `AdamW`, baseline re-export
- `SignSGD`, sign-based SGD and Signum
- `Lion`, EvoLved Sign Momentum
- `get_optimizer`, common factory
- `scale_lr` and `LR_MULTIPLIERS`, common learning-rate interface

Why this matches the requirements:

- Covers the `ТЗ 4`, point 3.2.3.7 direction on optimization methods for LLM and MLLM training or fine-tuning.
- Covers the supplement, point 2.5, on efficient adaptive optimization techniques.
- Covers the supplement, point 3.4, as an experimental code sample in library form.

### External robotics result

The user reported that Dmitry Yudin's team ran AdamW and Lion on a robotics model-training task. The plot below was supplied as evidence. It shows `eval/mAP2 vs step`.

![Lion vs AdamW on robotics task](artifacts/opt1_lion_vs_adamw_robotics.jpg)

Status of this evidence:

- Useful as a demonstration slide.
- Not enough for a strict report yet because the plot has no legend, no raw logs, no model name, no dataset name and no exact command.
- To make it acceptance-ready, we should request the raw metric export or at least the run names, task description, model, dataset, seed, optimizer hyperparameters and final metric table.

## OPT-2: matrix optimizers

Current mapping: OPT-2 is the matrix and preconditioned optimizer block.

Implemented in `brain_opt`:

- `Muon`, Newton-Schulz orthogonalized momentum with AdamW fallback for 1D parameters, embeddings and lm-head
- `Shampoo`, matrix preconditioning
- `SOAP`, Adam in Shampoo eigenbasis
- `get_optimizer`, common factory
- `LR_MULTIPLIERS`, common learning-rate interface

Why this matches the requirements:

- Covers the `ТЗ 4`, point 3.2.3.7 direction on optimization methods for training and fine-tuning.
- Covers the supplement, point 2.5, on efficient optimization approaches for different training setups.
- Covers the supplement, points 3.4 and 3.5, as a Python library component.

### External robotics result

The user reported that Dmitry Yudin's team ran AdamW and Muon on a robotics model-training task. The plot below was supplied as evidence. It shows `eval/mAP2 vs step`.

![Muon vs AdamW on robotics task](artifacts/opt2_muon_vs_adamw_robotics.jpg)

Status of this evidence:

- Useful as a demonstration slide.
- Not enough for a strict report yet because the plot has no legend, no raw logs, no model name, no dataset name and no exact command.
- To make it acceptance-ready, we need the same metadata as for OPT-1.

## OPT-3: federated and distributed optimization

Implemented in `brain_opt.federated`.

Public API:

```python
FederatedClient
FederatedResult
FedAvgConfig
AsyncConfig
run_fedavg
run_async_sgd
run_async_local_sgd
```

Implemented methods:

- `run_fedavg`: synchronous FedAvg with configurable active client count
- `run_async_sgd`: asynchronous server-side SGD with delayed client gradients
- `run_async_local_sgd`: asynchronous Local SGD with delayed local deltas
- staleness tracking for async updates
- staleness weighting on the server
- simulated latency through `min_delay`, `max_delay` and `max_pending`
- history logging for loss, step, simulated time, staleness and update weight

Verification:

```text
pytest tests/test_federated.py -q
4 passed

python -m pytest -q
19 passed
```

### Demo

Demo script:

```bash
python examples/opt3_federated_demo.py
```

Demo artifacts:

- [`artifacts/opt3_metrics.csv`](artifacts/opt3_metrics.csv)
- [`artifacts/opt3_summary.json`](artifacts/opt3_summary.json)
- [`artifacts/opt3_summary.md`](artifacts/opt3_summary.md)
- [`artifacts/opt3_loss_by_step.png`](artifacts/opt3_loss_by_step.png)
- [`artifacts/opt3_loss_by_time.png`](artifacts/opt3_loss_by_time.png)
- [`artifacts/opt3_staleness_hist.png`](artifacts/opt3_staleness_hist.png)

Results:

- `FedAvg-5clients`: final MSE `0.001011`, improvement `11430.54x`
- `FedAvg-10clients`: final MSE `0.001032`, improvement `11201.83x`
- `FedAvg-20clients`: final MSE `0.001022`, improvement `11309.82x`
- `AsyncSGD`: final MSE `1.074178`, mean staleness `6.83`, max staleness `16`
- `Async-LocalSGD`: final MSE `0.003963`, mean staleness `6.72`, max staleness `14`

![OPT-3 loss by step](artifacts/opt3_loss_by_step.png)

![OPT-3 loss by simulated time](artifacts/opt3_loss_by_time.png)

![OPT-3 async staleness distribution](artifacts/opt3_staleness_hist.png)

Requirement mapping:

- Closes `ТЗ 4`, point 3.2.3.7, for distributed optimization modifications at simulator level.
- Closes the supplement, point 2.5, for efficient distributed optimization methods at simulator level.
- Closes the supplement, point 3.4, as experimental Python code.
- Partially closes the supplement, point 3.7.2, because the demo compares FedAvg, AsyncSGD and Async-LocalSGD by loss, simulated time and staleness.

What is not fully closed:

- It does not satisfy the strict reading of point 3.5.2.1 if actual multi-device training is required.
- It does not yet use CIFAR-10, ResNet-18 or nanoGPT from the distributed-training requirement.
- It is not yet an A2.Pro stage.

## OPT-4: efficient storage through quantization

Implemented as:

- `a2_kvant`, Python quantization library
- A2.Pro `quantize` stage, PlatformAPI wrapper around `a2_kvant`
- base-image integration through `a2_kvant==0.1.0`

Recipes:

- `w8a8`: SmoothQuant plus GPTQ, INT8 weights and INT8 activations
- `w4a16`: GPTQ INT4 weights and FP16 activations
- `fp8`: FP8 dynamic weights and activations

The A2.Pro stage flow:

- reads parameters from PlatformAPI
- gets input checkpoint `in_model`
- downloads model files
- runs `a2_kvant.quantize.quantize_model`
- creates output checkpoint collection `out_model`
- uploads the quantized checkpoint
- writes `out_metrics` as an artifact
- publishes stage progress

### Demo 1: Qwen3-8B benchmark

Model: `Qwen/Qwen3-8B`

Recipe: `w8a8`

Artifacts:

- [`artifacts/opt4_run_config.json`](artifacts/opt4_run_config.json)
- [`artifacts/opt4_compare_vllm.json`](artifacts/opt4_compare_vllm.json)
- [`artifacts/opt4_qwen3_w8a8_comparison.png`](artifacts/opt4_qwen3_w8a8_comparison.png)

Results:

- FP16 disk size: `15.26 GiB`
- W8A8 disk size: `8.79 GiB`
- FP16 VRAM for weights: `15.27 GiB`
- W8A8 VRAM for weights: `8.80 GiB`
- FP16 generation speed: `78.16 tok/s`
- W8A8 generation speed: `114.02 tok/s`
- FP16 WikiText-2 perplexity: `8.622`
- W8A8 WikiText-2 perplexity: `8.533`
- FP16 HellaSwag acc_norm: `0.7500`
- W8A8 HellaSwag acc_norm: `0.7542`
- FP16 GSM8K strict: `0.906`
- W8A8 GSM8K strict: `0.910`

![Qwen3-8B FP16 vs W8A8](artifacts/opt4_qwen3_w8a8_comparison.png)

### Demo 2: A2.Pro stand run

Stand run metadata:

- run: `61af8071-adb5-4dbe-8c06-8200bfba3c66`
- project: `3d07ae35-3d92-4af5-ba0d-732f00664959`
- experiment: `58d8b9cc-6d23-4cc5-9136-832cb52e76a7`
- stage: `295120f3-ef63-4c37-9240-832a462eeb8b`
- image: `a2pro-quantization:v1.0.16-base-optlibs`
- input checkpoint: `172aabea-8fac-4c41-a17a-0824ddc57170`
- output checkpoint collection: `0b535f00-85fa-414c-af27-9efe8a6ad9c7`
- metrics artifact: `dbbf6083-1224-4159-a450-c1a9a4fdd65c`

Metrics:

- input size: `2.8861 GiB`
- output disk size: `1.6671 GiB`
- compression ratio: `1.7312`

Artifact:

- [`artifacts/opt4_stand_metrics.json`](artifacts/opt4_stand_metrics.json)

Internal stand API link:

```text
https://10.0.116.16:31885/api/v1.10/runs/61af8071-adb5-4dbe-8c06-8200bfba3c66/
```

What the stand run proves:

- RustFS and PlatformAPI checkpoint read blocker did not reproduce.
- Stage read the input checkpoint.
- Stage quantized the model.
- Stage uploaded a new checkpoint collection.
- Stage wrote a metrics artifact.
- Run finished with `COMPLETED` and progress `100`.

Known limitation:

The stand run caught evaluation as `evaluation_error`:

```text
HfUriError: Invalid HF URI 'hf://datasets/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3/.huggingface.yaml'
```

The run proves quantization, checkpoint IO and resource metrics. It does not provide a clean stand-side perplexity metric yet. To fix this, rebuild the base image with the updated `a2_kvant` wheel and rerun.

Requirement mapping:

- Closes `ТЗ 4`, point 3.2.3.7, for efficient storage of large models.
- Closes `ТЗ 3`, point 3.2, for quantization based on optimization formulations.
- Closes the supplement, point 3.5.2.6, through the A2.Pro Format 2 stage.
- Closes the supplement, point 3.7.2, for resource metrics.
- Closes quality metrics in the Qwen3-8B benchmark, but only partially on the stand because of the evaluation error.

What is not fully closed:

- Pruning is not implemented.
- `ТЗ 3`, point 3.3, is only partially covered. Current code does post-training quantization, not quantization inside the training loop.

## What to show

For OPT-1:

- Show the Lion vs AdamW robotics plot.
- Say that this is an external run from Dmitry Yudin's team.
- Do not claim acceptance-level evidence until raw logs and task metadata are received.

For OPT-2:

- Show the Muon vs AdamW robotics plot.
- Say the same caveat about missing raw logs and metadata.

For OPT-3:

- Show `opt3_summary.md`.
- Show the three plots: loss by step, loss by simulated time, staleness histogram.
- Say directly that it is a simulator-level demo.

For OPT-4:

- Show the A2.Pro run `61af8071-adb5-4dbe-8c06-8200bfba3c66`.
- Show output checkpoint `0b535f00-85fa-414c-af27-9efe8a6ad9c7`.
- Show metrics artifact `dbbf6083-1224-4159-a450-c1a9a4fdd65c`.
- Show the Qwen3-8B comparison graph.

## What should be run next

### OPT-1 and OPT-2

First, ask Dmitry's team for the raw run metadata. This is faster and more valuable than starting a new benchmark immediately.

Needed fields:

- task name
- model architecture
- dataset
- metric definition for `eval/mAP2`
- optimizer hyperparameters
- seed
- final metric table
- raw CSV or W&B export
- mapping from plot color to optimizer

If an independent benchmark is still needed, the best next run is not plain HellaSwag. HellaSwag is an evaluation task, not an optimizer-training task by itself. A better optimizer demo is:

- fine-tune a small language model with AdamW, Lion and Muon
- evaluate the resulting checkpoints on HellaSwag and optionally GSM8K
- report training loss, validation loss, HellaSwag accuracy, wall-clock time and peak memory

Cheaper alternative:

- train nanoGPT or a small Transformer on a small text corpus
- compare AdamW, Lion and Muon by validation loss, time and memory

### OPT-3

The strongest next demo for the current TЗ is CIFAR-10 with a ResNet-style model and federated client splits.

Recommended path:

1. Add `examples/opt3_cifar_federated_demo.py`.
2. Use CIFAR-10 split into clients.
3. Train a small CNN or ResNet-18-style model with FedAvg, AsyncSGD and Async-LocalSGD.
4. Save accuracy, loss, communication steps, simulated time and staleness plots.
5. Package the same demo as an A2.Pro stage that writes metrics and plots as artifacts.

This directly targets the distributed-training requirement that mentions CIFAR-10 and ResNet-18. It also gives a cleaner story than synthetic regression.

If we need to show use of other A2.Pro modules, the practical version is:

- take dataset input through the platform dataset or artifact interface
- take optional initial checkpoint through the platform checkpoint interface
- run federated training inside the OPT-3 stage
- write output model checkpoint and metrics artifacts back to A2.Pro

That would make OPT-3 look like a real platform component rather than a local library demo.

## Honest status

We can say that OPT-1, OPT-2, OPT-3 and OPT-4 are wrapped as Python libraries and can be included in the base image.

We can say that OPT-4 has a completed A2.Pro stand run with output checkpoint and metrics artifact.

We can say that OPT-3 has a reproducible demo with plots and metrics.

We should not say that OPT-3 is already verified on multiple GPUs or clusters.

We should not say that OPT-1 and OPT-2 have acceptance-level benchmark evidence until the raw robotics run logs are received or we run a new benchmark ourselves.

We should not say that the OPT-4 stand run has clean perplexity until the evaluation wheel is rebuilt and the stage is rerun.
