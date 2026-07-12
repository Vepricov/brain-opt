# OPT-1/2 Vision Hard-8 Benchmark

Config: synthetic CIFAR-shaped classification, 5 seeds, 8 optimizer steps, batch size 64, 512 samples seen per run, LR grid per optimizer.

## Mean of best LR per optimizer

- `Muon`: mean_acc=0.8499, std=0.1015, min=0.7480, max=0.9907, mean_loss=1.1303
- `AdamW`: mean_acc=0.4245, std=0.1165, min=0.2002, max=0.5239, mean_loss=1.9646
- `Lion`: mean_acc=0.2696, std=0.0597, min=0.1997, max=0.3481, mean_loss=2.2471

## Best LR rows by seed

- seed 123 `AdamW`: acc=0.4272, loss=1.8268, lr=0.003, effective_lr=0.003
- seed 123 `Lion`: acc=0.3003, loss=2.2374, lr=0.001, effective_lr=0.001
- seed 123 `Muon`: acc=0.8115, loss=0.9895, lr=0.003, effective_lr=0.03
- seed 124 `AdamW`: acc=0.5239, loss=1.9741, lr=0.003, effective_lr=0.003
- seed 124 `Lion`: acc=0.3481, loss=2.2343, lr=0.001, effective_lr=0.001
- seed 124 `Muon`: acc=0.7500, loss=1.1619, lr=0.003, effective_lr=0.03
- seed 125 `AdamW`: acc=0.4961, loss=1.8517, lr=0.003, effective_lr=0.003
- seed 125 `Lion`: acc=0.1997, loss=2.2333, lr=0.001, effective_lr=0.001
- seed 125 `Muon`: acc=0.9907, loss=0.9545, lr=0.003, effective_lr=0.03
- seed 126 `AdamW`: acc=0.2002, loss=2.2316, lr=0.003, effective_lr=0.003
- seed 126 `Lion`: acc=0.3003, loss=2.2807, lr=0.001, effective_lr=0.001
- seed 126 `Muon`: acc=0.7480, loss=1.4821, lr=0.003, effective_lr=0.03
- seed 127 `AdamW`: acc=0.4751, loss=1.9389, lr=0.003, effective_lr=0.003
- seed 127 `Lion`: acc=0.1997, loss=2.2499, lr=0.001, effective_lr=0.001
- seed 127 `Muon`: acc=0.9492, loss=1.0637, lr=0.003, effective_lr=0.03
