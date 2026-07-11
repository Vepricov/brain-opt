# OPT-3 Federated Solution for A2.Pro

Минимальное Format 2 решение для приемки OPT-3. Стадия не содержит реализацию
алгоритмов внутри себя: она импортирует `run_fedavg`, `run_async_sgd` и
`run_async_local_sgd` из `brain_opt`. Это проверяет схему “метод завернут в
Python-библиотеку, библиотека входит в базовый образ, решение вызывает метод из
БО”.

## Что делает стадия

1. Генерирует детерминированный CIFAR-shaped датасет без сетевых загрузок.
2. Разбивает данные на клиентов (`iid` или `label-skew`).
3. Запускает `FedAvg`, `AsyncSGD`, `Async-LocalSGD` из `brain_opt`.
4. Выбирает лучшую модель по test accuracy.
5. Пишет `out_model` как `checkpoint-collection`: `model.pt`, `model_config.json`.
6. Пишет `out_metrics` как artifact: `summary.json`, `summary.md`, `metrics.csv`,
   и PNG-графики, если в образе есть `matplotlib`.

## Локальный smoke-run

```bash
FEDERATED_OFFLINE=1 \
PYTHONPATH=src:/path/to/brain-opt \
python src/federated_train_main.py
```

Ожидаемые локальные файлы:

```text
/tmp/a2pro_federated/out_model/model.pt
/tmp/a2pro_federated/out_model/model_config.json
/tmp/a2pro_federated/out_metrics/summary.json
/tmp/a2pro_federated/out_metrics/summary.md
/tmp/a2pro_federated/out_metrics/metrics.csv
```

Такие же примерные outputs лежат в `sample_outputs/` и закоммичены без бинарного
`model.pt`.

Проверенный smoke-run на дефолтных параметрах:

```text
FedAvg final_loss=0.8816 final_accuracy=0.9609
AsyncSGD final_loss=2.2907 final_accuracy=0.2188 mean_staleness=4.62
Async-LocalSGD final_loss=2.0089 final_accuracy=0.2031 mean_staleness=4.62
```

## Сборка

```bash
docker build -t a2pro-opt3-federated:v1.0.0-base-optlibs .
```

Если `brain_opt` не установлен в базовом образе, build упадет на smoke-check
импортов.
