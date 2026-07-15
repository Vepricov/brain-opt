# A2.Pro Optimization Storage Demo

Format 2 решение для демонстрации ОПТ-1/2 и ОПТ-3 через библиотеки базового
образа. Решение не копирует реализации оптимизаторов внутрь stages: методы
вызываются из `brain_opt`, который должен быть установлен в БО.

Текущая сборка рассчитана на базовый образ `plibs:jaguar-2.6.7-a2`. В нем
используется обновленный PlatformAPI с потоковым чтением и записью файлов.

## Stages

### `lm_finetune`

Сценарий для ОПТ-1/2.

1. Получает `in_model` как HF-compatible causal LM checkpoint из A2.Pro.
2. Скачивает checkpoint локально через PlatformAPI.
3. Загружает модель через `AutoModelForCausalLM.from_pretrained(local_path)`.
4. Запускает short fine-tuning для `AdamW`, `Lion`, `Muon` через
   `brain_opt.get_optimizer`.
5. Выбирает лучший запуск по validation loss.
6. Пишет `out_model` как checkpoint collection.
7. Пишет `out_metrics` как artifact: `summary.json`, `summary.md`,
   `metrics.csv`, `val_loss_by_method.png`.

### `federated_lm_finetune`

Сценарий для ОПТ-3.

1. Получает тот же тип `in_model` из A2.Pro.
2. Делит synthetic token-level задачу на клиентов.
3. Запускает `FedAvg`, `AsyncSGD`, `Async-LocalSGD` из `brain_opt.federated`.
4. Логирует loss, validation loss и staleness.
5. Пишет лучшую server model в `out_model`.
6. Пишет `out_metrics` с таблицами и графиком.

Важно: это по-прежнему single-process simulator, а не multi-device runtime. Но
он демонстрирует чтение модели из A2.Pro storage, вызов методов ОПТ-3 из
базового образа и запись нового checkpoint обратно в A2.Pro.

### `federated_train`

Исходный сценарий ОПТ-3 на synthetic CIFAR-shaped данных. Он оставлен как
быстрая демонстрация FedAvg/AsyncSGD/Async-LocalSGD без входного checkpoint.

## Offline smoke

`lm_finetune`:

```bash
LM_FINETUNE_OFFLINE=1 \
PYTHONPATH=src:/path/to/brain-opt \
python src/lm_finetune_main.py
```

`federated_lm_finetune`:

```bash
FEDERATED_LM_OFFLINE=1 \
PYTHONPATH=src:/path/to/brain-opt \
python src/federated_lm_finetune_main.py
```

Offline mode создает локальный tiny HF-compatible causal LM checkpoint и гонит
тот же код загрузки/дообучения/сохранения, но без PlatformAPI.

Ожидаемые локальные файлы:

```text
/tmp/a2pro_lm_finetune/out_model/config.json
/tmp/a2pro_lm_finetune/out_model/pytorch_model.bin
/tmp/a2pro_lm_finetune/out_metrics/summary.json
/tmp/a2pro_lm_finetune/out_metrics/summary.md
/tmp/a2pro_lm_finetune/out_metrics/metrics.csv

/tmp/a2pro_federated_lm/out_model/config.json
/tmp/a2pro_federated_lm/out_model/pytorch_model.bin
/tmp/a2pro_federated_lm/out_metrics/summary.json
/tmp/a2pro_federated_lm/out_metrics/summary.md
/tmp/a2pro_federated_lm/out_metrics/metrics.csv
```

## A2.Pro inputs and outputs

Для `lm_finetune` и `federated_lm_finetune`:

- input `in_model`: HF-compatible checkpoint causal LM из хранилища A2.Pro.
- output `out_model`: новый checkpoint collection с дообученной моделью.
- output `out_metrics`: artifact с метриками и графиком.

Для демонстрации достаточно загрузить маленькую модель, например
`sshleifer/tiny-gpt2` или другой HF-compatible checkpoint. Для `distilgpt2`
централизованный stage подходит, но федеративный stage будет заметно тяжелее,
поскольку симулятор делает локальные копии модели.

## Build

```bash
docker build -t a2pro-opt3-federated:v1.2.0-plibs-2.6.7-a2 .
```

Build падает, если в базовом образе отсутствуют `brain_opt` или `transformers`.
