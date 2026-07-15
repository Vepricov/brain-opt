# Базовый образ платформы: PlatformAPI + brain_opt уже внутри.
ARG BASE_TAG=jaguar-a2pro-optlibs
FROM plibs:${BASE_TAG}

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY src ./src
RUN python -c "import brain_opt, transformers; from brain_opt import get_optimizer, run_fedavg, run_async_sgd, run_async_local_sgd; print('brain_opt:', brain_opt.__version__, get_optimizer.__name__, run_fedavg.__name__, run_async_sgd.__name__, run_async_local_sgd.__name__, 'transformers:', transformers.__version__)"

COPY main.json ./main.json
COPY stages ./stages

CMD ["python", "src/federated_train_main.py"]
