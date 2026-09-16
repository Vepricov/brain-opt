#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
import os
import socket
import torch
from mpi4py import MPI

comm = MPI.COMM_WORLD
info = {
    "mpi_rank": comm.Get_rank(),
    "mpi_size": comm.Get_size(),
    "host": socket.gethostname(),
    "cuda_devices": torch.cuda.device_count(),
    "ompi_rank": os.environ.get("OMPI_COMM_WORLD_RANK"),
    "ompi_size": os.environ.get("OMPI_COMM_WORLD_SIZE"),
}
print("NODE_TOPOLOGY", json.dumps(info, sort_keys=True), flush=True)
assert comm.Get_size() == 2, info
assert torch.cuda.device_count() == 8, info
PY

python multinode_16gpu_collective.py
