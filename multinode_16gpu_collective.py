#!/usr/bin/env python3
import json
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mpi4py import MPI


def gpu_worker(local_rank: int, node_rank: int, nodes: int, master_addr: str, master_port: int) -> None:
    per_node = 8
    world_size = nodes * per_node
    global_rank = node_rank * per_node + local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=global_rank,
        world_size=world_size,
    )
    value = torch.tensor(float(global_rank), device=f"cuda:{local_rank}")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)
    expected = world_size * (world_size - 1) / 2
    assert value.item() == expected, (global_rank, value.item(), expected)
    gathered = [None] * world_size if global_rank == 0 else None
    dist.gather_object(
        {"rank": global_rank, "node": node_rank, "local_rank": local_rank,
         "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(local_rank)},
        gathered,
        dst=0,
    )
    if global_rank == 0:
        print("MULTINODE_16GPU_OK", json.dumps({
            "world_size": world_size,
            "nodes": nodes,
            "gpus_per_node": per_node,
            "all_reduce_sum": value.item(),
            "ranks": gathered,
        }, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    comm = MPI.COMM_WORLD
    node_rank = comm.Get_rank()
    nodes = comm.Get_size()
    assert nodes == 2, nodes
    assert torch.cuda.device_count() == 8, torch.cuda.device_count()
    master_addr = socket.gethostbyname(socket.gethostname()) if node_rank == 0 else None
    master_addr = comm.bcast(master_addr, root=0)
    master_port = 29616
    print("LAUNCH_LOCAL_RANKS", node_rank, master_addr, flush=True)
    mp.spawn(gpu_worker, nprocs=8, args=(node_rank, nodes, master_addr, master_port), join=True)


if __name__ == "__main__":
    main()
