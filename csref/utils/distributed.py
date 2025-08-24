# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
# Copyright 2025 The CSRef Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist

from csref.utils.metric import AverageMeter

_LOCAL_PROCESS_GROUP = None

def get_world_size():
    """
    Get the total number of processes in the distributed training setup.
    
    This function checks if distributed training is available and initialized,
    and returns the total number of processes (GPUs) across all machines.
    If distributed training is not available or not initialized, it returns 1,
    indicating single-process (CPU or single GPU) mode.
    
    Returns:
        int: The total number of processes in the distributed setup, or 1 if not distributed
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process in the distributed training setup.
    
    This function checks if distributed training is available and initialized,
    and returns the rank (ID) of the current process. The rank is a unique
    identifier for each process in the distributed setup, ranging from 0 to
    world_size-1. If distributed training is not available or not initialized,
    it returns 0, indicating the single process.
    
    Returns:
        int: The rank of the current process, or 0 if not distributed
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Check if the current process is the main process (rank 0).
    
    In distributed training, the main process (rank 0) is typically responsible
    for tasks like logging, saving checkpoints, and printing progress. This function
    provides a convenient way to check if the current process is the main process.
    
    Returns:
        bool: True if the current process is the main process, False otherwise
    """
    return get_rank() == 0


def get_local_size() -> int:
    """
    Get the number of processes per machine in the distributed training setup.
    
    In multi-machine distributed training, each machine may have multiple processes
    (GPUs). This function returns the number of processes on the local machine.
    If distributed training is not available or not initialized, it returns 1.
    
    Returns:
        int: The number of processes per machine, or 1 if not distributed
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def synchronize():
    """
    Synchronize all processes in the distributed training setup.
    
    This function creates a barrier that blocks until all processes have reached
    this point. It's useful for ensuring that all processes have completed a
    certain operation before proceeding. The function handles different backends
    (NCCL for GPU, Gloo for CPU) appropriately.
    
    If distributed training is not available or not initialized, or if there's
    only one process, this function does nothing.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def cleanup_distributed():
    """
    Clean up the distributed process group.
    
    This function destroys the distributed process group, releasing resources
    and cleaning up after distributed training. It should be called when
    distributed training is complete.
    """
    dist.destroy_process_group()


def reduce_tensor(tensor):
    """
    Reduce a tensor across all processes by averaging.
    
    This function performs an all-reduce operation on the input tensor,
    summing the tensor values across all processes and then dividing by
    the number of processes to get the average. This is useful for
    computing global statistics (e.g., loss, accuracy) during distributed
    training.
    
    Args:
        tensor (torch.Tensor): The tensor to be reduced
        
    Returns:
        torch.Tensor: The reduced tensor with averaged values
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def reduce_meters(meters, rank, cfg):
    """
    Synchronize and reduce AverageMeter objects across all processes.
    
    This function collects AverageMeter objects from all processes, computes
    the global average for each meter, and updates the meters on the main process
    with the global averages. This is useful for aggregating metrics (e.g., loss,
    accuracy) across all processes during distributed training.
    
    Args:
        meters (dict): A dictionary of AverageMeter objects to be synchronized
        rank (int): The rank of the current process
        cfg (object): Configuration object (not used in the function)
        
    Raises:
        TypeError: If a meter is not an AverageMeter object
    """
    assert isinstance(meters, dict), "collect AverageMeters into a dict"
    for name in sorted(meters.keys()):
        meter = meters[name]
        if not isinstance(meter, AverageMeter):
            raise TypeError("meter should be AverageMeter type")
        # Convert the meter's average to a tensor and move it to the specified device
        avg = torch.tensor(meter.avg).unsqueeze(0).to(rank)
        # Create a list to receive the averages from all processes
        avg_reduce = [torch.ones_like(avg) for _ in range(dist.get_world_size())]
        # Gather the averages from all processes
        dist.all_gather(avg_reduce, avg)
        # On the main process, compute the global average and update the meter
        if is_main_process():
            value = torch.mean(torch.cat(avg_reduce)).item()
            meter.update_reduce(value)


def find_free_port():
    """
    Find a free network port for distributed training.
    
    This function creates a socket, binds it to port 0 (which tells the OS
    to find an available port), and returns the port number. This is useful
    for setting up distributed training when the port number needs to be
    determined dynamically.
    
    Note: There is still a small chance that the port could be taken by
    another process between the time it is released and when it is used.
    
    Returns:
        int: A free port number
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
