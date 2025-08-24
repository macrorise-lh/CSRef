# coding=utf-8
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
"""
Data Loader Module for SREC Datasets in CSRef Project

This module provides functionality to create data loaders for SREC datasets in the CSRef project.
It includes a custom collate function for handling speech and image data, and functions for building
training and testing data loaders with support for distributed training across multiple GPUs.

The module is designed to work with SREC datasets that contain audio data, corresponding images, 
ground truth boxes, and additional metadata. It supports both distributed and
sequential sampling strategies for training and evaluation.

Key Components:
- Custom collate function for handling variable-length audio data with fixed-size tensors
- Distributed training support with proper batch size scaling
- Configurable evaluation modes (distributed vs sequential)
- Memory-efficient data loading with pin_memory support
"""

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, SequentialSampler
from torch.utils.data import DataLoader


def my_collate_fn(batch):
    """
    Custom collate function for processing batches of speech dataset items.
    
    This function handles the batching of data items that contain variable-length audio data
    along with fixed-size tensors like images and bounding boxes. It processes each component
    of the batch items appropriately, stacking tensors where possible and keeping lists for
    variable-length data.
    
    Args:
        batch (list): A list of tuples where each tuple contains:
            - audio_data: Variable-length audio tensor or array
            - image_data: Fixed-size image tensor
            - box_data: Bounding box tensor
            - gt_box_data: Ground truth bounding box tensor
            - info_data: Additional metadata or information
    
    Returns:
        list: A list containing batched data components:
            - audios (list): List of audio data (kept as list due to variable lengths)
            - images (torch.Tensor): Stacked image tensors
            - boxs (torch.Tensor): Stacked bounding box tensors
            - gt_boxs (torch.Tensor): Stacked ground truth bounding box tensors
            - infos (list): List of metadata information (kept as list)
    """
    # Extract each component from the batch items
    audios = [item[0] for item in batch]  # Audio data (variable length)
    images = [item[1] for item in batch]  # Image data (fixed size)
    boxs = [item[2] for item in batch]    # Bounding box data (YOLO format)
    gt_boxs = [item[3] for item in batch] # Ground truth bounding box data (original)
    infos = [item[4] for item in batch]   # Additional metadata
    
    # Stack fixed-size tensors and return variable-length data as lists
    return [audios,
            torch.stack(images),      # Stack images into a single tensor
            torch.stack(boxs),        # Stack bounding boxes into a single tensor
            torch.stack(gt_boxs),     # Stack ground truth boxes into a single tensor
            infos,                    # Keep infos as a list
            ]


def build_train_speech_loader(cfg, dataset: torch.utils.data.Dataset, shuffle=True, drop_last=False):
    """
    Build a distributed data loader for training speech datasets.
    
    This function creates a PyTorch DataLoader configured for distributed training across
    multiple GPUs. It handles the proper distribution of data among processes and ensures
    consistent batch sizes across all distributed processes.
    
    Args:
        cfg: Configuration object containing training parameters:
            - cfg.train.batch_size: Total batch size across all processes
            - cfg.train.data.num_workers: Number of worker processes for data loading
            - cfg.train.data.pin_memory: Whether to use pinned memory for faster GPU transfers
        dataset (torch.utils.data.Dataset): The dataset to load data from
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
    
    Returns:
        torch.utils.data.DataLoader: Configured data loader for distributed training
    """
    # Get distributed training information
    num_tasks = dist.get_world_size()  # Total number of processes in distributed training
    global_rank = dist.get_rank()     # Rank of the current process

    # Validate distributed training setup
    assert cfg.train.batch_size % num_tasks == 0, "Batch size must be divisible by number of processes"
    assert dist.is_initialized(), "Distributed training must be initialized"

    # Calculate per-process batch size
    train_micro_batch_size = cfg.train.batch_size // num_tasks

    # Create distributed sampler for training data
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=num_tasks,      # Total number of processes
        shuffle=shuffle,             # Shuffle data for training
        rank=global_rank,            # Rank of current process
    )
    
    # Create and configure the data loader
    data_loader = DataLoader(
        dataset,
        batch_size=train_micro_batch_size,  # Batch size per process
        sampler=train_sampler,              # Distributed sampler
        num_workers=cfg.train.data.num_workers,  # Number of data loading workers
        pin_memory=cfg.train.data.pin_memory,    # Use pinned memory for faster GPU transfers
        drop_last=drop_last,                   # Drop last incomplete batch if specified
        collate_fn=my_collate_fn              # Custom collate function for speech data
    )
    return data_loader


def build_test_speech_loader(cfg, dataset: torch.utils.data.Dataset, shuffle=False, drop_last=False):
    """
    Build a data loader for testing/evaluating speech datasets.
    
    This function creates a PyTorch DataLoader for evaluation, supporting both distributed
    and sequential sampling strategies. It allows for consistent evaluation across multiple
    processes or sequential evaluation on a single process.
    
    Args:
        cfg: Configuration object containing evaluation parameters:
            - cfg.train.evaluation.eval_batch_size: Total batch size for evaluation
            - cfg.train.evaluation.sequential: Whether to use sequential sampling
            - cfg.train.data.num_workers: Number of worker processes for data loading
            - cfg.train.data.pin_memory: Whether to use pinned memory for faster GPU transfers
        dataset (torch.utils.data.Dataset): The dataset to load data from
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False for evaluation.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
    
    Returns:
        torch.utils.data.DataLoader: Configured data loader for evaluation
    
    Raises:
        AssertionError: If distributed training is not initialized or if eval_batch_size
                        is not evenly divisible by the number of processes (in distributed mode)
    """
    # Get distributed training information
    num_tasks = dist.get_world_size()  # Total number of processes in distributed training
    global_rank = dist.get_rank()     # Rank of the current process

    # Validate distributed training setup and batch size
    assert cfg.train.evaluation.eval_batch_size % num_tasks == 0, "Eval batch size must be divisible by number of processes"
    assert dist.is_initialized(), "Distributed training must be initialized"

    # Calculate per-process batch size for evaluation
    eval_micro_batch_size = cfg.train.evaluation.eval_batch_size // num_tasks

    # Choose sampling strategy based on configuration
    if cfg.train.evaluation.sequential:
        # Sequential evaluation mode - use full batch size on each process
        eval_micro_batch_size = cfg.train.evaluation.eval_batch_size
        eval_sampler = SequentialSampler(dataset)  # Process data sequentially
    else:
        # Distributed evaluation mode - distribute data across processes
        eval_sampler = DistributedSampler(
            dataset,
            num_replicas=num_tasks,      # Total number of processes
            shuffle=shuffle,             # Typically False for evaluation
            rank=global_rank,            # Rank of current process
        )

    # Create and configure the evaluation data loader
    data_loader = DataLoader(
        dataset,
        batch_size=eval_micro_batch_size,  # Batch size per process
        sampler=eval_sampler,              # Chosen sampler (sequential or distributed)
        num_workers=cfg.train.data.num_workers,  # Number of data loading workers
        pin_memory=cfg.train.data.pin_memory,    # Use pinned memory for faster GPU transfers
        drop_last=drop_last,                   # Drop last incomplete batch if specified
        collate_fn=my_collate_fn              # Custom collate function for speech data
    )
    return data_loader
