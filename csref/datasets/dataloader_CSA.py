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
CSA (Contrastive Semantic Alignment) Data Loader Module

This module provides data loading functionality for the CSA (Contrastive Semantic Alignment)
component of the CSRef project. CSA is a key technique that aligns audio and text modalities
in a shared semantic space, enabling direct semantic extraction from raw speech.

The module contains:
1. Custom collate function for processing CSA dataset batches
2. Training data loader with distributed sampling support for multi-GPU training
3. Evaluation data loader with support for both distributed and sequential evaluation modes

This module is designed to work with the CSRef framework's distributed training infrastructure,
ensuring efficient data loading and proper distribution across multiple processes/GPUs during
both training and evaluation phases.
"""

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, SequentialSampler
from torch.utils.data import DataLoader


def my_collate_fn(batch):
    """
    Custom collate function for CSA (Contrastive Semantic Alignment) dataset.
    
    This function processes a batch of samples from the CSA dataset, which contains
    pairs of audio and transcript data. It separates the audio and transcript data
    into different lists for further processing by the CSA model.
    
    
    Args:
        batch (list): A list of samples, where each sample is a tuple containing
                     (audio, transcript) data. The audio data typically consists
                     of waveform, while transcripts are text.

    Returns:
        list: A list containing two elements:
              - audios (list): List of audio data from each sample. Each element
                contains the audio that will be processed by the
                audio encoder in the CSA model.
              - transcripts (list): List of transcript data from each sample.
                Each element contains the text that will be
                processed by the text encoder in the CSA model.
    """
    # Extract audio data from each sample in the batch
    # Audio data is typically a tensor representing waveform
    audios = [item[0] for item in batch]
    # Extract transcript data from each sample in the batch
    # Transcript data is typically a string
    transcripts = [item[1] for item in batch]
    return [
        audios,
        transcripts,
    ]


def build_train_csa_loader(cfg, dataset: torch.utils.data.Dataset, shuffle=True, drop_last=False):
    """
    Build a distributed data loader for CSA training.
    
    This function creates a PyTorch DataLoader for training the CSA model with
    distributed data parallelism. It handles the distribution of data across
    multiple GPUs/processes and ensures that each process receives a unique
    subset of the data.
    
    The data loader is configured with a distributed sampler that partitions
    the dataset among all available processes, ensuring that each process works
    on a different subset of the data during each epoch. This enables efficient
    parallel training while preventing data overlap between processes.
    
    Args:
        cfg (object): Configuration object containing training parameters. This should
                     include attributes such as:
                     - cfg.train.batch_size: Total batch size across all processes
                     - cfg.train.data.num_workers: Number of worker processes for data loading
                     - cfg.train.data.pin_memory: Whether to pin memory in GPU RAM
        dataset (torch.utils.data.Dataset): The dataset to load data from. This should
                                           be a CSA dataset that returns audio-transcript
                                           pairs when indexed.
        shuffle (bool): Whether to shuffle the data. Defaults to True, which is
                       recommended for training to ensure random ordering of samples
                       in each epoch.
        drop_last (bool): Whether to drop the last incomplete batch. Defaults to False.
                         Setting to True can be useful to ensure consistent batch
                         sizes during training, especially when using batch normalization.
    
    Returns:
        torch.utils.data.DataLoader: The configured data loader for CSA training.
                                    This loader will yield batches of audio-transcript
                                    pairs processed by my_collate_fn.
    
    Example:
        >>> # Assuming cfg and dataset are properly initialized
        >>> train_loader = build_train_csa_loader(cfg, train_dataset, shuffle=True)
        >>> for batch_idx, (audios, transcripts) in enumerate(train_loader):
        ...     # Training loop with audio-transcript pairs
        ...     loss = model(audios, transcripts)
        ...     loss.backward()
        ...     optimizer.step()
    """
    # Get the total number of processes (GPUs) in the distributed setup
    num_tasks = dist.get_world_size()
    # Get the rank of the current process
    global_rank = dist.get_rank()

    # Ensure that the batch size is divisible by the number of processes
    # This is necessary for even distribution of data across processes
    # If not divisible, some processes would get more samples than others
    assert cfg.train.batch_size % num_tasks == 0, \
        f"Batch size {cfg.train.batch_size} must be divisible by num_tasks {num_tasks}"
    # Ensure that distributed training is initialized
    assert dist.is_initialized(), "Distributed training must be initialized"

    # Calculate the micro batch size for each process
    train_micro_batch_size = cfg.train.batch_size // num_tasks

    # Create a distributed sampler that ensures each process gets a unique subset of data
    # The sampler handles the partitioning of the dataset across processes
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=num_tasks,          # Total number of processes
        shuffle=shuffle,                 # Whether to shuffle the data
        rank=global_rank,                # Rank of the current process
    )
    # Create the data loader with the specified parameters
    # The DataLoader orchestrates the actual loading and batching of data
    data_loader = DataLoader(
        dataset,                         # The dataset to load from
        batch_size=train_micro_batch_size,  # Batch size per process
        sampler=train_sampler,           # Distributed sampling strategy
        num_workers=cfg.train.data.num_workers,  # Parallel data loading
        pin_memory=cfg.train.data.pin_memory,    # Optimize GPU memory transfers
        drop_last=drop_last,             # Handle incomplete batches
        collate_fn=my_collate_fn        # Custom batch processing
    )
    return data_loader


def build_test_csa_loader(cfg, dataset: torch.utils.data.Dataset, shuffle=False, drop_last=False):
    """
    Build a data loader for CSA testing/evaluation.
    
    This function creates a PyTorch DataLoader for evaluating the CSA model.
    It supports both distributed and sequential evaluation modes. In distributed
    mode, data is split across multiple processes for parallel evaluation, while
    in sequential mode, each process evaluates the entire dataset independently.
    
    The evaluation mode is controlled by the configuration parameter
    cfg.train.evaluation.sequential. Sequential mode is useful for debugging
    or when consistent results are needed across all processes, while distributed
    mode provides faster evaluation by dividing the workload.
    
    Args:
        cfg (object): Configuration object containing evaluation parameters. This should
                     include attributes such as:
                     - cfg.train.evaluation.eval_batch_size: Total batch size for evaluation
                     - cfg.train.evaluation.sequential: Whether to use sequential evaluation
                     - cfg.train.data.num_workers: Number of worker processes for data loading
                     - cfg.train.data.pin_memory: Whether to pin memory in GPU RAM
        dataset (torch.utils.data.Dataset): The dataset to load data from. This should
                                           be a CSA dataset that returns audio-transcript
                                           pairs when indexed.
        shuffle (bool): Whether to shuffle the data. Defaults to False, which is
                       recommended for evaluation to ensure consistent ordering
                       of samples across multiple evaluation runs.
        drop_last (bool): Whether to drop the last incomplete batch. Defaults to False.
                         For evaluation, it's typically set to False to ensure
                         all samples are processed.
    
    Returns:
        torch.utils.data.DataLoader: The configured data loader for CSA evaluation.
                                    This loader will yield batches of audio-transcript
                                    pairs processed by my_collate_fn.
    
    Example:
        >>> # Assuming cfg and dataset are properly initialized
        >>> test_loader = build_test_csa_loader(cfg, test_dataset, shuffle=False)
        >>> for batch_idx, (audios, transcripts) in enumerate(test_loader):
        ...     # Evaluation loop with audio-transcript pairs
        ...     with torch.no_grad():
        ...         outputs = model(audios, transcripts)
        ...         metrics = compute_metrics(outputs, targets)
    """
    # Get the total number of processes (GPUs) in the distributed setup
    num_tasks = dist.get_world_size()
    # Get the rank of the current process
    global_rank = dist.get_rank()

    # Ensure that the evaluation batch size is divisible by the number of processes
    # This is necessary for even distribution of data across processes in distributed mode
    assert cfg.train.evaluation.eval_batch_size % num_tasks == 0, \
        f"Eval batch size {cfg.train.evaluation.eval_batch_size} must be divisible by num_tasks {num_tasks}"
    # Ensure that distributed training is initialized
    assert dist.is_initialized(), "Distributed training must be initialized"

    # Calculate the micro batch size for each process
    eval_micro_batch_size = cfg.train.evaluation.eval_batch_size // num_tasks

    # Choose between sequential and distributed sampling based on configuration
    if cfg.train.evaluation.sequential:
        # In sequential mode, each process evaluates the entire dataset
        eval_micro_batch_size = cfg.train.evaluation.eval_batch_size
        eval_sampler = SequentialSampler(dataset)  # Process all data in order
    else:
        # In distributed mode, each process evaluates a subset of the dataset
        eval_sampler = DistributedSampler(
            dataset,
            num_replicas=num_tasks,          # Total number of processes
            shuffle=shuffle,                 # Whether to shuffle the data
            rank=global_rank,                # Rank of the current process
        )

    # Create the data loader with the specified parameters
    data_loader = DataLoader(
        dataset,                         # The dataset to load from
        batch_size=eval_micro_batch_size,  # Batch size per process
        sampler=eval_sampler,            # Sampling strategy (sequential or distributed)
        num_workers=cfg.train.data.num_workers,  # Parallel data loading
        pin_memory=cfg.train.data.pin_memory,    # Optimize GPU memory transfers
        drop_last=drop_last,             # Handle incomplete batches
        collate_fn=my_collate_fn        # Custom batch processing
    )
    return data_loader
