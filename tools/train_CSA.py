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
Training script for CSA (Contrastive Speech Alignment) models.

This script implements the complete training pipeline for CSA models, which learn to align
speech and text representations through contrastive learning. The script includes:

- Distributed training setup with data parallelism
- Model initialization and checkpoint management
- Training loop with mixed precision support
- Learning rate scheduling and optimization
- Model EMA (Exponential Moving Average) for improved generalization
- Validation and checkpoint saving
- Tensorboard logging for monitoring training progress

The script supports both single-GPU and multi-GPU distributed training environments,
with automatic resumption from checkpoints and comprehensive logging for experiment tracking.
"""

import os
import sys
import time
import datetime
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from csref.config import LazyConfig, instantiate

from csref.datasets.dataloader_CSA import build_train_csa_loader, build_test_csa_loader

from csref.scheduler.build import build_lr_scheduler
from csref.utils.model_ema import EMA
from csref.utils.logger import create_logger
from csref.utils.env import seed_everything
from csref.utils.metric import AverageMeter
from csref.utils.distributed import reduce_meters, is_main_process, cleanup_distributed
from csref.utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume_helper

from tools.eval_CSA import validate

from transformers import Wav2Vec2FeatureExtractor, AutoTokenizer


def train_one_epoch(cfg, model, optimizer, scheduler, data_loader, scalar, writer, epoch, rank, audio_preprocessor,
                    text_tokenizer, ema=None):
    """
    Train the CSA model for one epoch.
    
    This function implements a single training epoch with support for mixed precision training,
    distributed data parallelism, and model EMA. It processes batches of audio-text pairs,
    computes contrastive loss, performs backpropagation, and updates model parameters.
    
    Args:
        cfg: Configuration object containing training parameters
        model: The CSA model to train
        optimizer: PyTorch optimizer for parameter updates
        scheduler: Learning rate scheduler
        data_loader: DataLoader providing training batches
        scalar: GradScaler for mixed precision training (None if disabled)
        writer: TensorBoard writer for logging metrics
        epoch: Current epoch number
        rank: Process rank in distributed training
        audio_preprocessor: Wav2Vec2FeatureExtractor for audio preprocessing
        text_tokenizer: Tokenizer for text preprocessing
        ema: Optional EMA (Exponential Moving Average) model
        
    Returns:
        None
    """
    # Set model to training mode
    model.train()
    # Set epoch for distributed sampler to ensure proper shuffling across epochs
    data_loader.sampler.set_epoch(epoch)

    # Calculate total number of iterations in this epoch
    num_iters = len(data_loader)
    # Initialize metrics tracking for training performance
    batch_time = AverageMeter('Time', ':6.5f')  # Time per batch
    data_time = AverageMeter('Data', ':6.5f')   # Data loading time
    losses = AverageMeter('Loss', ':.4f')       # Training loss
    meters = [batch_time, data_time, losses]
    # Create dictionary for easy access to meters during distributed training
    meters_dict = {meter.name: meter for meter in meters}

    # Track epoch start time for duration calculation
    start = time.time()
    end = time.time()
    # Main training loop - iterate over batches of audio-text pairs
    for idx, (audio_iter, text_iter) in enumerate(data_loader):
        # Update data loading time metric
        data_time.update(time.time() - end)

        # Preprocess audio data using Wav2Vec2 feature extractor
        batch_audio = audio_preprocessor(raw_speech=audio_iter, padding=True, max_length=None, truncation=False,
                                         pad_to_multiple_of=None, return_attention_mask=True, return_tensors="pt",
                                         sampling_rate=cfg.dataset.target_sample_rate)
        # Tokenize text data using BERT tokenizer
        batch_text = text_tokenizer.batch_encode_plus(
            text_iter,
            padding=True,
            truncation=True,
            max_length=None,
            return_tensors='pt',
            return_attention_mask=True
        )
        # Extract processed audio features and attention mask
        audio_iter = batch_audio.input_values
        audio_mask_iter = batch_audio.attention_mask

        # Extract tokenized text IDs and attention mask
        text_iter = batch_text.input_ids
        text_mask_iter = batch_text.attention_mask

        # Forward pass with mixed precision if enabled
        if scalar is not None:
            with torch.cuda.amp.autocast():
                # Compute contrastive loss between audio and text representations
                loss = model(audio_iter, audio_mask_iter, text_iter, text_mask_iter)
        else:
            # Standard forward pass without mixed precision
            loss = model(audio_iter, audio_mask_iter, text_iter, text_mask_iter)

        # Clear gradients before backward pass
        optimizer.zero_grad()
        
        # Backward pass and optimization with mixed precision support
        if scalar is not None:
            # Scale loss for mixed precision training
            scalar.scale(loss).backward()
            # Update optimizer with scaled gradients
            scalar.step(optimizer)
            # Apply gradient clipping to prevent exploding gradients
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.clip_grad_norm
                )
            # Update gradient scaler for next iteration
            scalar.update()
        else:
            # Standard backward pass without mixed precision
            loss.backward()
            # Apply gradient clipping
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.clip_grad_norm
                )
            # Update model parameters
            optimizer.step()
        
        # Update learning rate according to scheduler
        scheduler.step()

        # Update EMA model if enabled
        if ema is not None:
            ema.update_params()

        # Update loss metric with current batch loss
        losses.update(loss.item(), audio_iter.size(0))

        # Synchronize metrics across all processes in distributed training
        reduce_meters(meters_dict, rank, cfg)
        # Log metrics on main process
        if is_main_process():
            # Calculate global step for TensorBoard logging
            global_step = epoch * num_iters + idx
            # Log training loss
            writer.add_scalar("loss/train", losses.avg_reduce, global_step=global_step)

            # Log current learning rate
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("lr/train", lr, global_step=global_step)

        # Log training progress at specified intervals
        if idx % cfg.train.log_period == 0 or idx == len(data_loader):
            lr = optimizer.param_groups[0]['lr']
            # Calculate GPU memory usage in MB
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            # Estimate remaining time for current epoch
            etas = batch_time.avg * (num_iters - idx)
            logger.info(
                f'Train: [{epoch}/{cfg.train.epochs}][{idx}/{num_iters}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.7f}  '
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                f'Mem {memory_used:.0f}MB')

        # Update batch processing time
        batch_time.update(time.time() - end)
        end = time.time()

    # Calculate and log total epoch training time
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def main(cfg):
    """
    Main training function for the CSA model.
    
    This function orchestrates the entire training process including:
    - Model and data initialization
    - Distributed training setup
    - Optimizer and scheduler configuration
    - Checkpoint management and resumption
    - Training loop execution
    - Model validation and checkpoint saving
    
    Args:
        cfg: Configuration object containing all training parameters
        
    Returns:
        None
    """
    # Initialize global variable to track the lowest validation loss
    global lowest_loss
    lowest_loss = sys.float_info.max

    # Initialize audio and text preprocessors from pretrained models
    audio_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.model.speech_encoder.pretrained_path)
    text_tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder.pretrained_path)

    # Build training dataset and dataloader
    cfg.dataset.train_split = "train"
    train_set = instantiate(cfg.dataset)
    train_loader = build_train_csa_loader(
        cfg,
        train_set,
        shuffle=True,      # Shuffle training data for better generalization
        drop_last=True     # Drop last incomplete batch for consistent batch sizes
    )

    # Build validation dataset and dataloader
    cfg.dataset.train_split = "val"
    val_set = instantiate(cfg.dataset)
    val_loader = build_test_csa_loader(
        cfg,
        val_set,
        shuffle=False,     # No need to shuffle validation data
        drop_last=False,   # Keep all validation samples
    )

    # Initialize CSA model from configuration
    model = instantiate(cfg.model)

    # Build optimizer with trainable parameters only
    params = filter(lambda p: p.requires_grad, model.parameters())
    cfg.optim.params = params
    optimizer = instantiate(cfg.optim)

    # Initialize EMA (Exponential Moving Average) model as None (will be set later if enabled)
    ema = None

    # Set up distributed training environment
    torch.cuda.set_device(dist.get_rank())
    # Convert BatchNorm layers to synchronized version for multi-GPU training
    if cfg.train.sync_bn.enabled:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Converted model to use Synchronized BatchNorm.")
    # Wrap model with DistributedDataParallel for multi-GPU training
    model = DistributedDataParallel(model.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True)
    # Keep reference to model without DDP wrapper for checkpoint saving
    model_without_ddp = model.module

    # Log model information on main process
    if is_main_process():
        # Calculate total and trainable parameter counts
        total_params = sum([param.nelement() for param in model.parameters()])
        trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logger.info(str(model))
        logger.info("Number of all params: %.2fM" % (total_params / 1e6))
        logger.info("Number of trainable params: %.2fM" % (trainable_params / 1e6))

    # Build learning rate scheduler
    scheduler = build_lr_scheduler(cfg, optimizer, len(train_loader))

    # Initialize starting epoch (will be updated if resuming from checkpoint)
    start_epoch = 0

    # Auto-resume training from the latest checkpoint if enabled
    if cfg.train.auto_resume.enabled:
        resume_file = auto_resume_helper(cfg.train.output_dir)
        if resume_file:
            # If both auto-resume and manual resume are specified, use auto-resume
            if cfg.train.resume_path:
                logger.warning(f"auto-resume changing resume file from {cfg.train.resume_path} to {resume_file}")
            cfg.train.resume_path = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {cfg.train.output_dir}, ignoring auto resume')

    # Load checkpoint if resume path is specified
    if cfg.train.resume_path:
        start_epoch = load_checkpoint(cfg, model_without_ddp, optimizer, scheduler, logger)

    # Load pretrained weights for fine-tuning if specified
    if os.path.isfile(cfg.train.vl_pretrain_weight):
        checkpoint = torch.load(cfg.train.vl_pretrain_weight, map_location=lambda storage, loc: storage.cuda())
        logger.warning("loading pretrained weight for finetuning, ignoring resume training, reset start epoch to 0")
        # Load state dict with strict=False to allow partial parameter loading
        msg = model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(msg)
        start_epoch = 0
        logger.info("==> loaded checkpoint from {}\n".format(cfg.train.vl_pretrain_weight) +
                    "==> epoch: {} lr: {} ".format(checkpoint['epoch'], checkpoint['lr']))

    # Initialize mixed precision training if enabled
    if cfg.train.amp.enabled:
        assert torch.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    # Initialize TensorBoard writer on main process only
    if is_main_process():
        writer = SummaryWriter(log_dir=cfg.train.output_dir)
    else:
        writer = None


    # Main training loop
    for epoch in range(start_epoch, cfg.train.epochs):
        # Initialize EMA model if enabled and not already created
        if cfg.train.ema.enabled and ema is None:
            ema = EMA(model, cfg.train.ema.alpha, cfg.train.ema.buffer_ema)
        
        # Train for one epoch
        train_one_epoch(cfg, model, optimizer, scheduler, train_loader, scalar, writer, epoch, dist.get_rank(),
                        audio_preprocessor, text_tokenizer, ema)
        
        # Validate model and compute validation loss
        loss = validate(cfg, model, val_loader, writer, epoch, logger, dist.get_rank(),
                        save_ids=None, audio_preprocessor=audio_preprocessor, text_tokenizer=text_tokenizer,
                        ema=ema)

        # Save periodic checkpoints
        if epoch % cfg.train.save_period == 0 or epoch == (cfg.train.epochs - 1):
            logger.info(f"saving checkpoints......")
            if is_main_process():
                # Apply EMA weights before saving if EMA is enabled
                if ema is not None:
                    ema.apply_shadow()
                # Save checkpoint with model weights, optimizer state, and scheduler
                save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger)
                # Restore original model weights after saving
                if ema is not None:
                    ema.restore()
            logger.info(f"checkpoints saved !!!")

        # Save best checkpoint based on validation loss
        if is_main_process():
            # Apply EMA weights before evaluation if EMA is enabled
            if ema is not None:
                ema.apply_shadow()
            # Check if current model has the best validation loss
            if loss < lowest_loss:
                save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, det_best=True)
                lowest_loss = loss
                logger.info(f"best_det_checkpoints saved !!!")
            # Restore original model weights after evaluation
            if ema is not None:
                ema.restore()

    # Clean up distributed training environment
    cleanup_distributed()


# Main execution block
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="csref_CSA")
    parser.add_argument('--config', type=str, required=True, default='./configs/csref_CSA_librispeech.py')
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    
    # Load configuration from file and apply command-line overrides
    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    # Set random seed for reproducibility
    seed_everything(cfg.train.seed)

    # Set up distributed training environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    # Initialize CUDA device and distributed process group
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend=cfg.train.ddp.backend,
        init_method=cfg.train.ddp.init_method,
        world_size=world_size,
        rank=rank
    )
    # Synchronize all processes before starting training
    torch.distributed.barrier()

    # Create output directory for logs and checkpoints
    output_dir = cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # Initialize logger for distributed training
    logger = create_logger(output_dir=cfg.train.output_dir, dist_rank=dist.get_rank())

    # Save configuration file to output directory on main process
    if is_main_process():
        path = os.path.join(cfg.train.output_dir, "config.yaml")
        LazyConfig.save(cfg, path)

    # Start training
    main(cfg)
