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
Training Engine for Speech Referring Expression Comprehension (SREC)

This module implements the training pipeline for SREC models.
It handles the complete training workflow including data loading, model initialization,
distributed training setup, checkpoint management, and evaluation.

The training engine supports:
- Multi-scale training for improved robustness
- Automatic Mixed Precision (AMP) for faster training
- Exponential Moving Average (EMA) for model stabilization
- Distributed training across multiple GPUs
- Automatic checkpoint resumption
- TensorBoard logging for monitoring training progress
- Audio encoder pre-training weight loading

Key Components:
- train_one_epoch: Executes a single training epoch with forward/backward passes
- main: Orchestrates the entire training process including setup and evaluation
"""

import os
import time
import datetime
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from csref.config import LazyConfig, instantiate

from csref.datasets.dataloader_speech import build_train_speech_loader, build_test_speech_loader

from csref.scheduler.build import build_lr_scheduler
from csref.utils.model_ema import EMA
from csref.utils.logger import create_logger
from csref.utils.env import seed_everything
from csref.utils.metric import AverageMeter
from csref.utils.distributed import reduce_meters, is_main_process, cleanup_distributed
from csref.utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume_helper, \
    load_checkpoint_for_audio_encoder

from tools.eval_engine_speech import validate


def train_one_epoch(cfg, model, optimizer, scheduler, data_loader, scalar, writer, epoch, rank, audio_preprocessor, ema=None):
    """
    Train the model for one epoch.
    
    This function executes a complete training epoch, including data loading, forward pass,
    backward pass, optimization, and logging. It supports both regular training and
    Automatic Mixed Precision (AMP) training for improved performance.
    
    Args:
        cfg: Configuration object containing training parameters
        model: The neural network model to train
        optimizer: The optimizer for updating model parameters
        scheduler: Learning rate scheduler
        data_loader: DataLoader for training data
        scalar: GradScaler for AMP training (None if AMP is disabled)
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        rank: Process rank for distributed training
        audio_preprocessor: Audio preprocessing function
        ema: Exponential Moving Average model (optional)
        
    Returns:
        None
    """
    # Set model to training mode
    model.train()
    # Set epoch for distributed sampler to ensure proper data shuffling
    data_loader.sampler.set_epoch(epoch)

    # Initialize metrics tracking
    num_iters = len(data_loader)
    batch_time = AverageMeter('Time', ':6.5f')  # Time per batch
    data_time = AverageMeter('Data', ':6.5f')   # Data loading time
    losses = AverageMeter('Loss', ':.4f')       # Training loss
    meters = [batch_time, data_time, losses]
    meters_dict = {meter.name: meter for meter in meters}

    # Track time for epoch duration calculation
    start = time.time()
    end = time.time()
    
    # Main training loop
    for idx, (audio_iter, image_iter, box_iter, gt_box_iter, info_iter) in enumerate(data_loader):
        # Update data loading time
        data_time.update(time.time() - end)

        # Preprocess audio data with padding and attention mask
        batch_audio = audio_preprocessor(audio_iter, padding=True, max_length=None, truncation=False,
                                         pad_to_multiple_of=None, return_attention_mask=True, return_tensors="pt",
                                         sampling_rate=cfg.dataset.target_sample_rate)
        audio_iter = batch_audio.input_values
        audio_mask_iter = batch_audio.attention_mask

        # Move image and box data to GPU
        image_iter = image_iter.cuda(non_blocking=True)
        box_iter = box_iter.cuda(non_blocking=True)

        # Apply multi-scale training if enabled
        if cfg.train.multi_scale_training:
            img_scales = cfg.train.multi_scale_training.img_scales
            h, w = img_scales[np.random.randint(0, len(img_scales))]
            image_iter = F.interpolate(image_iter, (h, w))

        # Forward pass with or without AMP
        if scalar is not None:
            # Use Automatic Mixed Precision for faster training
            with torch.cuda.amp.autocast():
                loss = model(image_iter, audio_iter, audio_mask_iter, det_label=box_iter)
        else:
            loss = model(image_iter, audio_iter, audio_mask_iter, det_label=box_iter)

        # Backward pass and optimization
        optimizer.zero_grad()
        if scalar is not None:
            # Scale loss for AMP training
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            # Apply gradient clipping if enabled
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.clip_grad_norm
                )
            scalar.update()
        else:
            # Standard training without AMP
            loss.backward()
            # Apply gradient clipping if enabled
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.clip_grad_norm
                )
            optimizer.step()
        # Update learning rate
        scheduler.step()

        # Update EMA model if enabled
        if ema is not None:
            ema.update_params()

        # Update loss metrics
        losses.update(loss.item(), image_iter.size(0))

        # Reduce metrics across all processes in distributed training
        reduce_meters(meters_dict, rank, cfg)
        if is_main_process():
            # Log to TensorBoard
            global_step = epoch * num_iters + idx
            writer.add_scalar("loss/train", losses.avg_reduce, global_step=global_step)

            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("lr/train", lr, global_step=global_step)

        # Log training progress
        if idx % cfg.train.log_period == 0 or idx == len(data_loader):
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_iters - idx)
            logger.info(
                f'Train: [{epoch}/{cfg.train.epochs}][{idx}/{num_iters}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.7f}  '
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'Det Loss {losses.val:.4f} ({losses.avg:.4f})  '
                f'Mem {memory_used:.0f}MB')

        # Update batch time
        batch_time.update(time.time() - end)
        end = time.time()

    # Log total epoch time
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def main(cfg):
    """
    Main training function that orchestrates the entire training process.
    
    This function sets up the training environment, initializes the model and data loaders,
    handles checkpoint loading/resuming, runs the training loop, and manages model evaluation
    and checkpoint saving.
    
    Args:
        cfg: Configuration object containing all training parameters
        
    Returns:
        None
    """
    # Initialize best detection accuracy for model checkpointing
    global best_det_acc
    best_det_acc = 0.

    # Initialize audio preprocessor for speech data
    audio_preprocessor = instantiate(cfg.preprocessor)

    # Build training dataset and dataloader
    cfg.dataset.split = "train"
    train_set = instantiate(cfg.dataset)
    train_loader = build_train_speech_loader(
        cfg,
        train_set,
        shuffle=True,
        drop_last=True  # Drop last incomplete batch for consistent batch sizes
    )

    # Build validation dataset and dataloader
    cfg.dataset.split = "val"
    val_set = instantiate(cfg.dataset)
    val_loader = build_test_speech_loader(
        cfg,
        val_set,
        shuffle=False,
        drop_last=False,  # Keep all validation data
    )

    # Initialize model from configuration
    model = instantiate(cfg.model)

    # Build optimizer with trainable parameters only
    params = filter(lambda p: p.requires_grad, model.parameters())
    cfg.optim.params = params
    optimizer = instantiate(cfg.optim)

    # Initialize EMA (Exponential Moving Average) model (will be set later if enabled)
    ema = None

    # Setup distributed training
    torch.cuda.set_device(dist.get_rank())
    if cfg.train.sync_bn.enabled:
        # Convert to synchronized batch norm for distributed training
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Converted model to use Synchronized BatchNorm.")
    # Wrap model with DistributedDataParallel for multi-GPU training
    model = DistributedDataParallel(model.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True)
    model_without_ddp = model.module  # Keep reference to model without DDP wrapper

    # Log model information on main process
    if is_main_process():
        total_params = sum([param.nelement() for param in model.parameters()])
        trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logger.info(str(model))
        logger.info("Number of all params: %.2fM" % (total_params / 1e6))
        logger.info("Number of trainable params: %.2fM" % (trainable_params / 1e6))

    # Build learning rate scheduler
    scheduler = build_lr_scheduler(cfg, optimizer, len(train_loader))

    # Initialize starting epoch
    start_epoch = 0

    # Auto-resume training from latest checkpoint if enabled
    if cfg.train.auto_resume.enabled:
        resume_file = auto_resume_helper(cfg.train.output_dir)
        if resume_file:
            if cfg.train.resume_path:
                logger.warning(f"auto-resume changing resume file from {cfg.train.resume_path} to {resume_file}")
            cfg.train.resume_path = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {cfg.train.output_dir}, ignoring auto resume')

    # Load pre-trained audio encoder weights if specified
    if cfg.train.audio_encoder_ckpt_path:
        load_checkpoint_for_audio_encoder(cfg, model_without_ddp, logger)

    # Load checkpoint if resume path is specified
    if cfg.train.resume_path:
        start_epoch = load_checkpoint(cfg, model_without_ddp, optimizer, scheduler, logger)

    # Load vision-language pre-trained weights if specified
    if os.path.isfile(cfg.train.vl_pretrain_weight):
        checkpoint = torch.load(cfg.train.vl_pretrain_weight, map_location=lambda storage, loc: storage.cuda())
        logger.warning("loading pretrained weight for finetuning, ignoring resume training, reset start epoch to 0")
        msg = model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(msg)
        start_epoch = 0
        logger.info("==> loaded checkpoint from {}\n".format(cfg.train.vl_pretrain_weight) +
                    "==> epoch: {} lr: {} ".format(checkpoint['epoch'], checkpoint['lr']))

    # Setup Automatic Mixed Precision (AMP) if enabled
    if cfg.train.amp.enabled:
        assert torch.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    # Setup TensorBoard writer on main process
    if is_main_process():
        writer = SummaryWriter(log_dir=cfg.train.output_dir)
    else:
        writer = None

    # Generate random sample IDs for visualization if image logging is enabled
    save_ids = np.random.randint(1, len(val_loader) * cfg.train.batch_size, 100) if cfg.train.log_image else None

    # Main training loop
    for epoch in range(start_epoch, cfg.train.epochs):
        # Initialize EMA model if enabled and not already created
        if cfg.train.ema.enabled and ema is None:
            ema = EMA(model, cfg.train.ema.alpha, cfg.train.ema.buffer_ema)
        
        # Train for one epoch
        train_one_epoch(cfg, model, optimizer, scheduler, train_loader, scalar, writer, epoch, dist.get_rank(), audio_preprocessor, ema)
        
        # Validate model performance
        box_ap = validate(cfg, model, val_loader, writer, epoch, logger, dist.get_rank(),
                          save_ids=save_ids, audio_preprocessor=audio_preprocessor, ema=ema)

        # Save periodic checkpoints
        if epoch % cfg.train.save_period == 0 or epoch == (cfg.train.epochs - 1):
            logger.info(f"saving checkpoints......")
            if is_main_process():
                if ema is not None:
                    ema.apply_shadow()  # Apply EMA weights for checkpoint
                # Save checkpoint with model weights, optimizer state, and scheduler
                save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger)
                if ema is not None:
                    ema.restore()  # Restore original model weights
            logger.info(f"checkpoints saved !!!")

        # Save best checkpoint based on detection accuracy
        if is_main_process():
            if ema is not None:
                ema.apply_shadow()  # Apply EMA weights for evaluation
            if box_ap > best_det_acc:
                # Save checkpoint with best detection accuracy
                save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, det_best=True)
                best_det_acc = box_ap
                logger.info(f"best_det_checkpoints saved !!!")
            if ema is not None:
                ema.restore()  # Restore original model weights

    # Clean up distributed training
    cleanup_distributed()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="csref_SREC")
    parser.add_argument('--config', type=str, required=True, default='./config/simrec_refcoco_scratch.py')
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
    
    # Load configuration from file and apply command line overrides
    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    # Set random seed for reproducibility
    seed_everything(cfg.train.seed)

    # Setup distributed training environment
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
    # Synchronize all processes
    torch.distributed.barrier()

    # Setup output directory and logger
    output_dir = cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=cfg.train.output_dir, dist_rank=dist.get_rank())

    # Save configuration file on main process
    if is_main_process():
        path = os.path.join(cfg.train.output_dir, "config.yaml")
        LazyConfig.save(cfg, path)

    # Start main training process
    main(cfg)
