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
Evaluation script for CSA (Contrastive Speech Alignment) model.

This module provides functionality to validate a trained CSA model by evaluating its performance
on a validation dataset. It processes audio and text inputs through the model, computes loss,
and logs evaluation metrics. 

Key Features:
- Processes audio and text data in batches
- Computes and tracks evaluation metrics including loss
- Supports distributed training environments
- Integrates with TensorBoard for logging
- Handles EMA model weights for evaluation
"""

import time
import numpy as np

import torch
from csref.utils.metric import AverageMeter
from csref.utils.distributed import is_main_process, reduce_meters


def validate(cfg, model, data_loader, writer, epoch, logger, rank, audio_preprocessor, text_tokenizer, save_ids=None,
             prefix='Val', ema=None):
    """
    Validate the CSA model on the given dataset.
    
    This function evaluates the model's performance by processing batches of audio-text pairs
    through the model and computing the loss. It handles data preprocessing, model inference,
    metric tracking, and logging of evaluation results.
    
    Args:
        cfg: Configuration object containing model and dataset settings
        model: The CSA model to be evaluated
        data_loader: DataLoader providing batches of audio-text pairs
        writer: TensorBoard writer for logging metrics (can be None)
        epoch: Current epoch number (used for logging)
        logger: Logger object for outputting evaluation information
        rank: Process rank in distributed training setup
        audio_preprocessor: Processor for audio inputs (handles feature extraction)
        text_tokenizer: Tokenizer for text inputs (handles tokenization)
        save_ids: Optional parameter for saving specific sample IDs (unused in current implementation)
        prefix: String prefix for logging (default 'Val' for validation)
        ema: Optional EMA (Exponential Moving Average) model wrapper
        
    Returns:
        float: The average loss value across all validation batches
    """
    # Apply EMA shadow weights if EMA model is provided
    if ema is not None:
        ema.apply_shadow()
    
    # Set model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()

    # Initialize metric trackers for evaluation
    batch_time = AverageMeter('Time', ':6.5f')  # Tracks batch processing time
    data_time = AverageMeter('Data', ':6.5f')  # Tracks data loading time
    losses = AverageMeter('Loss', ':.4f')      # Tracks loss values
    
    # Create list of all meters and dictionary for easy access
    meters = [batch_time, data_time, losses]
    meters_dict = {meter.name: meter for meter in meters}

    # Disable gradient calculations for evaluation to save memory and computation
    with torch.no_grad():
        end = time.time()  # Start timer for batch processing
        
        # Iterate over batches of audio-text pairs from the data loader
        for idx, (audio_iter, text_iter) in enumerate(data_loader):
            # Preprocess audio inputs: convert raw audio to features with attention masks
            batch_audio = audio_preprocessor(raw_speech=audio_iter, padding=True, max_length=None, truncation=False,
                                             pad_to_multiple_of=None, return_attention_mask=True, return_tensors="pt",
                                             sampling_rate=cfg.dataset.target_sample_rate)
            
            # Tokenize text inputs: convert text to token IDs with attention masks
            batch_text = text_tokenizer.batch_encode_plus(
                text_iter,
                padding=True,          # Pad sequences to same length
                truncation=True,       # Truncate sequences if too long
                max_length=None,       # Use model's max length
                return_tensors='pt',   # Return PyTorch tensors
                return_attention_mask=True  # Generate attention masks
            )
            
            # Extract processed audio features and attention mask
            audio_iter = batch_audio.input_values
            audio_mask_iter = batch_audio.attention_mask

            # Extract tokenized text IDs and attention mask
            text_iter = batch_text.input_ids
            text_mask_iter = batch_text.attention_mask

            # Forward pass through the model to compute loss
            loss = model(audio_iter, audio_mask_iter, text_iter, text_mask_iter)

            # Update loss meter with current batch loss value
            losses.update(loss.item(), audio_iter.size(0))

            # Reduce metrics across all processes in distributed training
            reduce_meters(meters_dict, rank, cfg)

            # Log evaluation progress at specified intervals
            if (idx % cfg.train.log_period == 0 or idx == (len(data_loader) - 1)):
                # Calculate GPU memory usage in MB
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Evaluation on {prefix}: [{idx}/{len(data_loader)}]  '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                    f'Mem {memory_used:.0f}MB')
            
            # Update batch processing time and reset timer
            batch_time.update(time.time() - end)
            end = time.time()

        # Log final metrics to TensorBoard if this is the main process
        if is_main_process() and writer is not None:
            writer.add_scalar("Loss", losses.avg_reduce, global_step=epoch)
        
        # Log final average loss
        logger.info(f' * Loss {losses.avg_reduce:.4f}')

    # Restore original model weights if EMA was used
    if ema is not None:
        ema.restore()
    
    # Return the average loss across all batches
    return losses.avg_reduce
