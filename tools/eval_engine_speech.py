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
Evaluation Engine for Speech Referring Expression Comprehension (SREC) models.

This module provides functionality for evaluating SREC models.
It handles distributed evaluation across multiple GPUs, data loading, model initialization,
and metric computation for speech-grounded visual tasks. 

Key features:
- Distributed evaluation using PyTorch's DistributedDataParallel
- Support for multiple evaluation datasets (val, testA, testB, test)
- Audio preprocessing pipeline for speech inputs
- Model checkpoint loading and evaluation state management
"""

import os
import time
import argparse
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from csref.config import instantiate, LazyConfig
from csref.datasets.dataloader_speech import build_test_speech_loader
from csref.datasets.utils import yolobox2label
from csref.models.utils import batch_box_iou
from csref.utils.env import seed_everything
from csref.utils.logger import create_logger
from csref.utils.metric import AverageMeter
from csref.utils.distributed import is_main_process, reduce_meters


def validate(cfg, model, data_loader, writer, epoch, logger, rank, audio_preprocessor, save_ids=None, prefix='Val',
             ema=None):
    """
    Validate the model on the given dataset.
    
    This function performs evaluation of the speech-referenced object detection model
    on the provided data loader. It computes various metrics including Box IoU and logs the results. 
    The function handles distributed evaluation and supports Exponential Moving Average (EMA) model weights.
    
    Args:
        cfg (LazyConfig): Configuration object containing model and training parameters
        model (nn.Module): The neural network model to evaluate
        data_loader (DataLoader): DataLoader for the evaluation dataset
        writer (SummaryWriter): TensorBoard writer for logging (can be None)
        epoch (int): Current epoch number for logging purposes
        logger (Logger): Logger instance for outputting evaluation metrics
        rank (int): Process rank for distributed evaluation
        audio_preprocessor: Audio preprocessing pipeline for speech inputs
        save_ids (list, optional): List of sample IDs to save for visualization
        prefix (str, optional): Prefix for logging messages (default: 'Val')
        ema (ModelEma, optional): Exponential Moving Average model wrapper
    
    Returns:
        float: Average Box IoU score
    """
    # Apply EMA weights if available for more stable evaluation
    if ema is not None:
        ema.apply_shadow()
    model.eval()

    # Initialize evaluation metrics
    num_iters = len(data_loader)
    batch_time = AverageMeter('Time', ':6.5f')  # Time per batch
    sample_time = AverageMeter('Sample', ':6.5f')  # Time per sample
    data_time = AverageMeter('Data', ':6.5f')  # Data loading time
    box_ap = AverageMeter('BoxIoU@0.5', ':6.2f')  # Box IoU at 0.5 threshold
    inconsistency_error = AverageMeter('IE', ':6.2f')  # Inconsistency error
    
    # Collect all metrics for distributed synchronization
    meters = [batch_time, data_time, box_ap, inconsistency_error]
    meters_dict = {meter.name: meter for meter in meters}

    # Evaluation loop without gradient computation
    with torch.no_grad():
        end = time.time()
        for idx, (audio_iter, image_iter, box_iter, gt_box_iter, info_iter) in enumerate(data_loader):
            # Preprocess audio inputs with padding and attention masks
            batch_audio = audio_preprocessor(audio_iter, padding=True, max_length=None, truncation=False,
                                             pad_to_multiple_of=None, return_attention_mask=True, return_tensors="pt",
                                             sampling_rate=cfg.dataset.target_sample_rate)
            audio_iter = batch_audio.input_values
            audio_mask_iter = batch_audio.attention_mask

            # Move data to GPU for faster computation
            image_iter = image_iter.cuda(non_blocking=True)
            box_iter = box_iter.cuda(non_blocking=True)
            
            # Forward pass: predict bounding boxes from image and audio
            box = model(image_iter, audio_iter, audio_mask_iter)

            # Convert ground truth boxes from [x1, y1, w, h] to [x1, y1, x2, y2] format
            gt_box_iter = gt_box_iter.squeeze(1)
            gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])  # x2 = x1 + w
            gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])  # y2 = y1 + h
            gt_box_iter = gt_box_iter.cpu().numpy()
            box = box.squeeze(1).cpu().numpy()

            # Store image information for coordinate conversion
            info_iter = np.array(info_iter)

            # Convert predicted boxes from normalized coordinates to pixel coordinates
            for i in range(len(gt_box_iter)):
                box[i] = yolobox2label(box[i], info_iter[i])

            # Compute IoU between predicted and ground truth boxes
            box_iou = batch_box_iou(torch.from_numpy(gt_box_iter), torch.from_numpy(box)).cpu().numpy()

            # Update Box IoU metric at 0.5 threshold
            box_ap.update((box_iou > 0.5).astype(np.float32).mean() * 100., box_iou.shape[0])

            # Synchronize metrics across all processes in distributed evaluation
            reduce_meters(meters_dict, rank, cfg)

            # Log evaluation progress at specified intervals
            if (idx % cfg.train.log_period == 0 or idx == (len(data_loader) - 1)):
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Evaluation on {prefix}: [{idx}/{len(data_loader)}]  '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    f'BoxIoU@0.5 {box_ap.val:.4f} ({box_ap.avg:.4f})  '
                    f'Mem {memory_used:.0f}MB')
            
            # Update timing metrics
            sample_time.update((time.time() - end) / cfg.train.evaluation.eval_batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

        # Log final metrics to TensorBoard if main process
        if is_main_process() and writer is not None:
            writer.add_scalar("Acc/BoxIoU_0.5", box_ap.avg_reduce, global_step=epoch)

        # Log final evaluation results
        logger.info(f' * BoxIoU@0.5 {box_ap.avg_reduce:.3f}  Sample_time {sample_time.avg:.5f}')

    # Restore original model weights if EMA was used
    if ema is not None:
        ema.restore()
    return box_ap.avg_reduce


def main(cfg):
    """
    Main evaluation function for speech-referenced object detection models.
    
    This function sets up and runs the evaluation pipeline for speech-referenced
    object detection models. It handles dataset loading, model initialization,
    checkpoint loading, and distributed evaluation across multiple datasets.
    The function supports various evaluation splits (val, testA, testB, test) based
    on the dataset type.
    
    Args:
        cfg (LazyConfig): Configuration object containing all model, dataset, and
                         training parameters for evaluation
    
    Returns:
        None: Results are logged through the logger instance
    """
    # Initialize global variables for tracking best accuracy scores
    global best_det_acc, best_seg_acc
    best_det_acc, best_seg_acc = 0., 0.

    # Instantiate audio preprocessor for speech inputs
    audio_preprocessor = instantiate(cfg.preprocessor)

    # Build data loaders for validation and test sets based on dataset type
    loaders = []
    prefixs = ['val']
    
    # Always include validation set
    cfg.dataset.split = "val"
    val_set = instantiate(cfg.dataset)
    val_loader = build_test_speech_loader(cfg, val_set, shuffle=False, drop_last=False)
    loaders.append(val_loader)

    # Add test sets based on dataset type
    if cfg.dataset.dataset in ['refcoco_speech', 'refcoco+_speech']:
        # RefCOCO datasets have testA and testB splits
        cfg.dataset.split = "testA"
        testA_dataset = instantiate(cfg.dataset)
        testA_loader = build_test_speech_loader(cfg, testA_dataset, shuffle=False, drop_last=False)
        cfg.dataset.split = "testB"
        testB_dataset = instantiate(cfg.dataset)
        testB_loader = build_test_speech_loader(cfg, testB_dataset, shuffle=False, drop_last=False)
        prefixs.extend(['testA', 'testB'])
        loaders.extend([testA_loader, testB_loader])
    elif cfg.dataset.dataset in ['srefface', 'srefface+']:
        # SRFace datasets have testA split
        cfg.dataset.split = "testA"
        testA_dataset = instantiate(cfg.dataset)
        testA_loader = build_test_speech_loader(cfg, testA_dataset, shuffle=False, drop_last=False)
        prefixs.extend(['testA'])
        loaders.extend([testA_loader])
    else:
        # Other datasets have a single test split
        cfg.dataset.split = "test"
        test_dataset = instantiate(cfg.dataset)
        test_loader = build_test_speech_loader(cfg, test_dataset, shuffle=False, drop_last=False)
        prefixs.append('test')
        loaders.append(test_loader)

    # Instantiate the model for evaluation
    model = instantiate(cfg.model)

    # Setup distributed evaluation
    torch.cuda.set_device(dist.get_rank())
    model = DistributedDataParallel(model.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True)
    model_without_ddp = model.module  # Access to the original model without DDP wrapper

    # Log model information on main process
    if is_main_process():
        total_params = sum([param.nelement() for param in model.parameters()])
        trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logger.info(str(model))
        logger.info("Number of all params: %.2fM" % (total_params / 1e6))
        logger.info("Number of trainable params: %.2fM" % (trainable_params / 1e6))

    # Load model checkpoint and optimizer state
    checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage.cuda())
    model_without_ddp.load_state_dict(checkpoint['state_dict'])

    # TensorBoard writer setup (currently disabled)
    # if is_main_process():
    #     writer = SummaryWriter(log_dir=cfg.train.output_dir)
    # else:
    #     writer = None
    writer = None

    # Generate random sample IDs for visualization if enabled
    save_ids = np.random.randint(1, len(val_loader) * cfg.train.batch_size, 100) if cfg.train.log_image else None
    
    # Run evaluation on all datasets
    for data_loader, prefix in zip(loaders, prefixs):
        box_ap = validate(
            cfg=cfg,
            model=model,
            data_loader=data_loader,
            writer=writer,
            epoch=0,
            logger=logger,
            rank=dist.get_rank(),
            audio_preprocessor=audio_preprocessor,
            save_ids=save_ids,
            prefix=prefix)
        logger.info(f' * BoxIoU@0.5 {box_ap:.3f}')


if __name__ == '__main__':
    # Parse command line arguments for evaluation
    parser = argparse.ArgumentParser(description="csref_SREC")
    parser.add_argument('--config', type=str, required=True, default='./configs/csref_refcoco+_speech.py')
    parser.add_argument('--eval-weights', type=str, required=True, default='')
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
    
    # Load configuration and apply command-line overrides
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

    # Initialize distributed process group
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend=cfg.train.ddp.backend,
        init_method=cfg.train.ddp.init_method,
        world_size=world_size,
        rank=rank
    )
    torch.distributed.barrier()

    # Create output directories for evaluation results
    output_dir = cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "eval_result_log"), exist_ok=True)
    checkpoint_name = os.path.basename(args.eval_weights)
    logger = create_logger(output_dir=cfg.train.output_dir, dist_rank=dist.get_rank(), name=f"eval_{checkpoint_name}")

    # Set checkpoint path for evaluation
    cfg.train.resume_path = args.eval_weights
    logger.info(f"Running evaluation from specific checkpoint {cfg.train.resume_path}......")

    # Save evaluation configuration on main process
    if is_main_process():
        path = os.path.join(cfg.train.output_dir, "eval_result_log", "config_eval.yaml")
        LazyConfig.save(cfg, path)

    # Run main evaluation function
    main(cfg)
