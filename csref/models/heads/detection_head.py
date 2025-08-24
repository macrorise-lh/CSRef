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

# --------------------------------------------------------
# References:
# https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
# --------------------------------------------------------

import math

import torch
import torch.nn as nn

from csref.layers.activation import get_activation

from ..utils.box_op import bboxes_iou
from ..losses.iou_loss import IOUloss


class BaseConv(nn.Module):
    """
    A basic convolutional block with Conv2d -> BatchNorm -> Activation.
    
    This block serves as a fundamental building unit in the detection head,
    providing feature extraction capabilities with normalization and non-linearity.
    
    Architecture:
    1. Convolutional layer with specified kernel size, stride, and padding
    2. Batch normalization for stable training
    3. Activation function (SiLU or Leaky ReLU) for non-linearity
    """

    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        """
        Initialize the BaseConv block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            ksize (int): Kernel size for the convolution
            stride (int): Stride for the convolution
            groups (int): Number of groups for grouped convolution. Defaults to 1.
            bias (bool): Whether to use bias in the convolution. Defaults to False.
            act (str): Activation function to use ('silu' or 'leaky_relu'). Defaults to 'silu'.
        """
        super().__init__()
        # Calculate same padding to maintain spatial dimensions
        # For odd kernel sizes, padding = (kernel_size - 1) // 2
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        # Batch normalization stabilizes training by normalizing activations
        self.bn = nn.BatchNorm2d(out_channels)
        # Activation function introduces non-linearity
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        """
        Forward pass of the BaseConv block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height', width')
                         where height' and width' depend on the stride
        """
        # Apply convolution, batch normalization, and activation in sequence
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        """
        Optimized forward pass for deployment, fusing convolution and activation.
        
        This method skips batch normalization for faster inference during deployment,
        which can be useful when the model is optimized for production environments.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """
    Depthwise Separable Convolution block.
    
    This block implements depthwise separable convolution, which is more efficient
    than standard convolution. It consists of:
    1. Depthwise convolution: Applies a single filter per input channel
    2. Pointwise convolution: Combines the results using 1x1 convolutions
    
    This architecture reduces computational cost and model size while maintaining
    similar performance to standard convolutions.
    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        """
        Initialize the DWConv block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            ksize (int): Kernel size for the depthwise convolution
            stride (int): Stride for the depthwise convolution. Defaults to 1.
            act (str): Activation function to use. Defaults to 'silu'.
        """
        super().__init__()
        # Depthwise convolution: applies a single filter per input channel
        # groups=in_channels makes it a depthwise convolution
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        # Pointwise convolution: 1x1 convolution to combine channels
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        """
        Forward pass of the DWConv block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height', width')
        """
        # Apply depthwise convolution followed by pointwise convolution
        x = self.dconv(x)
        return self.pconv(x)


class SREChead(nn.Module):
    """
    Speech Referring Expression Comprehension (SREC) detection head.
    
    
    Key features:
    1. Multi-scale feature processing: Handles features at different scales
    2. Regression and objectness prediction: Predicts bounding boxes and object presence
    3. Dynamic label assignment: Uses a sophisticated assignment strategy during training
    4. Loss computation: Combines IoU loss, objectness loss, and optional L1 loss
    
    Architecture:
    - Stem layers: Initial 1x1 convolutions to reduce channel dimensions
    - Regression convolutions: Two 3x3 convolutions for bounding box regression
    - Regression predictions: 1x1 convolution to predict 4 bounding box coordinates
    - Objectness predictions: 1x1 convolution to predict object presence score
    """
    def __init__(
            self,
            label_smooth=0.0,
            num_classes=0,
            width=1.0,
            strides=[32, ],
            in_channels=[512, ],
            act="silu",
            depthwise=False,
    ):
        """
        Initialize the SREChead detection head.
        
        Args:
            label_smooth (float): Label smoothing factor for training. Default: 0.0.
            num_classes (int): Number of object classes. Default: 0.
            width (float): Width multiplier for model scaling. Default: 1.0.
            strides (list): Strides for each feature level. Default: [32].
            in_channels (list): Number of input channels for each feature level. Default: [512].
            act (str): Activation function type ('silu' or 'leaky_relu'). Default: 'silu'.
            depthwise (bool): Whether to use depthwise convolution. Default: False.
        """
        super().__init__()
        self.label_smooth = label_smooth
        self.n_anchors = 1  # Number of anchors per position
        self.num_classes = num_classes
        self.decode_in_inference = True  # Whether to decode outputs during inference

        # Initialize module lists for different components
        self.cls_convs = nn.ModuleList()  # Classification convolution layers (not used in this implementation)
        self.reg_convs = nn.ModuleList()  # Regression convolution layers
        self.cls_preds = nn.ModuleList()  # Classification prediction layers (not used)
        self.reg_preds = nn.ModuleList()  # Regression prediction layers
        self.obj_preds = nn.ModuleList()  # Objectness prediction layers
        self.stems = nn.ModuleList()      # Stem layers for initial processing
        
        # Choose convolution type based on depthwise flag
        Conv = DWConv if depthwise else BaseConv

        # Create network components for each input feature level
        for i in range(len(in_channels)):
            # Stem layer: 1x1 convolution to reduce channel dimensions
            self.stems = nn.ModuleList(
                [BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                ), ]
            )
            
            # Regression convolutions: Two 3x3 convolutions for feature refinement
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            
            # Regression predictions: 1x1 convolution to predict 4 bounding box coordinates (x, y, w, h)
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            
            # Objectness predictions: 1x1 convolution to predict object presence score
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        
        # Loss functions
        self.use_l1 = False  # Whether to use L1 loss (not used in this implementation)
        self.l1_loss = nn.L1Loss(reduction="none")  # L1 loss for bounding box regression
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")  # Binary cross-entropy with logits for objectness
        self.iou_loss = IOUloss(reduction="none")  # IoU loss for bounding box regression
        
        # Strides and grids for anchor generation
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        """
        Initialize the biases of the objectness prediction layers.
        
        This method initializes the biases to reflect a prior probability of an object
        being present at each location. This helps with training stability and convergence.
        
        Args:
            prior_prob (float): Prior probability of an object being present
        """
        # For each objectness prediction layer
        for conv in self.obj_preds:
            # Reshape the bias to match the anchor structure
            b = conv.bias.view(self.n_anchors, -1)
            # Initialize bias based on prior probability using the formula:
            # b = -log((1 - prior_prob) / prior_prob)
            # This ensures that the initial sigmoid output is approximately prior_prob
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            # Reshape back and set as a learnable parameter
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None, ):
        """
        Forward pass of the SREChead detection head.
        
        This method processes the input features through the detection head network,
        generating either loss values during training or bounding box predictions during inference.
        
        Args:
            xin (torch.Tensor or list): Input feature maps from the backbone
            labels (torch.Tensor, optional): Ground truth labels for training. Defaults to None.
            imgs (torch.Tensor, optional): Input images for visualization. Defaults to None.
            
        Returns:
            During training:
                torch.Tensor: Computed loss value
            During inference:
                torch.Tensor: Predicted bounding boxes in format [x1, y1, x2, y2]
        """
        # Initialize output containers
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        
        # Scale labels to match the feature map resolution if provided
        if labels is not None:
            labels *= xin.size()[-1] * self.strides[0]
            
        # Convert single tensor to list for consistent processing
        if isinstance(xin, torch.Tensor):
            xin = [xin]

        # Process each feature level
        for k, (reg_conv, stride_this_level, x) in enumerate(
                zip(self.reg_convs, self.strides, xin)
        ):
            # Apply stem layer to reduce channel dimensions
            x = self.stems[k](x)
            # Use the same features for regression (classification is not used in this implementation)
            reg_x = x
            
            # Apply regression convolutions for feature refinement
            reg_feat = reg_conv(reg_x)
            # Generate regression predictions (bounding box coordinates)
            reg_output = self.reg_preds[k](reg_feat)
            # Generate objectness predictions (object presence score)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                # During training: concatenate regression and objectness predictions
                output = torch.cat([reg_output, obj_output], 1)
                # Convert predictions to absolute coordinates and generate grid
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                # Store grid coordinates for each feature level
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                # Store stride information for each feature level
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                        .fill_(stride_this_level)
                        .type_as(xin[0])
                )
                
                # If using L1 loss, store original regression predictions
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                # During inference: concatenate regression and sigmoid-activated objectness predictions
                output = torch.cat(
                    [reg_output, obj_output.sigmoid()], 1
                )

            # Store output for this feature level
            outputs.append(output)

        if self.training:
            # During training: compute and return loss
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            # During inference: decode predictions to get final bounding boxes
            # Store height and width of each feature level
            self.hw = [x.shape[-2:] for x in outputs]
            # Concatenate outputs from all feature levels
            # [batch, n_anchors_all, 5] where 5 = 4 (bbox) + 1 (obj_score)
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            # Decode outputs from relative coordinates to absolute coordinates
            outputs = self.decode_outputs(outputs, dtype=xin[0].type())
            
            # Convert from center coordinates (x, y, w, h) to corner coordinates (x1, y1, x2, y2)
            outputs[:, :, 0] = outputs[:, :, 0] - outputs[:, :, 2] / 2  # x1 = x - w/2
            outputs[:, :, 1] = outputs[:, :, 1] - outputs[:, :, 3] / 2  # y1 = y - h/2
            outputs[:, :, 2] = outputs[:, :, 0] + outputs[:, :, 2]      # x2 = x1 + w
            outputs[:, :, 3] = outputs[:, :, 1] + outputs[:, :, 3]      # y2 = y1 + h
            
            # Select the prediction with the highest objectness score for each image
            score = outputs[:, :, 4]
            batchsize = outputs.size()[0]
            # Get indices of the highest scoring predictions
            ind = torch.argmax(score, -1).unsqueeze(1).unsqueeze(1).repeat(1, 1, outputs.size()[-1])
            # Gather the highest scoring predictions
            pred = torch.gather(outputs, 1, ind)
            
            # Return the highest scoring bounding box for each image
            return pred.view(batchsize, -1)

    def get_output_and_grid(self, output, k, stride, dtype):
        """
        Convert network outputs to absolute coordinates and generate grid.
        
        This method is used during training to convert the relative coordinate predictions
        from the network to absolute coordinates in the image space. It also generates
        a grid of anchor positions for computing the training targets.
        
        Args:
            output (torch.Tensor): Network output tensor
            k (int): Index of the current feature level
            stride (int): Stride of the current feature level
            dtype (torch.dtype): Data type for the grid
            
        Returns:
            tuple: (decoded_output, grid)
                decoded_output (torch.Tensor): Output in absolute coordinates
                grid (torch.Tensor): Grid of anchor positions
        """
        # Get the grid for the current feature level
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes  # 4 (bbox) + 1 (obj) + num_classes
        hsize, wsize = output.shape[-2:]
        
        # If the grid size doesn't match the output size, regenerate it
        if grid.shape[2:4] != output.shape[2:4]:
            # Create a meshgrid of coordinates
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # Stack and reshape to create a grid of shape [1, 1, hsize, wsize, 2]
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        # Reshape output to [batch_size, n_anchors, n_ch, hsize, wsize]
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        # Permute and reshape to [batch_size, n_anchors * hsize * wsize, n_ch]
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        # Reshape grid to [1, n_anchors * hsize * wsize, 2]
        grid = grid.view(1, -1, 2)
        
        # Decode center coordinates (x, y) from relative to absolute
        # Formula: absolute_x = (relative_x + grid_x) * stride
        output[..., :2] = (output[..., :2] + grid) * stride
        # Decode width and height using exponential to ensure positive values
        # Formula: absolute_w = exp(relative_w) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        
        return output, grid

    def decode_outputs(self, outputs, dtype):
        """
        Decode network outputs to absolute coordinates during inference.
        
        This method is used during inference to convert the relative coordinate predictions
        from the network to absolute coordinates in the image space. It handles multiple
        feature levels and combines them into a single output.
        
        Args:
            outputs (torch.Tensor): Network output tensor
            dtype (torch.dtype): Data type for the grids and strides
            
        Returns:
            torch.Tensor: Decoded outputs in absolute coordinates
        """
        grids = []
        strides = []
        
        # Generate grids and strides for each feature level
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            # Create a meshgrid of coordinates
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # Stack and reshape to create a grid of shape [1, hsize * wsize, 2]
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            # Create a tensor filled with the stride value
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        # Concatenate grids and strides from all feature levels
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        # Decode center coordinates (x, y) from relative to absolute
        # Formula: absolute_x = (relative_x + grid_x) * stride
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        # Decode width and height using exponential to ensure positive values
        # Formula: absolute_w = exp(relative_w) * stride
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        
        return outputs

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
    ):
        """
        Compute the detection loss during training.
        
        This method calculates the loss for the detection head, which includes:
        1. IoU loss for bounding box regression
        2. Binary cross-entropy loss for objectness prediction
        3. Optional L1 loss for bounding box regression
        
        The method uses a dynamic label assignment strategy to match anchors
        to ground truth boxes and computes the loss only for positive samples.
        
        Args:
            imgs (torch.Tensor): Input images (for visualization purposes)
            x_shifts (list): X coordinates of anchor centers for each feature level
            y_shifts (list): Y coordinates of anchor centers for each feature level
            expanded_strides (list): Strides for each feature level
            labels (torch.Tensor): Ground truth labels
            outputs (torch.Tensor): Network outputs
            origin_preds (list): Original regression predictions (for L1 loss)
            dtype (torch.dtype): Data type for computations
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Extract bounding box predictions [batch, n_anchors_all, 4]
        bbox_preds = outputs[:, :, :4]
        # Extract objectness predictions [batch, n_anchors_all, 1]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)
        # Classification predictions are not used in this implementation
        # cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # Calculate number of objects per image
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)

        total_num_anchors = outputs.shape[1]
        # Concatenate coordinate shifts and strides from all feature levels
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        # Initialize target containers
        # cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        # Counters for positive samples and ground truth boxes
        num_fg = 0.0
        num_gts = 0.0

        # Process each image in the batch
        for batch_idx in range(outputs.shape[0]):
            # Get number of ground truth boxes for this image
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            
            if num_gt == 0:
                # If no ground truth boxes, create empty targets
                # cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                # Extract ground truth bounding boxes for this image
                gt_bboxes_per_image = labels[batch_idx, :num_gt, :4]
                # Extract bounding box predictions for this image
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    # Try to assign anchors to ground truth boxes on GPU
                    (
                        fg_mask,          # Foreground mask indicating positive anchors
                        pred_ious_this_matching,  # IoU between predictions and matched ground truth
                        matched_gt_inds,  # Indices of matched ground truth boxes
                        num_fg_img,       # Number of positive anchors for this image
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        None,  # Classification predictions are not used
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        None,  # Classification predictions are not used
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    # If GPU runs out of memory, fall back to CPU
                    torch.cuda.empty_cache()
                    (
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        None,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        None,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                # Clear GPU cache
                torch.cuda.empty_cache()
                # Update total number of positive anchors
                num_fg += num_fg_img

                # Create objectness targets (1 for positive anchors, 0 for negative)
                obj_target = fg_mask.unsqueeze(-1)
                # Create regression targets (matched ground truth boxes)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                
                # If using L1 loss, create L1 targets
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            # Append targets for this image (classification targets are not used)
            # cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        # Concatenate targets from all images
        # cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        # Ensure at least one positive sample to avoid division by zero
        num_fg = max(num_fg, 1)
        # Compute IoU loss for bounding box regression
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        # Compute binary cross-entropy loss for objectness prediction with optional label smoothing
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets.clamp(self.label_smooth,
                                                                                     1. - self.label_smooth) if self.label_smooth > 0. else obj_targets)
                   ).sum() / num_fg

        # Compute L1 loss if enabled
        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        # Combine losses with weights
        reg_weight = 5.0  # Weight for regression loss
        loss = reg_weight * loss_iou + loss_obj + loss_l1

        return loss

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        """
        Generate L1 targets for bounding box regression.
        
        This method converts ground truth bounding boxes to the format expected
        by the L1 loss function. It transforms absolute coordinates to relative
        coordinates with respect to the anchor positions.
        
        Args:
            l1_target (torch.Tensor): Empty tensor to be filled with L1 targets
            gt (torch.Tensor): Ground truth bounding boxes in format [x, y, w, h]
            stride (int): Stride of the feature level
            x_shifts (torch.Tensor): X coordinates of anchor centers
            y_shifts (torch.Tensor): Y coordinates of anchor centers
            eps (float): Small value to avoid log(0). Defaults to 1e-8.
            
        Returns:
            torch.Tensor: L1 targets for bounding box regression
        """
        # Convert center coordinates to relative values with respect to anchor positions
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts  # x_center relative to anchor
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts  # y_center relative to anchor
        # Convert width and height to log-space relative values
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)  # log(width / stride)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)  # log(height / stride)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):
        """
        Assign anchors to ground truth boxes for training.
        
        This method implements a dynamic label assignment strategy that matches
        anchors to ground truth boxes based on IoU and spatial constraints. It uses
        a cost-based approach to determine the best matches.
        
        Args:
            batch_idx (int): Index of the current batch
            num_gt (int): Number of ground truth boxes
            total_num_anchors (int): Total number of anchors across all feature levels
            gt_bboxes_per_image (torch.Tensor): Ground truth bounding boxes
            gt_classes (torch.Tensor): Ground truth classes (not used in this implementation)
            bboxes_preds_per_image (torch.Tensor): Predicted bounding boxes
            expanded_strides (torch.Tensor): Strides for each feature level
            x_shifts (torch.Tensor): X coordinates of anchor centers
            y_shifts (torch.Tensor): Y coordinates of anchor centers
            cls_preds (torch.Tensor): Classification predictions (not used)
            bbox_preds (torch.Tensor): Bounding box predictions
            obj_preds (torch.Tensor): Objectness predictions
            labels (torch.Tensor): Ground truth labels
            imgs (torch.Tensor): Input images (not used)
            mode (str): Computation mode ('gpu' or 'cpu'). Defaults to 'gpu'.
            
        Returns:
            tuple: (fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg)
                fg_mask (torch.Tensor): Foreground mask indicating positive anchors
                pred_ious_this_matching (torch.Tensor): IoU between predictions and matched ground truth
                matched_gt_inds (torch.Tensor): Indices of matched ground truth boxes
                num_fg (int): Number of positive anchors
        """
        # If CPU mode is specified, move tensors to CPU
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            # gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # Get information about which anchors are inside ground truth boxes
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        # Filter predictions to only those from foreground anchors
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]

        # Get objectness predictions for foreground anchors
        obj_preds_ = obj_preds[batch_idx][fg_mask]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # Compute pairwise IoU between ground truth boxes and predictions
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        # Convert IoU to IoU loss (negative log IoU)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            # cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()
            obj_preds_ = obj_preds_.cpu()

        # Compute assignment cost based on IoU loss and spatial constraints
        cost = (3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)  # Large penalty for anchors outside boxes
                )

        # Perform dynamic k-matching to assign anchors to ground truth boxes
        (
            num_fg,                  # Number of positive anchors
            pred_ious_this_matching,  # IoU between predictions and matched ground truth
            matched_gt_inds,        # Indices of matched ground truth boxes
        ) = self.dynamic_k_matching(cost, pair_wise_ious, num_gt, fg_mask)
        
        # Clean up memory
        del cost, pair_wise_ious, pair_wise_ious_loss

        # If CPU mode was used, move results back to GPU
        if mode == "cpu":
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        """
        Determine which anchors are inside ground truth boxes.
        
        This method implements a spatial constraint strategy to identify anchors
        that are likely to be good matches for ground truth boxes. It uses two criteria:
        1. Anchors that are inside the ground truth boxes
        2. Anchors that are within a fixed center radius of the ground truth box centers
        
        Args:
            gt_bboxes_per_image (torch.Tensor): Ground truth bounding boxes
            expanded_strides (torch.Tensor): Strides for each feature level
            x_shifts (torch.Tensor): X coordinates of anchor centers
            y_shifts (torch.Tensor): Y coordinates of anchor centers
            total_num_anchors (int): Total number of anchors across all feature levels
            num_gt (int): Number of ground truth boxes
            
        Returns:
            tuple: (is_in_boxes_anchor, is_in_boxes_and_center)
                is_in_boxes_anchor (torch.Tensor): Mask indicating anchors that are either
                                                 inside boxes or within center radius
                is_in_boxes_and_center (torch.Tensor): Mask indicating anchors that are both
                                                      inside boxes and within center radius
        """
        # Get stride and coordinate shifts for the current image
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        
        # Compute anchor center coordinates in image space
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )

        # Compute ground truth box boundaries
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )

        # Compute distances from anchor centers to box boundaries
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        # Identify anchors that are inside ground truth boxes
        # An anchor is inside a box if all distances to boundaries are positive
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0  # At least one ground truth box
        
        # Fixed center region constraint
        center_radius = 2.5  # Radius multiplier for center region

        # Compute center region boundaries
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        # Compute distances from anchor centers to center region boundaries
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        # Identify anchors that are within center regions
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0  # At least one ground truth box

        # Combine constraints: anchors must be either inside boxes or within center regions
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        # Further refine: anchors must be both inside boxes and within center regions
        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt, fg_mask):
        """
        Implement dynamic k-matching for anchor assignment.
        
        This method implements a dynamic label assignment strategy that assigns
        a variable number of anchors to each ground truth box based on the IoU
        between the anchors and the ground truth box. The number of anchors assigned
        to each ground truth box is proportional to the sum of the top-k IoUs.
        
        Args:
            cost (torch.Tensor): Assignment cost matrix
            pair_wise_ious (torch.Tensor): Pairwise IoU between anchors and ground truth boxes
            num_gt (int): Number of ground truth boxes
            fg_mask (torch.Tensor): Foreground mask indicating candidate anchors
            
        Returns:
            tuple: (num_fg, pred_ious_this_matching, matched_gt_inds)
                num_fg (int): Number of positive anchors
                pred_ious_this_matching (torch.Tensor): IoU between predictions and matched ground truth
                matched_gt_inds (torch.Tensor): Indices of matched ground truth boxes
        """
        # Dynamic K-matching algorithm
        # ---------------------------------------------------------------
        # Initialize matching matrix
        matching_matrix = torch.zeros_like(cost)

        # Use pairwise IoUs to determine the number of anchors to assign to each ground truth
        ious_in_boxes_matrix = pair_wise_ious
        # Limit the number of candidate anchors to at most 10
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        # Get top-k IoUs for each ground truth box
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        # Determine the number of anchors to assign to each ground truth box
        # This is proportional to the sum of the top-k IoUs, with a minimum of 1
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        
        # Assign anchors to ground truth boxes based on cost
        for gt_idx in range(num_gt):
            # For each ground truth box, select the k anchors with the lowest cost
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            # Mark these anchors as assigned to this ground truth box
            matching_matrix[gt_idx][pos_idx] = 1.0

        # Clean up memory
        del topk_ious, dynamic_ks, pos_idx

        # Resolve conflicts where an anchor is assigned to multiple ground truth boxes
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            # For anchors assigned to multiple ground truth boxes,
            # keep only the assignment with the lowest cost
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            
        # Identify positive anchors (those assigned to at least one ground truth box)
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        # Update the foreground mask
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        # For each positive anchor, find the index of the matched ground truth box
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        # Compute the IoU between each positive anchor and its matched ground truth box
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, pred_ious_this_matching, matched_gt_inds
