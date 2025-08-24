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
import torch.nn as nn

from csref.layers.blocks import ConvBnAct, C3Block
from csref.layers.sppf import SPPF


class CspDarkNet(nn.Module):
    """
    CspDarkNet: A Cross Stage Partial DarkNet backbone for visual feature extraction.
    
    This backbone network is based on the CSPDarknet architecture, which is known for
    its efficiency and effectiveness in object detection tasks. The architecture uses
    Cross Stage Partial connections to reduce computational cost while maintaining
    high accuracy.
    
    Key features:
    1. Progressive downsampling: The network progressively reduces spatial dimensions
       while increasing channel dimensions, capturing features at multiple scales.
    2. C3 blocks: These blocks use cross-stage partial connections to improve gradient
       flow and reduce computation.
    3. SPPF module: Spatial Pyramid Pooling Fast module at the end captures multi-scale
       context information.
    4. Multi-scale outputs: The network outputs features at different scales for use in
       the detection head.
    
    Architecture overview:
    - Input: (3, H, W) RGB image
    - Output: Multi-scale feature maps at 1/8, 1/16, and 1/32 of input resolution
    - Total downsampling factor: 32x
    """
    def __init__(
            self,
            pretrained_weight_path=None,
            pretrained=False,
            multi_scale_outputs=False,
            freeze_backbone=True,
    ):
        """
        Initialize the CspDarkNet backbone.
        
        Args:
            pretrained_weight_path (str, optional): Path to pretrained weights. Defaults to None.
            pretrained (bool): Whether to load pretrained weights. Defaults to False.
            multi_scale_outputs (bool): Whether to return multi-scale feature maps. Defaults to False.
            freeze_backbone (bool): Whether to freeze the backbone parameters. Defaults to True.
        """
        super().__init__()
        # Construct the backbone network as a sequential model
        self.model = nn.Sequential(
            # Initial convolution with large kernel (6x6) and stride 2
            # Input: (3, H, W) -> Output: (64, H/2, W/2)
            ConvBnAct(c1=3, c2=64, k=6, s=2, p=2),  # /2
            # i = 0 ch = [64]
            
            # Downsample to 1/4 resolution
            # Input: (64, H/2, W/2) -> Output: (128, H/4, W/4)
            ConvBnAct(c1=64, c2=128, k=3, s=2),  # /2
            # i = 1 ch = [64,128]
            
            # C3 block with 3 repetitions
            # Maintains resolution at 1/4
            C3Block(c1=128, c2=128, n=3),
            # i = 2 ch =[64,128,128]
            
            # Downsample to 1/8 resolution
            # Input: (128, H/4, W/4) -> Output: (256, H/8, W/8)
            ConvBnAct(c1=128, c2=256, k=3, s=2),  # /2
            # i = 3 ch =[64,128,128,256]
            
            # C3 block with 6 repetitions
            # Maintains resolution at 1/8 - this is one of the output scales
            C3Block(c1=256, c2=256, n=6),
            # i = 4 ch =[64,128,128,256,256]
            
            # Downsample to 1/16 resolution
            # Input: (256, H/8, W/8) -> Output: (512, H/16, W/16)
            ConvBnAct(c1=256, c2=512, k=3, s=2),  # /2
            # i = 5 ch =[64,128,128,256,256,512]
            
            # C3 block with 9 repetitions
            # Maintains resolution at 1/16 - this is another output scale
            C3Block(c1=512, c2=512, n=9),
            # i = 6 ch =[64,128,128,256,256,512,512]
            
            # Downsample to 1/32 resolution
            # Input: (512, H/16, W/16) -> Output: (1024, H/32, W/32)
            ConvBnAct(c1=512, c2=1024, k=3, s=2),  # /2
            # i = 7 ch =[64,128,128,256,256,512,512,1024]
            
            # C3 block with 3 repetitions
            # Maintains resolution at 1/32
            C3Block(c1=1024, c2=1024, n=3),
            # i = 8 ch =[64,128,128,256,256,512,512,1024,1024]
            
            # Spatial Pyramid Pooling Fast module
            # Captures multi-scale context at the highest level of abstraction
            # Input: (1024, H/32, W/32) -> Output: (1024, H/32, W/32)
            SPPF(c1=1024, c2=1024, k=5),
            # i = 9 ch =[64,128,128,256,256,512,512,1024,1024,1024]
        )

        self.multi_scale_outputs = multi_scale_outputs

        # Load pretrained weights if specified
        if pretrained:
            self.weight_dict = torch.load(pretrained_weight_path)
            self.load_state_dict(self.weight_dict, strict=False)

        # Freeze backbone parameters if specified (except the last two layers)
        # This is useful in transfer learning scenarios where we want to keep
        # the early layers fixed and only fine-tune the later layers
        if freeze_backbone:
            self.frozen(self.model[:-2])

    def frozen(self, module):
        """
        Freeze the parameters of a module to prevent gradient updates during training.
        
        Args:
            module (nn.Module): The module whose parameters should be frozen
        """
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass of the CspDarkNet backbone.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor or list: If multi_scale_outputs is True, returns a list of
                                 feature maps at different scales [1/8, 1/16, 1/32].
                                 Otherwise, returns only the final feature map at 1/32 scale.
        """
        outputs = []
        # Process input through each layer in the sequential model
        for i, module in enumerate(self.model):
            x = module(x)
            # Collect intermediate feature maps at scales 1/8 (i=4) and 1/16 (i=6)
            # These are used for multi-scale feature fusion in the detection head
            if i in [4, 6]:
                outputs.append(x)
        # Add the final feature map at 1/32 scale
        outputs.append(x)
        
        # Return either multi-scale outputs or just the final feature map
        if self.multi_scale_outputs:
            return outputs
        else:
            return x
