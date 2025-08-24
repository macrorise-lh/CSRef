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

import torch
import torch.nn as nn

torch.backends.cudnn.enabled = False


# Speech Referring Expression Comprehension (SREC) stage
class CSRef(nn.Module):
    """
    CSRef: A multi-modal model for Speech Referring Expression Comprehension.
    
    The architecture consists of:
    1. Visual encoder for extracting visual features from images
    2. Speech encoder for processing audio inputs and extracting speech semantic representations
    3. Multi-scale feature processing for handling different visual resolutions
    4. Fusion mechanisms for combining visual and speech representations
    5. Attention mechanisms for focusing on relevant features
    6. Detection head for generating bounding box predictions
    """
    def __init__(
            self,
            visual_backbone: nn.Module,
            speech_encoder: nn.Module,
            multi_scale_manner: nn.Module,
            fusion_manner: nn.Module,
            attention_manner: nn.Module,
            head: nn.Module,
    ):
        """
        Initialize the CSRef model with its components.
        
        Args:
            visual_backbone (nn.Module): Backbone network for visual feature extraction (e.g., CSPDarknet)
            speech_encoder (nn.Module): Encoder for processing speech inputs (e.g., Wav2Vec2)
            multi_scale_manner (nn.Module): Module for processing multi-scale visual features
            fusion_manner (nn.Module): Module for fusing visual and speech features
            attention_manner (nn.Module): Attention mechanism for feature refinement
            head (nn.Module): Detection head for generating final predictions
        """
        super(CSRef, self).__init__()
        self.visual_encoder = visual_backbone
        self.speech_encoder = speech_encoder
        self.multi_scale_manner = multi_scale_manner
        self.fusion_manner = fusion_manner
        self.attention_manner = attention_manner
        self.head = head

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

    def forward(self, x, y, audio_mask, det_label=None):
        """
        Forward pass of the CSRef model.
        
        Args:
            x (torch.Tensor): Visual input tensor of shape (batch_size, channels, height, width)
            y (torch.Tensor): Speech input tensor
            audio_mask (torch.Tensor): Mask for speech input to handle variable lengths
            det_label (torch.Tensor, optional): Detection labels for training. Defaults to None.
            
        Returns:
            torch.Tensor: During training, returns the computed loss.
                         During inference, returns predicted bounding boxes.
        """
        # == Vision and Language Encoding ==
        # Extract visual features at multiple scales from the input image
        x = self.visual_encoder(x)
        # Extract speech features from the audio input, considering the mask for variable lengths
        y = self.speech_encoder(y, audio_mask)

        # == Vision and Language Fusion ==
        # Fuse visual features with speech features at each scale
        # This enables the model to attend to relevant visual regions based on speech content
        for i in range(len(self.fusion_manner)):
            x[i] = self.fusion_manner[i](x[i], y['flat_feat'])

        # Process multi-scale visual features to combine information across different resolutions
        x = self.multi_scale_manner(x)

        # Apply attention mechanism to focus on the most relevant features
        # The attention mechanism uses speech features to guide attention on visual features
        top_feats, _, _ = self.attention_manner(y['flat_feat'], x[-1])

        # == Output Generation ==
        if self.training:
            # During training, compute the loss based on detection labels
            loss = self.head(top_feats, labels=det_label)
            return loss
        else:
            # During inference, generate bounding box predictions
            box = self.head(top_feats)
            return box
