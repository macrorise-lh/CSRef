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

from csref.models.losses.clip_loss import CLIPLoss1D

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ContrastiveSemanticAlignment (CSA) stage
class ContrastiveSemanticAlignment(nn.Module):
    """
    Contrastive Semantic Alignment (CSA) module for aligning speech and text modalities.
    
    This module implements the first stage of the two-stage training approach in CSRef.
    It learns to align speech and text representations in a shared embedding space
    using contrastive learning. This alignment enables the model to understand the
    semantic relationship between spoken and textual descriptions of the same content.
    
    The CSA stage is crucial for:
    1. Learning shared representations between speech and text
    2. Enabling cross-modal understanding
    3. Providing a foundation for the subsequent SREC stage
    """
    def __init__(
            self,
            speech_encoder: nn.Module,
            text_encoder: nn.Module,
    ):
        """
        Initialize the Contrastive Semantic Alignment module.
        
        Args:
            speech_encoder (nn.Module): Encoder for processing speech inputs (e.g., Wav2Vec2)
            text_encoder (nn.Module): Encoder for processing text inputs (e.g., BERT)
        """
        super(ContrastiveSemanticAlignment, self).__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder

        # Initialize the contrastive loss function for alignment
        self.contrastive_loss = CLIPLoss1D()

    def frozen(self, module):
        """
        Freeze the parameters of a module to prevent gradient updates during training.
        
        This is used in the two-stage training approach where certain components
        may be frozen during specific stages of training.
        
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

    def forward(self, audios, audios_mask, text_ids, text_ids_mask):
        """
        Forward pass of the Contrastive Semantic Alignment module.
        
        Args:
            audios (torch.Tensor): Speech input tensor of shape (batch_size, sequence_length, features)
            audios_mask (torch.Tensor): Mask for speech input to handle variable lengths
            text_ids (torch.Tensor): Text input token IDs of shape (batch_size, sequence_length)
            text_ids_mask (torch.Tensor): Mask for text input to handle variable lengths
            
        Returns:
            torch.Tensor: The computed contrastive loss between speech and text representations
        """
        # Stage 1: Speech and Text Encoding
        # Encode speech inputs to extract speech features
        x = self.speech_encoder(audios, audios_mask)
        # Encode text inputs to extract text features
        y = self.text_encoder(text_ids, text_ids_mask)

        # Extract flattened speech features for contrastive learning
        x = x['flat_feat']

        # Compute contrastive loss between speech and text representations
        # This loss encourages matching speech-text pairs to have similar representations
        # in the shared embedding space, while mismatching pairs have dissimilar representations
        loss = self.contrastive_loss(x, y)

        return loss


