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
import numpy as np

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------------------------------------------------------------
# reference:
# https://github.com/descriptinc/lyrebird-wav2clip/blob/1864b3924be5a785e2d49d975b8a26ff93f62951/wav2clip/pre_training/loss.py#L9
# -------------------------------------------------------------------------------------------------------------------------------

class CLIPLoss1D(nn.Module):
    """
    CLIP-inspired contrastive loss for 1D feature alignment.
    
    This loss function implements a contrastive learning approach similar to CLIP
    (Contrastive Language-Image Pre-training) but adapted for speech-text alignment.
    It maximizes the cosine similarity between matching speech-text pairs while
    minimizing it for non-matching pairs in the batch.
    
    Mathematical formulation:
    For a batch of N speech-text pairs:
    1. Compute normalized feature vectors for speech and text
    2. Calculate cosine similarity between all pairs in the batch
    3. Apply a learnable temperature parameter to scale the similarities
    4. Compute cross-entropy loss in both directions (speech->text and text->speech)
    5. Average the two losses
    """
    def __init__(self):
        """
        Initialize the CLIPLoss1D module.
        
        The temperature parameter is initialized to log(1/0.07), which corresponds
        to the default temperature used in the original CLIP model.
        """
        super(CLIPLoss1D, self).__init__()
        # Learnable temperature parameter for scaling the logits
        # Initialized to log(1/0.07) ≈ 2.659, which is the inverse of 0.07 (default CLIP temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # Cross-entropy loss for speech-to-text direction
        self.loss_audio = nn.CrossEntropyLoss()
        # Cross-entropy loss for text-to-speech direction
        self.loss_text = nn.CrossEntropyLoss()

    def forward(self, audio_features, text_features):
        """
        Compute the contrastive loss between speech and text features.
        
        Args:
            audio_features (torch.Tensor): Speech features of shape (batch_size, feature_dim)
            text_features (torch.Tensor): Text features of shape (batch_size, feature_dim)
            
        Returns:
            torch.Tensor: The computed contrastive loss
        """
        # Normalize features to unit length
        # This ensures that the cosine similarity is computed correctly
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity as logits
        # The learnable temperature parameter scales the logits to control the sharpness
        # of the distribution - higher values make the model more confident
        logit_scale = self.logit_scale.exp()
        # Compute similarity matrix from speech to text
        logits_per_image = logit_scale * audio_features @ text_features.t()
        # Compute similarity matrix from text to speech (transpose of the above)
        logits_per_text = logit_scale * text_features @ audio_features.t()

        # Create ground truth labels where each speech sample should match its
        # corresponding text sample in the batch
        batch_size = audio_features.shape[0]
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
        
        # Compute symmetric loss by averaging both directions
        # This ensures that the model learns bidirectional alignment between speech and text
        return (
                       self.loss_audio(logits_per_image, ground_truth)
                       + self.loss_text(logits_per_text, ground_truth)
               ) / 2