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

from csref.layers.sa_layer import AttFlat

from transformers import Wav2Vec2Model
from transformers.activations import GELUActivation


class GELUConv(nn.Module):
    """
    A convolutional block with Conv2d -> BatchNorm -> GELU activation.
    
    This block serves as a building unit for processing multi-dimensional features,
    particularly useful for fusing information from different hidden states of the
    Wav2Vec2 model. It optionally includes a shortcut connection for residual learning.
    
    Architecture:
    1. Convolutional layer with specified kernel size, stride, and padding
    2. Batch normalization for stable training
    3. GELU activation for non-linearity (smoother than ReLU)
    4. Optional shortcut connection for residual learning
    """
    def __init__(
            self, in_channels, out_channels, ksize, stride, shortcut=False, groups=1, bias=False
    ):
        """
        Initialize the GELUConv block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            ksize (int): Kernel size for the convolution
            stride (int): Stride for the convolution
            shortcut (bool): Whether to use a shortcut connection. Defaults to False.
            groups (int): Number of groups for grouped convolution. Defaults to 1.
            bias (bool): Whether to use bias in the convolution. Defaults to False.
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
        # GELU activation provides smooth non-linearity
        self.act = GELUActivation()

        # If using shortcut connection, ensure input and output channels match
        if shortcut:
            assert in_channels == out_channels

        # Store whether to add shortcut connection
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        """
        Forward pass of the GELUConv block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height', width')
                         where height' and width' depend on the stride
        """
        # Apply convolution, batch normalization, and activation
        # If shortcut is enabled, add input to convolution output before normalization
        return self.act(self.bn(x + self.conv(x) if self.add else self.conv(x)))


def make_mask(feature):
    """
    Create a mask for zero-padded features.
    
    This function generates a binary mask that identifies which positions in the
    feature tensor are zero-padded. This is useful for handling variable-length
    sequences in transformer models.
    
    Args:
        feature (torch.Tensor): Input feature tensor of shape (batch_size, sequence_length, feature_dim)
        
    Returns:
        torch.Tensor: Binary mask of shape (batch_size, 1, 1, sequence_length)
                     where 1 indicates a padded position and 0 indicates a valid position
    """
    # Sum absolute values along the feature dimension
    # If all features are zero at a position, it's likely a padded position
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


class Wav2vec2(nn.Module):
    """
    Wav2Vec2 speech encoder with feature extraction and attention mechanisms.
    
    This class wraps the Hugging Face Wav2Vec2 model and extends it with additional
    components for speech feature extraction in the CSRef model. It provides flexibility
    in how features are extracted from the model's hidden states and includes mechanisms
    for handling variable-length sequences.
    
    Key features:
    1. Pre-trained Wav2Vec2 model as the base encoder
    2. Flexible feature extraction from different hidden states
    3. Optional fusion of multiple hidden states
    4. Attention-based flattening of sequence features
    5. Support for freezing different parts of the model for transfer learning
    
    Architecture:
    - Base Wav2Vec2 model for speech encoding
    - Optional fusion modules for combining multiple hidden states
    - Attention-based flattening module for converting sequence features to fixed-size vectors
    """
    def __init__(
            self,
            hidden_size=1024,
            flat_glimpses=1,
            dropout_rate=0.1,
            target_sr=16000,
            pretrained_path="data/weights/wav2vec2-base-960h",
            freeze_model=False,
            use_one_hidden_state_as_feat=True,
            hidden_state_index=-1,
            use_att_flat_mask=True,
            fusion_times=1,
            freeze_layers=None,  # freeze the first few layers of encoder.layers
            short_cut=False,
    ):
        """
        Initialize the Wav2vec2 speech encoder.
        
        Args:
            hidden_size (int): Hidden size of the Wav2Vec2 model. Defaults to 1024.
            flat_glimpses (int): Number of glimpses for attention flattening. Defaults to 1.
            dropout_rate (float): Dropout rate for attention flattening. Defaults to 0.1.
            target_sr (int): Target sample rate for audio input. Defaults to 16000.
            pretrained_path (str): Path to pre-trained Wav2Vec2 model. Defaults to "data/weights/wav2vec2-base-960h".
            freeze_model (bool): Whether to freeze the model parameters. Defaults to False.
            use_one_hidden_state_as_feat (bool): Whether to use a single hidden state as features. Defaults to True.
            hidden_state_index (int): Index of hidden state to use as features. Defaults to -1 (last layer).
            use_att_flat_mask (bool): Whether to use mask in attention flattening. Defaults to True.
            fusion_times (int): Number of times to fuse features when using multiple hidden states. Defaults to 1.
            freeze_layers (int, optional): Number of encoder layers to freeze. Defaults to None.
            short_cut (bool): Whether to use shortcut connections in fusion modules. Defaults to False.
        """
        super(Wav2vec2, self).__init__()

        self.hidden_size = hidden_size
        self.target_sample_rate = target_sr
        self.use_one_hidden_state_as_feat = use_one_hidden_state_as_feat

        # The index of hidden states to use as features.
        # When use_one_hidden_state_as_feat is True, it refers to outputs.hidden_states[hidden_state_index];
        # when False, it refers to outputs.hidden_states[hidden_state_index:]
        self.hidden_state_index = hidden_state_index
        self.use_att_flat_mask = use_att_flat_mask
        self.fusion_times = fusion_times

        # Load pre-trained Wav2Vec2 model
        self.model = Wav2Vec2Model.from_pretrained(pretrained_path, gradient_checkpointing=False)

        # Freeze model parameters if specified
        if freeze_model:
            if freeze_layers is not None:
                # Freeze specific components of the model
                self.model.masked_spec_embed.required_grad = False
                self.frozen(self.model.feature_extractor)
                self.frozen(self.model.feature_projection)
                self.frozen(self.model.encoder.pos_conv_embed)
                self.frozen(self.model.encoder.layer_norm)
                self.frozen(self.model.encoder.layers[:freeze_layers])
            else:
                # Freeze only the feature encoder
                self.model.freeze_feature_encoder()

        # If using multiple hidden states, create fusion modules
        if not use_one_hidden_state_as_feat:
            self.fusion_modules = GELUConv(
                in_channels=abs(hidden_state_index),
                out_channels=fusion_times,
                ksize=1,
                stride=1,
                shortcut=short_cut
            )
        # Create attention-based flattening module
        self.att_flat = AttFlat(hidden_size, flat_glimpses, dropout_rate)

    def frozen(self, module):
        """
        Freeze the parameters of a module to prevent gradient updates during training.
        
        This method is used to freeze specific parts of the Wav2Vec2 model during
        transfer learning, allowing only certain components to be trained while
        keeping others fixed.
        
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

    def forward(self, audio, mask):
        """
        Forward pass of the Wav2vec2 speech encoder.
        
        This method processes audio input through the Wav2Vec2 model and extracts
        features from the specified hidden states. It supports two modes of feature
        extraction: using a single hidden state or fusing multiple hidden states.
        
        Args:
            audio (torch.Tensor): Audio input tensor of shape (batch_size, sequence_length)
            mask (torch.Tensor): Attention mask for audio input of shape (batch_size, sequence_length)
            
        Returns:
            dict: A dictionary containing:
                - 'flat_feat' (torch.Tensor): Flattened features of shape (batch_size, hidden_size)
                - 'feat' (torch.Tensor): Sequence features of shape (batch_size, sequence_length, hidden_size)
        """
        # Process audio through the Wav2Vec2 model
        # Request attention outputs and all hidden states for feature extraction
        output = self.model(
            input_values=audio,
            attention_mask=mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract features from the model's hidden states
        feat = None
        if self.use_one_hidden_state_as_feat:
            # Use a single hidden state as features
            hidden_state = output.hidden_states[self.hidden_state_index]  # [batch, seq_len, hidden_size]
            feat = hidden_state  # (batch, seq_len, hidden_size)
        else:
            # Use multiple hidden states and fuse them
            # Stack hidden states along a new dimension
            feat = torch.stack(output.hidden_states[self.hidden_state_index:], 1)  # (batch, n_hidden, seq_len, hidden_size)
            # Apply fusion modules to combine information across hidden states
            feat = self.fusion_modules(feat)  # (batch, fusion_times, seq_len, hidden_size)
            # Flatten the fusion dimension to get a single feature sequence
            feat = torch.flatten(feat, 1, 2)  # (batch, seq_len, hidden_size)

        # Create mask for attention flattening if enabled
        mask_flip_bool = None
        if self.use_att_flat_mask:
            # Generate attention mask for the feature vectors
            first_attention_0 = self.model._get_feature_vector_attention_mask(output.hidden_states[-1].shape[1],
                                                                              mask)  # [batch, seq_len]
            # Convert to 4D tensor for compatibility with attention flattening
            mask_flip_bool = make_mask(first_attention_0.unsqueeze(2))  # (batch, 1, 1, seq_len)
            # Repeat mask for each fusion time if multiple fusions are used
            mask_flip_bool = torch.cat([mask_flip_bool for i in range(self.fusion_times)], 3)

        # Apply attention-based flattening to convert sequence features to fixed-size vectors
        flat_feat = self.att_flat(feat, mask_flip_bool)

        return {
            'flat_feat': flat_feat,  # Fixed-size feature vector for downstream tasks
            'feat': feat,
        }
