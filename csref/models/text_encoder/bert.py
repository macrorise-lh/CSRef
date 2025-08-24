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

import torch.nn as nn
from transformers import BertModel


class Bert(nn.Module):
    """
    BERT text encoder for extracting features from text inputs.
    
    This class wraps the Hugging Face BERT model and uses it to extract text features
    for the CSRef model. It focuses on extracting representations from the [CLS] token,
    which typically contains aggregated information about the entire text sequence.
    
    Key features:
    1. Pre-trained BERT model as the base text encoder
    2. Flexible selection of hidden states for feature extraction
    3. Support for freezing the model parameters for transfer learning
    4. Extraction of [CLS] token representations as text features
    
    Architecture:
    - Base BERT model for text encoding
    - Feature extraction from the [CLS] token of the specified hidden state
    """
    def __init__(
            self,
            pretrained_path="data/weights/bert-base-uncased",
            freeze_model=True,
            hidden_state_index=-1,
    ):
        """
        Initialize the BERT text encoder.
        
        Args:
            pretrained_path (str): Path to pre-trained BERT model.
                                  Defaults to "data/weights/bert-base-uncased".
            freeze_model (bool): Whether to freeze the model parameters. Defaults to True.
            hidden_state_index (int): Index of hidden state to use for feature extraction.
                                     Defaults to -1 (last layer).
        """
        super(Bert, self).__init__()

        # Store the index of the hidden state to use for feature extraction
        self.hidden_state_index = hidden_state_index

        # Load pre-trained BERT model
        self.model = BertModel.from_pretrained(pretrained_path)

        # Freeze model parameters if specified
        # This is useful for transfer learning scenarios where we want to keep
        # the BERT model fixed and only train other components of the system
        if freeze_model:
            self.frozen(self.model)

    def frozen(self, module):
        """
        Freeze the parameters of a module to prevent gradient updates during training.
        
        This method is used to freeze the BERT model during transfer learning,
        allowing only other components of the system to be trained while keeping
        the BERT representations fixed.
        
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

    def forward(self, text_ids, mask):
        """
        Forward pass of the BERT text encoder.
        
        This method processes text input through the BERT model and extracts
        features from the [CLS] token of the specified hidden state.
        
        Args:
            text_ids (torch.Tensor): Text input token IDs of shape (batch_size, sequence_length)
            mask (torch.Tensor): Attention mask for text input of shape (batch_size, sequence_length)
            
        Returns:
            torch.Tensor: Text features extracted from the [CLS] token,
                         of shape (batch_size, hidden_size)
        """
        # Process text through the BERT model
        # Request attention outputs and all hidden states for feature extraction
        output = self.model(
            input_ids=text_ids,
            attention_mask=mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract features from the specified hidden state
        hidden_state = output.hidden_states[self.hidden_state_index]
        # Use the [CLS] token representation as the text feature
        # The [CLS] token is at position 0 and typically contains aggregated
        # information about the entire text sequence
        feat = hidden_state[:, 0, :]  # corresponding to [CLS] token

        return feat
