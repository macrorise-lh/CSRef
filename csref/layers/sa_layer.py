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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    """
    Fully Connected layer with optional ReLU activation and dropout.
    
    This class implements a simple fully connected layer that can include
    ReLU activation and dropout regularization. It's a basic building block
    used in various parts of the model architecture.
    
    Attributes:
        dropout_r (float): Dropout rate
        use_relu (bool): Whether to use ReLU activation
        linear (nn.Linear): Linear transformation layer
        relu (nn.ReLU, optional): ReLU activation function
        dropout (nn.Dropout, optional): Dropout layer for regularization
    """
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        """
        Initialize the FC layer.
        
        Args:
            in_size (int): Input feature dimension
            out_size (int): Output feature dimension
            dropout_r (float, optional): Dropout rate. Defaults to 0.
            use_relu (bool, optional): Whether to use ReLU activation. Defaults to True.
        """
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        # Create the linear transformation layer
        self.linear = nn.Linear(in_size, out_size)

        # Create ReLU activation if specified
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        # Create dropout layer if dropout rate > 0
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        """
        Forward pass of the FC layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., in_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., out_size)
        """
        # Apply linear transformation
        x = self.linear(x)

        # Apply ReLU activation if specified
        if self.use_relu:
            x = self.relu(x)

        # Apply dropout if specified
        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class LayerNorm(nn.Module):
    """
    Layer Normalization module.
    
    This class implements layer normalization, which normalizes the inputs
    across the features. It's commonly used in transformer architectures
    to stabilize training and improve convergence.
    
    Attributes:
        eps (float): Small constant added to denominator for numerical stability
        weight (nn.Parameter): Learnable scaling parameter
        bias (nn.Parameter): Learnable shift parameter
    """
    def __init__(self, size, eps=1e-6):
        """
        Initialize the LayerNorm module.
        
        Args:
            size (int): Input feature dimension
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
        """
        super(LayerNorm, self).__init__()
        self.eps = eps

        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        """
        Forward pass of the LayerNorm module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., size)
            
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        # Compute mean and standard deviation along the last dimension
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # Apply layer normalization formula
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with one hidden layer.
    
    This class implements a simple MLP with one hidden layer, ReLU activation,
    and optional dropout. It's a common building block in neural networks.
    
    Attributes:
        fc (FC): Fully connected layer with ReLU activation and dropout
        linear (nn.Linear): Output linear transformation layer
    """
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        """
        Initialize the MLP module.
        
        Args:
            in_size (int): Input feature dimension
            mid_size (int): Hidden layer feature dimension
            out_size (int): Output feature dimension
            dropout_r (float, optional): Dropout rate. Defaults to 0.
            use_relu (bool, optional): Whether to use ReLU activation. Defaults to True.
        """
        super(MLP, self).__init__()

        # Create the first fully connected layer with activation and dropout
        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        # Create the output linear transformation layer
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        """
        Forward pass of the MLP module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., in_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., out_size)
        """
        # Apply the first fully connected layer and then the output linear layer
        return self.linear(self.fc(x))


class AttFlat(nn.Module):
    """
    Attention-based flattening module.
    
    This module uses attention mechanisms to flatten variable-length sequences
    into fixed-size vectors. It's particularly useful for converting sequence
    features (e.g., from audio or text encoders) into fixed-size representations
    that can be used for downstream tasks.
    
    The module computes attention weights for each position in the sequence,
    then uses these weights to compute a weighted sum of the sequence features.
    It supports multiple "glimpses", which allows it to extract multiple different
    summaries of the sequence.
    
    Attributes:
        hidden_size (int): Hidden feature dimension
        flat_glimpses (int): Number of attention glimpses
        dropout_rate (float): Dropout rate
        mlp (MLP): MLP for computing attention weights
        linear_merge (nn.Linear): Linear layer for merging glimpses
    """
    def __init__(self, hidden_size, flat_glimpses, dropout_rate):
        """
        Initialize the AttFlat module.
        
        Args:
            hidden_size (int): Hidden feature dimension
            flat_glimpses (int): Number of attention glimpses
            dropout_rate (float): Dropout rate
        """
        super(AttFlat, self).__init__()
        self.hidden_size = hidden_size
        self.flat_glimpses = flat_glimpses
        self.dropout_rate = dropout_rate

        # Create MLP for computing attention weights
        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=hidden_size//2,
            out_size=flat_glimpses,
            dropout_r=dropout_rate,
            use_relu=True
        )

        # Create linear layer for merging glimpses
        self.linear_merge = nn.Linear(
            hidden_size ,
            hidden_size
        )

    def forward(self, x, x_mask=None):
        """
        Forward pass of the AttFlat module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)
            x_mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, 1, seq_len)
                                             where 1 indicates valid positions and 0 indicates padding.
                                             Defaults to None.
            
        Returns:
            torch.Tensor: Flattened tensor of shape (batch_size, hidden_size * flat_glimpses)
        """
        b, l, c = x.size()  # batch_size, seq_len, hidden_size
        
        # Compute attention weights using MLP
        att = self.mlp(x).view(b, l, -1)  # (batch_size, seq_len, flat_glimpses)
        
        # Reshape input for glimpse-wise attention
        x = x.view(b, l, self.flat_glimpses, -1)  # (batch_size, seq_len, flat_glimpses, hidden_size/flat_glimpses)
        
        # Apply attention mask if provided
        if x_mask is not None:
            # Mask out attention weights for padded positions
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),  # Reshape mask to match attention weights
                -1e4  # Large negative value for masked positions
            )
        
        # Apply softmax to get attention probabilities
        att = F.softmax(att, dim=1)

        # Apply attention weights to get glimpse-wise summaries
        att_list = []
        for i in range(self.flat_glimpses):
            # Compute weighted sum for each glimpse
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x[:, :, i, :], dim=1)
            )

        # Concatenate all glimpses
        x_atted = torch.cat(att_list, dim=1)  # (batch_size, hidden_size * flat_glimpses)
        
        # Apply linear transformation to merge glimpses
        x_atted = self.linear_merge(x_atted)

        return x_atted

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    """
    Multi-Head Attention module.
    
    This class implements the multi-head attention mechanism introduced in the
    Transformer architecture. It allows the model to jointly attend to information
    from different representation subspaces at different positions.
    
    Multi-head attention splits the hidden dimension into multiple heads,
    computes scaled dot-product attention for each head independently, and then
    concatenates the results. This enables the model to capture different types
    of dependencies in the data.
    
    Attributes:
        num_heads (int): Number of attention heads
        hidden_size (int): Hidden feature dimension
        linear_v (nn.Linear): Linear transformation for values
        linear_k (nn.Linear): Linear transformation for keys
        linear_q (nn.Linear): Linear transformation for queries
        linear_merge (nn.Linear): Linear transformation for merging heads
        dropout (nn.Dropout): Dropout layer for attention weights
    """
    def __init__(self, hidden_size, num_heads, dropout_rate):
        """
        Initialize the MHAtt module.
        
        Args:
            hidden_size (int): Hidden feature dimension
            num_heads (int): Number of attention heads
            dropout_rate (float): Dropout rate for attention weights
        """
        super(MHAtt, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Create linear transformations for values, keys, and queries
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        # Create linear transformation for merging heads
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        # Create dropout layer for attention weights
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, v, k, q, mask):
        """
        Forward pass of the MHAtt module.
        
        Args:
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, hidden_size)
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, hidden_size)
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, hidden_size)
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, num_heads, seq_len, seq_len)
                                          where 0 indicates positions to be masked.
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        n_batches = q.size(0)
        
        # Apply linear transformations and reshape for multi-head attention
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.hidden_size / self.num_heads)
        ).transpose(1, 2)  # (batch_size, num_heads, seq_len, hidden_size/num_heads)
        
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.hidden_size / self.num_heads)
        ).transpose(1, 2)  # (batch_size, num_heads, seq_len, hidden_size/num_heads)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.hidden_size / self.num_heads)
        ).transpose(1, 2)  # (batch_size, num_heads, seq_len, hidden_size/num_heads)

        # Compute attention
        atted = self.att(v, k, q, mask)
        # Reshape and merge heads
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )  # (batch_size, seq_len, hidden_size)

        # Apply linear transformation to merge heads
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        """
        Compute scaled dot-product attention.
        
        This method implements the scaled dot-product attention mechanism,
        which computes attention weights based on the compatibility between
        queries and keys, then uses these weights to compute a weighted sum
        of values.
        
        Args:
            value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, hidden_size/num_heads)
            key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, hidden_size/num_heads)
            query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, hidden_size/num_heads)
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, num_heads, seq_len, seq_len)
                                          where 0 indicates positions to be masked.
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len, hidden_size/num_heads)
        """
        d_k = query.size(-1)  # Dimension of keys/queries

        # Compute attention scores
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # Scale by sqrt(d_k) to prevent vanishing gradients

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)  # Large negative value for masked positions

        # Compute attention probabilities
        att_map = F.softmax(scores, dim=-1)
        # Apply dropout to attention weights
        att_map = self.dropout(att_map)

        # Compute weighted sum of values
        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    """
    Feed-Forward Network module.
    
    This class implements the feed-forward network used in Transformer architectures.
    It consists of a two-layer MLP with a ReLU activation in between, which is applied
    to each position separately and identically.
    
    The feed-forward network helps the model to learn more complex representations
    by introducing non-linearity and increasing the model capacity.
    
    Attributes:
        mlp (MLP): Multi-layer perceptron with one hidden layer
    """
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        """
        Initialize the FFN module.
        
        Args:
            hidden_size (int): Input and output feature dimension
            ffn_size (int): Hidden layer feature dimension
            dropout_rate (float): Dropout rate
        """
        super(FFN, self).__init__()

        # Create MLP with one hidden layer
        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ffn_size,
            out_size=hidden_size,
            dropout_r=dropout_rate,
            use_relu=True
        )

    def forward(self, x):
        """
        Forward pass of the FFN module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        return self.mlp(x)



