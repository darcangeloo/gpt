"""
Layer Normalization Module

Implements layer normalization to stabilize training and improve convergence.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization implementation.
    
    Normalizes input tensors across the embedding dimension with learnable
    scale and shift parameters.
    """
    
    def __init__(self, emb_dim):
        """
        Initialize LayerNorm.
        
        Args:
            emb_dim (int): Embedding dimension size
        """
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        """
        Apply layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim)
            
        Returns:
            torch.Tensor: Normalized tensor of same shape as input
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
