"""
Feed-Forward Network Module

Implements the feed-forward network used in transformer layers.
"""

import torch.nn as nn
from .gelu import GELU


class FeedForward(nn.Module):
    """
    Feed-Forward Network.
    
    A two-layer feed-forward network with GELU activation used in transformer blocks.
    Expands the embedding dimension by a factor of 4 and then projects back.
    """
    
    def __init__(self, cfg):
        """
        Initialize FeedForward network.
        
        Args:
            cfg (dict): Configuration dictionary containing 'emb_dim' key
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )
    
    def forward(self, x):
        """
        Apply feed-forward transformation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, emb_dim)
        """
        return self.layers(x)
