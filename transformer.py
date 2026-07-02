"""
Transformer Block and GPT Model

Combines attention and feed-forward layers into transformer blocks,
and assembles them into the complete GPT model.
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention
from layers.layer_norm import LayerNorm
from layers.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer Block.
    
    Combines multi-head attention and feed-forward layers with residual
    connections and layer normalization (pre-norm architecture).
    """
    
    def __init__(self, cfg):
        """
        Initialize TransformerBlock.
        
        Args:
            cfg (dict): Configuration dictionary with model parameters
        """
        super().__init__()
        
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        """
        Apply transformer block transformation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, emb_dim)
        """
        # Attention with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        # Feed-forward with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x


class GPT(nn.Module):
    """
    GPT Model.
    
    Complete GPT architecture combining token embeddings, positional embeddings,
    transformer blocks, and output projection head.
    """
    
    def __init__(self, cfg):
        """
        Initialize GPT model.
        
        Args:
            cfg (dict): Configuration dictionary with model parameters
        """
        super().__init__()
        
        # Embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # Final normalization and output head
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        """
        Forward pass through GPT model.
        
        Args:
            in_idx (torch.Tensor): Input token indices of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = in_idx.shape
        
        # Token embeddings
        tok_embeds = self.tok_emb(in_idx)
        
        # Positional embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # Combine embeddings
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # Apply transformer blocks
        x = self.trf_blocks(x)
        
        # Final normalization and projection
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits
