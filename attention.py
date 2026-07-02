"""
Multi-Head Attention Module

Implements the multi-head attention mechanism used in transformer models.
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer.
    
    Implements scaled dot-product attention with multiple heads to allow
    the model to focus on different parts of the sequence simultaneously.
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialize MultiHeadAttention.
        
        Args:
            d_in (int): Input dimension
            d_out (int): Output dimension (must be divisible by num_heads)
            context_length (int): Maximum sequence length
            dropout (float): Dropout rate
            num_heads (int): Number of attention heads
            qkv_bias (bool): Whether to use bias in query, key, value projections
        """
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask to prevent attention to future tokens
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        """
        Apply multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out)
        """
        b, num_tokens, d_in = x.shape
        
        # Project to Q, K, V
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Reshape for multi-head attention
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)
        
        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(
            attn_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)), 
            dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context_vec = (attn_weights @ values).transpose(1, 2)
        
        # Reshape back to (batch_size, seq_len, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # Final projection
        context_vec = self.out_proj(context_vec)
        return context_vec
