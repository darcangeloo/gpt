"""
GELU Activation Function

Gaussian Error Linear Unit activation function commonly used in transformer models.
"""

import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    GELU Activation Function.
    
    Implements the GELU activation function using the tanh approximation,
    which is commonly used in transformer models like GPT.
    """
    
    def __init__(self):
        """Initialize GELU."""
        super().__init__()
    
    def forward(self, x):
        """
        Apply GELU activation.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with GELU activation applied
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.44715 * torch.pow(x, 3))
        ))
