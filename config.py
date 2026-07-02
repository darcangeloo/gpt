"""
GPT Model Configuration

Contains all hyperparameters for the GPT model.
"""

GPT_CONFIG = {
    'vocab_size': 50257,
    'context_length': 256,
    'emb_dim': 384,
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': False
}
