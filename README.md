# GPT from Scratch (Decoder-Only Transformer)

This repository contains a minimal implementation of a GPT-style decoder-only Transformer built from scratch in PyTorch for educational purposes.

## Overview

The goal of this project is to understand how large language models work internally by implementing all core components of a Transformer architecture from scratch.

The implementation includes:

- Token embeddings
- Positional embeddings
- Causal self-attention (multi-head attention)
- Feed-forward networks (MLP)
- Residual connections
- Layer normalization
- Autoregressive text generation

The model is currently **untrained** and produces random token sequences. It is used as a structural and experimental baseline.

## Current Status

- Architecture: implemented and functional
- Training: not implemented yet
- Output: random token generation
- Purpose: educational and research exploration

## Features

- Decoder-only Transformer (GPT-style)
- Multi-head causal self-attention
- Masked attention for autoregressive generation
- Greedy decoding strategy
- Fully modular PyTorch implementation

## How It Works

1. Input tokens are converted into embeddings
2. Positional information is added
3. Transformer blocks process the sequence using self-attention
4. Final linear layer projects to vocabulary logits
5. Tokens are generated autoregressively

## Future Work

- Training loop implementation (loss, optimizer, dataset)
- Character-level or BPE tokenization experiments
- Scaling depth, heads, and context size
- Mechanistic interpretability experiments (e.g., induction heads)
- Attention pattern analysis and visualization

## Example Output

Since the model is untrained, outputs are random:

Input: "Hello i am"
Output: "Hello i am Graphics Null spa499 pilgrimageCommunity"


## Motivation

This project is part of a learning journey to deeply understand transformer-based language models at a low level and prepare for research in mechanistic interpretability.

## Tech Stack

- Python
- PyTorch
- NumPy

## Notes

This is not a production-ready model. It is a purely educational implementation for studying Transformer architectures.