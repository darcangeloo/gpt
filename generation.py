"""
Text Generation Module

Implements the text generation loop for the GPT model.
"""

import torch
import torch.nn.functional as F


def generate(model, idx, max_new_tokens, context_size, temperature=0.8, top_k=40):
    """
    Generate text by repeatedly predicting the next token.

    Uses temperature-scaled sampling with top-k filtering to produce more
    natural-looking text than pure greedy decoding.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        logits = logits / max(temperature, 1e-6)

        if top_k is not None and top_k < logits.size(-1):
            top_k_values, top_k_indices = torch.topk(logits, k=top_k)
            probs = F.softmax(top_k_values, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            idx_next = top_k_indices.gather(-1, sampled)
        else:
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next.unsqueeze(-1)), dim=1)

    return idx
