from pathlib import Path
import torch
import tiktoken

from transformer import GPT
from generation import generate
from training import CHECKPOINT_PATH
from config import GPT_CONFIG as CFG

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    torch.manual_seed(42)
    tokenizer = tiktoken.get_encoding("gpt2")

    txt1, txt2 = "Every effort moves you", "Everyday holds a"
    batch = torch.stack([
        torch.tensor(tokenizer.encode(txt1)),
        torch.tensor(tokenizer.encode(txt2))
    ]).to(device)

    print("Input batch:\n", batch)

    model = GPT(CFG).to(device)

    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Loaded trained weights from: {CHECKPOINT_PATH}")
    else:
        print("No trained weights found. Using random init.")

    model.eval()

    logits = model(batch)
    print("Model output shape:", logits.shape)

    prompt = "ROMEO: "
    encoded = tokenizer.encode(prompt)
    print(f"Input text: '{prompt}'")
    print("Encoded tokens:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)

    output_indices = generate(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=40,
        context_size=CFG["context_length"]
    )

    decoded_text = tokenizer.decode(output_indices.squeeze(0).tolist())
    print(f"Generated text: '{decoded_text}'")


if __name__ == "__main__":
    main()