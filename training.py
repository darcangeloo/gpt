from pathlib import Path
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from config import GPT_CONFIG as CFG
from transformer import GPT
from generation import generate

CKPT_PATH = Path("gpt_trained.pth")


class DS(Dataset):
    def __init__(self, ids, L, S):
        self.ids = ids
        self.L, self.S = L, S

    def __len__(self):
        return (len(self.ids) - self.L) // self.S

    def __getitem__(self, i):
        x = self.ids[i * self.S : i * self.S + self.L + 1]
        return torch.tensor(x[:-1]), torch.tensor(x[1:])


def build_loader(ids, bs=8, L=128, S=64, shuffle=True):
    ds = DS(ids, L, S)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def loss_fn(x, y, model, dev):
    x, y = x.to(dev), y.to(dev)
    out = model(x)
    return torch.nn.functional.cross_entropy(out.flatten(0, 1), y.flatten())


@torch.no_grad()
def eval_loss(model, loader, dev, n=5):
    model.eval()
    loss = 0.0
    for i, (x, y) in enumerate(loader):
        if i >= n:
            break
        loss += loss_fn(x, y, model, dev).item()
    model.train()
    return loss / n


def train(model, train_loader, val_loader, opt, dev, epochs=3):
    step = 0

    print("START TRAINING")

    for e in range(epochs):
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(x, y, model, dev)
            loss.backward()
            opt.step()

            step += 1

            if step % 100 == 0:
                val = eval_loss(model, val_loader, dev)
                print(f"step {step} | loss {loss.item():.4f} | val {val:.4f}")

    torch.save(model.state_dict(), CKPT_PATH)
    print("saved:", CKPT_PATH)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("loading dataset...")
    raw = load_dataset("roneneldan/TinyStories", split="train")

    text = "\n".join(raw["text"])[:1_000_000]  # SMALL FOR SPEED

    print("tokenizing...")
    tok = tiktoken.get_encoding("gpt2")
    ids = tok.encode(text)

    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]

    train_loader = build_loader(train_ids)
    val_loader = build_loader(val_ids, shuffle=False)

    model = GPT(CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    train(model, train_loader, val_loader, opt, device)


if __name__ == "__main__":
    main()