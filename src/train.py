import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import NpySequenceDataset
from model import GRUClassifier

def load_classes(classes_txt="classes.txt"):
    with open(classes_txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    cfg = json.load(open("config.json", "r", encoding="utf-8"))
    seq_len = cfg["sequence_length"]
    classes = load_classes("classes.txt")

    samples_root = os.path.join("data", "samples")
    ds = NpySequenceDataset(samples_root, classes)

    # x shape is (T,F), so get F from first sample:
    x0, _ = ds[0]
    T, F = x0.shape

    n_val = max(1, int(0.15 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = cfg["model"]
    model = GRUClassifier(
        input_size=F,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_classes=len(classes)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"])

    train_loader = DataLoader(train_ds, batch_size=model_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=model_cfg["batch_size"], shuffle=False)

    epochs = model_cfg["epochs"]
    best_acc = -1.0
    os.makedirs("models", exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            xb = torch.tensor(xb, dtype=torch.float32).to(device)  # (B,T,F)
            yb = torch.tensor(yb, dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / max(1, total)

        # validation
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for xb, yb in val_loader:
                xb = torch.tensor(xb, dtype=torch.float32).to(device)
                yb = torch.tensor(yb, dtype=torch.long).to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
            val_acc = correct / max(1, total)

        print(f"[Epoch {ep}] train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join("models", "gru_airwrite_best.pt")
            torch.save(
                {"state_dict": model.state_dict(), "classes": classes, "F": F, "seq_len": seq_len},
                ckpt_path
            )
            print("Saved:", ckpt_path)

if __name__ == "__main__":
    main()