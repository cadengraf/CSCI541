"""Part 1 baseline CNN — trains on pj2_lite_cifar_train.zip and exports ONNX.

Architecture and training settings follow baseline_cnn_spec.md exactly.
"""

import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from export_onnx import load_lite_cifar, export_onnx, check_onnx


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Layers 1-4: backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # Layers 5-7: Flatten head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = (x - 0.5) / 0.25          # normalise inside forward
        x = self.features(x)
        x = self.classifier(x)
        return x                       # logits, no softmax


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="pj2_lite_cifar_train.zip")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="baseline_cnn.onnx")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_lite_cifar(args.data)
    n = len(X)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    val_idx = perm[:500]
    train_idx = perm[500:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(args.seed)
    model = BaselineCNN().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # Augmentation: random horizontal flip only
            flip = torch.rand(xb.size(0), device=xb.device) < 0.5
            if flip.any():
                xb[flip] = torch.flip(xb[flip], dims=[3])

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {train_loss:.4f}  "
              f"val acc {val_acc:.4f}  best {best_acc:.4f}")

    model.load_state_dict(best_state)
    print(f"best validation accuracy: {best_acc:.4f}")

    export_onnx(model, args.out)
    check_onnx(args.out, part="part1", images=X_val, labels=y_val)


if __name__ == "__main__":
    main()
