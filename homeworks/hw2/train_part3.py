"""Part 3 — own design, CNN review ideas applied to a part-2-style model.

Changes vs Part 2 (and why):
  * BatchNorm after every conv   -> better optimization, deeper net trainable
  * Residual shortcuts (ResNet)  -> lets extra depth actually train
  * Repeated 3x3 convs (VGG)     -> bigger receptive field, fewer params than 5x5
  * GAP head kept (NiN)          -> translation invariance (robustness requirement)
  * Dropout only in the head     -> regularization without clashing with BN
  * Augmentation: random horizontal flip ONLY (Part 3 rules: no other
    augmentation, no extra data, no pretrained weights)

Same data, split, seed handling, loop, and output format as train_baseline.py /
train_gap.py. Exports part3_cnn.onnx and runs check_onnx(part="part3").
"""

import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from export_onnx import load_lite_cifar, export_onnx, check_onnx


class ConvBNReLU(nn.Module):
    """3x3 conv -> BN -> ReLU"""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResStage(nn.Module):
    """Residual block: shortcut + 2 repeated Conv3x3(vgg), 1x1 downsample projection."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        )
        if stride != 1 or cin != cout:                     # channel/size change
            self.branch = nn.Sequential(                   # cheap 1x1 projection (NiN idea)
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        else:
            self.branch = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.branch(x))


class Part3CNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = ConvBNReLU(3, 32)                      # 64x64

        self.stage1 = nn.Sequential(ResStage(32, 64, stride=2),   # -> 32x32
                                    ResStage(64, 64))
        self.stage2 = nn.Sequential(ResStage(64, 128, stride=2),  # -> 16x16
                                    ResStage(128, 128))
        self.stage3 = nn.Sequential(ResStage(128, 256, stride=2), # -> 8x8
                                    ResStage(256, 256))

        # NiN ending: GAP -> Dropout -> Linear (translation-invariant head)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = (x - 0.5) / 0.25          # normalise inside forward
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
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
    ap.add_argument("--out", default="part3_cnn.onnx")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_lite_cifar(args.data)                       # same data + split as Parts 1-2
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
    model = Part3CNN().to(device)
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
            # Augmentation: random horizontal flip (the ONLY augmentation allowed)
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
    check_onnx(args.out, part="part3", images=X_val, labels=y_val)


if __name__ == "__main__":
    main()
