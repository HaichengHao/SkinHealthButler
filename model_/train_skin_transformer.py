"""Train skin disease classifier with Transformer backbone and export best model.

Example:
python3 model_/train_skin_transformer.py \
  --train-dir model_/SkinDisease/train \
  --test-dir model_/SkinDisease/test \
  --output-dir model_/exports/swinv2
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import timm

SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_classes(train_dir: Path) -> list[str]:
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])


class SkinDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Invalid image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def collect_samples(root: Path, cls2idx: dict[str, int]) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for cls, idx in cls2idx.items():
        cls_dir = root / cls
        if not cls_dir.exists():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                items.append((str(f), idx))
    return items


class SwinHead(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        image_size: int,
        dropout: float,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        feat = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Dropout(dropout),
            nn.Linear(feat, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_pred = []
    all_true = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        losses.append(loss.item())
        all_pred.extend(pred.detach().cpu().tolist())
        all_true.extend(y.detach().cpu().tolist())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return float(np.mean(losses)) if losses else 0.0, float(acc), float(f1)


def export_onnx(model: nn.Module, out_path: Path, image_size: int, device: torch.device) -> None:
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="model_/exports/swinv2")
    parser.add_argument("--model-name", type=str, default="swinv2_base_window16_256")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--pretrained", action="store_true", default=False)
    args = parser.parse_args()

    seed_everything(SEED)

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = list_classes(train_dir)
    cls2idx = {c: i for i, c in enumerate(class_names)}
    print(f"[INFO] classes={len(class_names)}", flush=True)

    train_samples = collect_samples(train_dir, cls2idx)
    random.shuffle(train_samples)
    split = int(len(train_samples) * (1 - args.val_ratio))
    tr_samples = train_samples[:split]
    va_samples = train_samples[split:]
    te_samples = collect_samples(test_dir, cls2idx)
    print(
        f"[INFO] train={len(tr_samples)} val={len(va_samples)} test={len(te_samples)}",
        flush=True,
    )

    train_ds = SkinDataset(tr_samples, args.image_size)
    val_ds = SkinDataset(va_samples, args.image_size)
    test_ds = SkinDataset(te_samples, args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} torch_cuda={torch.version.cuda}", flush=True)
    print(f"[INFO] building model={args.model_name} pretrained={args.pretrained}", flush=True)
    model = SwinHead(
        args.model_name,
        len(class_names),
        args.image_size,
        args.dropout,
        pretrained=args.pretrained,
    ).to(device)
    print("[INFO] model ready", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = -1.0
    best_path = out_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        , flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "best_val_acc": best_acc,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Best model test | loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}", flush=True)

    onnx_path = out_dir / "best_model.onnx"
    export_onnx(model, onnx_path, args.image_size, device)

    meta = {
        "model_name": args.model_name,
        "class_names": class_names,
        "num_classes": len(class_names),
        "image_size": args.image_size,
        "dropout": args.dropout,
        "best_val_acc": best_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "onnx_path": str(onnx_path),
        "checkpoint_path": str(best_path),
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Export done: {out_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
