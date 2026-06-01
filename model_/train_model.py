# @Time    : 2026/5/30 12:01
# @Author  : hero
# @File    : train_model.py
# =============================================================================
#  SKIN DISEASE CLASSIFICATION — Swin Transformer V2 + Checkpoint Resume
#  Author : ChatGPT Refactor Version
# =============================================================================

import os
import cv2
import math
import time
import random
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import timm
import albumentations as A

from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report
)

from sklearn.utils.class_weight import compute_class_weight

# =============================================================================
#  CONFIG
# =============================================================================

SEED = 42

TRAIN_DIR = "./SkinDisease/train"
TEST_DIR = "./SkinDisease/test"

OUTPUT_DIR = "/kaggle/working"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =============================================================================
#  MODEL CONFIG
# =============================================================================

MODEL_NAME = "swinv2_base_window16_256"
IMAGE_SIZE = 256

BATCH_SIZE = 24
NUM_WORKERS = 4

LR_HEAD = 3e-4
LR_PARTIAL = 5e-5
LR_FULL = 1e-5

WEIGHT_DECAY = 1e-4

EPOCHS_P1 = 8
EPOCHS_P2 = 12
EPOCHS_P3 = 20

LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0

DROPOUT = 0.4

SAVE_EVERY_EPOCH = True

# =============================================================================
#  SEED
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
#  DEVICE
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpus = torch.cuda.device_count()

print(f"\nDevice : {device}")
print(f"GPUs   : {n_gpus}")

# =============================================================================
#  DATASET ANALYSIS
# =============================================================================

def scan_dir(base_dir):
    result = {}

    for cls in sorted(os.listdir(base_dir)):
        cls_dir = os.path.join(base_dir, cls)

        if not os.path.isdir(cls_dir):
            continue

        count = 0

        for f in os.listdir(cls_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                count += 1

        result[cls] = count

    return result


train_counts = scan_dir(TRAIN_DIR)
test_counts = scan_dir(TEST_DIR)

CLASS_NAMES = sorted(train_counts.keys())

NUM_CLASSES = len(CLASS_NAMES)

cls2idx = {
    cls: idx
    for idx, cls in enumerate(CLASS_NAMES)
}

print(f"\nClasses : {NUM_CLASSES}")

# =============================================================================
#  CLASS WEIGHTS
# =============================================================================

flat_labels = []

for idx, cls in enumerate(CLASS_NAMES):
    flat_labels.extend([idx] * train_counts[cls])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=np.array(flat_labels)
)

CLASS_WEIGHTS = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

# =============================================================================
#  LOSS
# =============================================================================

class FocalLoss(nn.Module):

    def __init__(
            self,
            weight=None,
            gamma=2.0,
            label_smoothing=0.1
    ):
        super().__init__()

        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):

        log_prob = F.log_softmax(logits, dim=1)

        prob = torch.exp(log_prob)

        pt = prob.gather(
            1,
            targets.view(-1, 1)
        ).squeeze(1)

        ce_loss = F.nll_loss(
            log_prob,
            targets,
            reduction="none",
            weight=self.weight
        )

        smooth_loss = -log_prob.mean(dim=1)

        ce_loss = (
                (1 - self.label_smoothing) * ce_loss
                + self.label_smoothing * smooth_loss
        )

        focal_weight = (1 - pt).pow(self.gamma)

        loss = focal_weight * ce_loss

        return loss.mean()


criterion = FocalLoss(
    weight=CLASS_WEIGHTS,
    gamma=FOCAL_GAMMA,
    label_smoothing=LABEL_SMOOTHING
)

# =============================================================================
#  AUGMENTATION
# =============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transform = A.Compose([

    A.RandomResizedCrop(
        size=(IMAGE_SIZE, IMAGE_SIZE),
        scale=(0.75, 1.0),
        ratio=(0.9, 1.1)
    ),

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.3),

    A.Rotate(limit=30, p=0.5),

    A.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05,
        p=0.7
    ),

    A.CLAHE(p=0.4),

    A.GaussNoise(p=0.2),

    A.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),

    ToTensorV2(),
])

val_transform = A.Compose([

    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),

    ToTensorV2(),
])

# =============================================================================
#  DATASET
# =============================================================================

class SkinDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.samples = []
        self.transform = transform

        for cls in CLASS_NAMES:

            cls_dir = os.path.join(root_dir, cls)

            if not os.path.isdir(cls_dir):
                continue

            for fname in os.listdir(cls_dir):

                if fname.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp")
                ):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        cls2idx[cls]
                    ))

        random.shuffle(self.samples)

        print(f"Loaded {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = cv2.imread(path)

        if img is None:
            img = np.zeros(
                (IMAGE_SIZE, IMAGE_SIZE, 3),
                dtype=np.uint8
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label


train_dataset = SkinDataset(
    TRAIN_DIR,
    transform=train_transform
)

test_dataset = SkinDataset(
    TEST_DIR,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE * max(1, n_gpus),
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * max(1, n_gpus),
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =============================================================================
#  MODEL
# =============================================================================

class SwinV2Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=0,
            img_size=IMAGE_SIZE
        )

        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(

            nn.LayerNorm(feat_dim),

            nn.Dropout(DROPOUT),

            nn.Linear(feat_dim, 512),

            nn.GELU(),

            nn.LayerNorm(512),

            nn.Dropout(DROPOUT * 0.5),

            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):

        feat = self.backbone(x)

        out = self.head(feat)

        return out

    # -------------------------------------------------------------
    # freeze helpers
    # -------------------------------------------------------------

    def freeze_backbone(self):

        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_stages(self, n=2):

        layers = list(self.backbone.layers)

        for layer in layers[-n:]:

            for p in layer.parameters():
                p.requires_grad = True

        if hasattr(self.backbone, "norm"):

            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    def unfreeze_all(self):

        for p in self.backbone.parameters():
            p.requires_grad = True


model = SwinV2Classifier()

if n_gpus > 1:
    model = nn.DataParallel(model)

model = model.to(device)

# =============================================================================
#  OPTIMIZER
# =============================================================================

def build_optimizer(model, lr):

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )

    return optimizer


# =============================================================================
#  TRAIN
# =============================================================================

def train_epoch(
        model,
        loader,
        optimizer,
        scaler
):

    model.train()

    total_loss = 0
    total_correct = 0
    total = 0

    for imgs, labels in loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():

            logits = model(imgs)

            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)

        scaler.update()

        preds = logits.argmax(dim=1)

        total_correct += (preds == labels).sum().item()

        total += imgs.size(0)

        total_loss += loss.item() * imgs.size(0)

    return (
        total_loss / total,
        total_correct / total
    )


@torch.no_grad()
def eval_epoch(model, loader):

    model.eval()

    total_loss = 0
    total_correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for imgs, labels in loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        with autocast():

            logits = model(imgs)

            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        total_correct += (preds == labels).sum().item()

        total += imgs.size(0)

        total_loss += loss.item() * imgs.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = total_correct / total

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    bal_acc = balanced_accuracy_score(
        all_labels,
        all_preds
    )

    return (
        total_loss / total,
        acc,
        macro_f1,
        bal_acc
    )

# =============================================================================
#  CHECKPOINT SAVE
# =============================================================================

def save_checkpoint(
        epoch,
        model,
        optimizer,
        scaler,
        best_acc,
        phase_name,
        is_best=False
):

    ckpt = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "best_acc": best_acc,

        "model_name": MODEL_NAME,

        "image_size": IMAGE_SIZE,

        "class_names": CLASS_NAMES,

        "num_classes": NUM_CLASSES,
    }

    latest_path = os.path.join(
        CHECKPOINT_DIR,
        f"{phase_name}_latest.pth"
    )

    torch.save(ckpt, latest_path)

    if SAVE_EVERY_EPOCH:

        epoch_path = os.path.join(
            CHECKPOINT_DIR,
            f"{phase_name}_epoch_{epoch}.pth"
        )

        torch.save(ckpt, epoch_path)

    if is_best:

        best_path = os.path.join(
            CHECKPOINT_DIR,
            f"{phase_name}_best.pth"
        )

        torch.save(ckpt, best_path)

        print(f"\n✓ BEST MODEL SAVED : {best_path}")

# =============================================================================
#  CHECKPOINT LOAD
# =============================================================================

def load_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        scaler=None
):

    print(f"\nLoading checkpoint : {checkpoint_path}")

    ckpt = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            ckpt["optimizer_state_dict"]
        )

    if scaler is not None:
        scaler.load_state_dict(
            ckpt["scaler_state_dict"]
        )

    start_epoch = ckpt["epoch"] + 1

    best_acc = ckpt["best_acc"]

    print(f"Resume from epoch : {start_epoch}")

    return start_epoch, best_acc

# =============================================================================
#  TRAIN PHASE
# =============================================================================

def run_phase(
        model,
        phase_name,
        epochs,
        lr,
        freeze_fn=None,
        unfreeze_fn=None,
        resume_path=None
):

    print("\n" + "=" * 70)
    print(f"PHASE : {phase_name}")
    print("=" * 70)

    raw_model = model.module if isinstance(
        model,
        nn.DataParallel
    ) else model

    if freeze_fn:
        freeze_fn(raw_model)

    if unfreeze_fn:
        unfreeze_fn(raw_model)

    optimizer = build_optimizer(model, lr)

    scaler = GradScaler()

    start_epoch = 1
    best_acc = 0

    if resume_path is not None:

        start_epoch, best_acc = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scaler
        )

    patience = 8
    wait = 0

    history = []

    for epoch in range(start_epoch, epochs + 1):

        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler
        )

        val_loss, val_acc, val_f1, val_bal = eval_epoch(
            model,
            test_loader
        )

        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"TrainLoss={train_loss:.4f} "
            f"TrainAcc={train_acc*100:.2f}% "
            f"ValLoss={val_loss:.4f} "
            f"ValAcc={val_acc*100:.2f}% "
            f"F1={val_f1:.4f} "
            f"BalAcc={val_bal:.4f} "
            f"Time={elapsed:.0f}s"
        )

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "train_acc": train_acc,

            "val_loss": val_loss,

            "val_acc": val_acc,

            "val_f1": val_f1,

            "val_bal_acc": val_bal
        })

        is_best = val_acc > best_acc

        if is_best:

            best_acc = val_acc

            wait = 0

        else:

            wait += 1

        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            best_acc=best_acc,
            phase_name=phase_name,
            is_best=is_best
        )

        if wait >= patience:

            print("\nEarly stopping triggered")

            break

    best_path = os.path.join(
        CHECKPOINT_DIR,
        f"{phase_name}_best.pth"
    )

    ckpt = torch.load(best_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    print(f"\nBest Accuracy : {best_acc*100:.2f}%")

    return history

# =============================================================================
#  PHASE 1
# =============================================================================

history1 = run_phase(
    model=model,
    phase_name="phase1",
    epochs=EPOCHS_P1,
    lr=LR_HEAD,
    freeze_fn=lambda m: m.freeze_backbone()
)

# =============================================================================
#  PHASE 2
# =============================================================================

history2 = run_phase(
    model=model,
    phase_name="phase2",
    epochs=EPOCHS_P2,
    lr=LR_PARTIAL,
    unfreeze_fn=lambda m: m.unfreeze_last_stages(2)
)

# =============================================================================
#  PHASE 3
# =============================================================================

history3 = run_phase(
    model=model,
    phase_name="phase3",
    epochs=EPOCHS_P3,
    lr=LR_FULL,
    unfreeze_fn=lambda m: m.unfreeze_all()
)

# =============================================================================
#  FINAL EVALUATION
# =============================================================================

print("\nRunning final evaluation...")

val_loss, val_acc, val_f1, val_bal = eval_epoch(
    model,
    test_loader
)

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(f"Accuracy           : {val_acc*100:.2f}%")
print(f"Macro F1           : {val_f1:.4f}")
print(f"Balanced Accuracy  : {val_bal:.4f}")

# =============================================================================
#  EXPORT FINAL MODEL
# =============================================================================

final_export_path = os.path.join(
    OUTPUT_DIR,
    "skin_disease_swinv2_export.pth"
)

torch.save({

    "model_state_dict": model.state_dict(),

    "class_names": CLASS_NAMES,

    "num_classes": NUM_CLASSES,

    "model_name": MODEL_NAME,

    "image_size": IMAGE_SIZE,

    "accuracy": val_acc,

    "macro_f1": val_f1,

    "balanced_acc": val_bal

}, final_export_path)

print(f"\n✓ FINAL MODEL EXPORTED")
print(final_export_path)

# =============================================================================
#  OPTIONAL : LOAD BEST MODEL LATER
# =============================================================================
#
# ckpt = torch.load(
#     "/kaggle/working/checkpoints/phase3_best.pth"
# )
#
# model.load_state_dict(
#     ckpt["model_state_dict"]
# )
#
# =============================================================================