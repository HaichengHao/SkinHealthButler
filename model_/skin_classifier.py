"""Skin disease image classifier inference helper.

Supports two backends:
1) ONNX Runtime (preferred for production)
2) PyTorch checkpoint (fallback)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PredictionResult:
    top1_label: str
    top1_confidence: float
    topk: list[dict[str, Any]]


class SkinDiseaseClassifier:
    def __init__(
        self,
        class_names: list[str],
        image_size: int = 256,
        onnx_path: str | None = None,
        torch_ckpt_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.class_names = class_names
        self.image_size = image_size
        self.onnx_path = onnx_path
        self.torch_ckpt_path = torch_ckpt_path
        self.device = device

        self.backend: str | None = None
        self.session = None
        self.torch_model = None

        if onnx_path and Path(onnx_path).exists():
            self._load_onnx(onnx_path)
        elif torch_ckpt_path and Path(torch_ckpt_path).exists():
            self._load_torch(torch_ckpt_path, device)
        else:
            raise FileNotFoundError(
                "No valid model found. Please provide an existing onnx_path or torch_ckpt_path."
            )

    @classmethod
    def from_export_dir(
        cls,
        export_dir: str,
        prefer_onnx: bool = True,
        device: str = "cpu",
    ) -> "SkinDiseaseClassifier":
        export_path = Path(export_dir)
        meta_path = export_path / "model_meta.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"model metadata not found: {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        class_names = meta["class_names"]
        image_size = int(meta.get("image_size", 256))

        onnx_path = str(export_path / "best_model.onnx") if prefer_onnx else None
        ckpt_path = str(export_path / "best_model.pth")

        return cls(
            class_names=class_names,
            image_size=image_size,
            onnx_path=onnx_path,
            torch_ckpt_path=ckpt_path,
            device=device,
        )

    def _load_onnx(self, onnx_path: str) -> None:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.backend = "onnx"

    def _load_torch(self, ckpt_path: str, device: str) -> None:
        import torch
        import torch.nn as nn
        import timm

        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = ckpt["model_name"]
        num_classes = int(ckpt["num_classes"])
        dropout = float(ckpt.get("dropout", 0.4))

        backbone = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=self.image_size)
        feat_dim = backbone.num_features

        model = nn.Sequential(
            backbone,
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        model.to(device)

        self.torch_model = model
        self.backend = "torch"

    def _preprocess(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"failed to read image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image_path: str, topk: int = 3) -> PredictionResult:
        x = self._preprocess(image_path)

        if self.backend == "onnx":
            assert self.session is not None
            input_name = self.session.get_inputs()[0].name
            logits = self.session.run(None, {input_name: x})[0]
        elif self.backend == "torch":
            import torch

            assert self.torch_model is not None
            tensor = torch.from_numpy(x).to(self.device)
            with torch.no_grad():
                logits = self.torch_model(tensor).detach().cpu().numpy()
        else:
            raise RuntimeError("model backend not initialized")

        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        probs = probs[0]

        indices = np.argsort(probs)[::-1][:topk]
        results = [
            {
                "label": self.class_names[int(i)],
                "confidence": float(probs[int(i)]),
            }
            for i in indices
        ]

        return PredictionResult(
            top1_label=results[0]["label"],
            top1_confidence=results[0]["confidence"],
            topk=results,
        )
