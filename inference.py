from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from PIL import Image, ImageOps

from jedgenet_model import build_model


CLASS_NAMES = ("Cracked", "Dry", "Insect damaged", "Invalid")
IMAGE_SIZE = 64
RESIZE_SIDE = 73
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class Prediction:
    class_name: str
    class_index: int
    scores: tuple[float, ...]
    inference_ms: float


def load_rgb_image(source: str | Path | BinaryIO) -> Image.Image:
    image = Image.open(source)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("The image has invalid dimensions.")

    if width < height:
        resized_size = (RESIZE_SIDE, int(RESIZE_SIDE * height / width))
    else:
        resized_size = (int(RESIZE_SIDE * width / height), RESIZE_SIDE)

    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    left = int(round((resized.width - IMAGE_SIZE) / 2.0))
    top = int(round((resized.height - IMAGE_SIZE) / 2.0))
    cropped = resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    array = np.asarray(cropped, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


class JedgeNetPredictor:
    def __init__(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        self.model = build_model("jedgenet", num_classes=len(CLASS_NAMES))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self._lock = threading.Lock()

        with torch.inference_mode():
            self.model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))

    def predict(self, image: Image.Image) -> Prediction:
        tensor = preprocess_image(image)
        with self._lock, torch.inference_mode():
            started = time.perf_counter()
            logits = self.model(tensor)
            inference_ms = (time.perf_counter() - started) * 1000.0

        scores_tensor = torch.softmax(logits[0], dim=0)
        scores = tuple(float(value) for value in scores_tensor.tolist())
        class_index = int(torch.argmax(scores_tensor).item())
        return Prediction(
            class_name=CLASS_NAMES[class_index],
            class_index=class_index,
            scores=scores,
            inference_ms=inference_ms,
        )
