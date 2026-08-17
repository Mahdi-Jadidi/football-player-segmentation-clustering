import json
from pathlib import Path

import cv2
import numpy as np


def image_paths(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def load_image(path: Path, scale: float = 1.0) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None: raise ValueError(f"Cannot read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if scale != 1: image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def load_coco(path: Path) -> tuple[dict, dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8")); images = {item["id"]: item for item in data["images"]}; annotations = {}
    for annotation in data["annotations"]: annotations.setdefault(annotation["image_id"], []).append(annotation)
    return images, annotations, {Path(item["file_name"]).name: item["id"] for item in data["images"]}


def annotation_mask(annotations: list[dict], source_shape: tuple[int, int], target_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(source_shape, dtype=np.uint8)
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list):
            for polygon in segmentation: cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32).reshape(-1, 2)], 1)
    return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
