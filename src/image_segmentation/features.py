import cv2
import numpy as np


def make_features(image: np.ndarray, mode: str = "rgb_xy") -> np.ndarray:
    height, width, _ = image.shape; rgb = image.reshape(-1, 3).astype(np.float32) / 255; yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij"); xy = np.column_stack([xx.ravel() / width, yy.ravel() / height]).astype(np.float32); hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32) / 255
    if mode == "rgb": return rgb
    if mode == "rgb_xy": return np.column_stack([rgb, xy])
    if mode == "hsv_xy": return np.column_stack([hsv, xy])
    raise ValueError(f"Unknown feature mode: {mode}")
