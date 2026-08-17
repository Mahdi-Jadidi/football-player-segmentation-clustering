import numpy as np


def dice_iou(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction, target = prediction.astype(bool), target.astype(bool); intersection = np.logical_and(prediction, target).sum(); union = np.logical_or(prediction, target).sum(); return (2 * intersection / (prediction.sum() + target.sum() + 1e-8), intersection / (union + 1e-8))
