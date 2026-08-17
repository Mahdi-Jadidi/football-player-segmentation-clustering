import cv2
import numpy as np
from scipy import ndimage


def filter_clusters(label_map: np.ndarray, min_fraction: float, max_fraction: float, dilation_iterations: int, ignore_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    keep = np.zeros(label_map.shape, dtype=bool); total = label_map.size
    for label in np.unique(label_map):
        if ignore_noise and label == -1: continue
        cluster = label_map == label; fraction = cluster.sum() / total
        if min_fraction <= fraction <= max_fraction: keep |= cluster
    dilated = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=dilation_iterations); merged, _ = ndimage.label(dilated); return keep, merged * keep


def mask_and_centroids(merged_labels: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    mask = (merged_labels > 0).astype(np.uint8); labels, count = ndimage.label(mask); return mask, list(ndimage.center_of_mass(mask, labels, range(1, count + 1)))
