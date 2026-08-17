import numpy as np

from image_segmentation.evaluation import dice_iou
from image_segmentation.features import make_features


def test_feature_shapes() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    assert make_features(image, "rgb").shape == (200, 3)
    assert make_features(image, "rgb_xy").shape == (200, 5)


def test_perfect_mask_scores_one() -> None:
    mask = np.eye(5, dtype=np.uint8)
    dice, iou = dice_iou(mask, mask)
    assert dice > .999
    assert iou > .999
