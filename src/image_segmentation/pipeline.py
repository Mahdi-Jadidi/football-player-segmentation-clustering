import json

import cv2
import pandas as pd

from .clustering import cluster_image
from .config import SegmentationConfig
from .data import annotation_mask, image_paths, load_coco, load_image
from .evaluation import dice_iou
from .postprocess import filter_clusters, mask_and_centroids


def segment(image, config: SegmentationConfig):
    labels = cluster_image(image, config.method, config.feature_mode, config.clusters, config.eps, config.min_samples)
    _, merged = filter_clusters(labels, config.min_fraction, config.max_fraction, config.dilation_iterations, ignore_noise=config.method == "dbscan")
    return mask_and_centroids(merged)


def run_pipeline(config: SegmentationConfig) -> pd.DataFrame:
    mask_dir = config.output_dir / "masks"; mask_dir.mkdir(parents=True, exist_ok=True); coco = load_coco(config.annotations) if config.annotations else None; rows = []
    for path in image_paths(config.images_dir):
        image = load_image(path, config.scale); mask, centroids = segment(image, config); cv2.imwrite(str(mask_dir / f"{path.stem}.png"), mask * 255); record = {"file": path.name, "components": len(centroids), "centroids": json.dumps(centroids)}
        if coco and path.name in coco[2]:
            image_id = coco[2][path.name]; info = coco[0][image_id]; target = annotation_mask(coco[1].get(image_id, []), (info["height"], info["width"]), mask.shape); record["dice"], record["iou"] = dice_iou(mask, target)
        rows.append(record)
    results = pd.DataFrame(rows); results.to_csv(config.output_dir / "results.csv", index=False); (config.output_dir / "config.json").write_text(json.dumps({key: str(value) if hasattr(value, "parts") else value for key, value in config.__dict__.items()}, indent=2), encoding="utf-8"); return results
