# Football Player Segmentation via Pixel Clustering

Unsupervised football-player segmentation using pixel color, position, and optional deep features. The pipeline compares K-Means, DBSCAN, and agglomerative clustering, filters implausible regions, merges connected components, locates player centroids, and evaluates predicted masks against COCO polygon annotations.

## Modules

`data.py` handles images and COCO annotations, `features.py` builds RGB/HSV/spatial representations, `clustering.py` provides the three backends and parameter tuning, `postprocess.py` creates player candidates, and `evaluation.py` computes Dice and IoU. `pipeline.py` runs the complete batch workflow.

## Run

```bash
pip install -e .
player-segmentation run --images-dir dataset/images \
  --annotations dataset/annotations/instances_default.json \
  --method dbscan --output-dir outputs
```

Use `player-segmentation benchmark --image path/to/image.jpg` to compare cluster quality before a batch run. The output contains masks, centroids, per-image scores, aggregate metrics, and the resolved experiment configuration.

## Topics

`computer-vision` `image-segmentation` `clustering` `dbscan` `opencv` `coco`
