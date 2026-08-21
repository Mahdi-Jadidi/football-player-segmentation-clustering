# Football Player Segmentation via Pixel Clustering

An unsupervised computer-vision pipeline that turns football broadcast images into player candidates without training a semantic-segmentation network. It treats segmentation as a clustering and spatial-reasoning problem, then evaluates the resulting masks against COCO polygon annotations.

## Problem

Broadcast scenes are hard for naive colour thresholding: grass, kits, shadows, lines, crowds, and camera perspective all compete in the image. The goal is to locate plausible player regions using no class-labelled training images.

## What was built

- Pixel representations that combine RGB/HSV colour, spatial position, and optional deep features.
- K-Means, DBSCAN, and agglomerative clustering backends with comparable configuration.
- Region filtering, connected-component merging, and player-centroid extraction.
- COCO-based Dice and IoU evaluation, per-image reports, masks, and overlay-ready centroids.

## Main takeaways

The project demonstrates the trade-off between fast global colour grouping and density-aware clustering. DBSCAN is particularly useful when it can separate compact player-like regions from large, uniform background areas, while post-processing is what converts raw clusters into useful detections.

## Pipeline

```text
image + COCO polygons -> pixel features -> clustering -> region cleanup
                                             -> masks/centroids -> Dice and IoU
```

## Reproduce

```bash
pip install -e .
player-segmentation run --images-dir dataset/images \
  --annotations dataset/annotations/instances_default.json \
  --method dbscan --output-dir outputs
```

Use `player-segmentation benchmark` for a single-image comparison before a batch run. The repository is tested in GitHub Actions.
