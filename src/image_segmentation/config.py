from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentationConfig:
    images_dir: Path
    output_dir: Path = Path("outputs")
    annotations: Path | None = None
    method: str = "dbscan"
    feature_mode: str = "rgb_xy"
    scale: float = .25
    eps: float = .05
    min_samples: int = 3
    clusters: int = 9
    min_fraction: float = .00005
    max_fraction: float = .25
    dilation_iterations: int = 2
    seed: int = 42
