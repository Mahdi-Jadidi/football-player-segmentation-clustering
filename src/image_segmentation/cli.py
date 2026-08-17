import argparse
from pathlib import Path

from .clustering import benchmark
from .config import SegmentationConfig
from .data import load_image
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="player-segmentation"); commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run"); run.add_argument("--images-dir", type=Path, required=True); run.add_argument("--annotations", type=Path); run.add_argument("--output-dir", type=Path, default=Path("outputs")); run.add_argument("--method", choices=("dbscan", "kmeans", "agglomerative"), default="dbscan"); run.add_argument("--feature-mode", choices=("rgb", "rgb_xy", "hsv_xy"), default="rgb_xy"); run.add_argument("--scale", type=float, default=.25); run.add_argument("--eps", type=float, default=.05); run.add_argument("--min-samples", type=int, default=3); run.add_argument("--clusters", type=int, default=9)
    compare = commands.add_parser("benchmark"); compare.add_argument("--image", type=Path, required=True); compare.add_argument("--feature-mode", default="rgb_xy")
    args = parser.parse_args()
    if args.command == "benchmark": print(benchmark(load_image(args.image, .25), args.feature_mode).to_string(index=False))
    else: print(run_pipeline(SegmentationConfig(args.images_dir, args.output_dir, args.annotations, args.method, args.feature_mode, args.scale, args.eps, args.min_samples, args.clusters)).describe().to_string())


if __name__ == "__main__": main()
