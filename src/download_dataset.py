from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# Kaggle dataset slug in the form "owner/dataset-name".
# Override at runtime with --dataset or the KAGGLE_DATASET environment variable.
DEFAULT_DATASET = os.environ.get("KAGGLE_DATASET", "xavierlermusieaux/kitti-yolo")

SPLIT_DIRS = ("images/train", "images/val", "labels/train", "labels/val")


def download_with_kagglehub(dataset: str) -> Path:
    import kagglehub

    print(f"Downloading '{dataset}' via kagglehub...")
    path = kagglehub.dataset_download(dataset)
    return Path(path)


def download_with_cli(dataset: str, dest: Path) -> Path:
    import subprocess

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{dataset}' via kaggle CLI...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"],
        check=True,
    )
    return dest


def find_dataset_root(search_root: Path) -> Path:
    """Locate the directory that contains the YOLO split folders."""
    if all((search_root / d).exists() for d in SPLIT_DIRS):
        return search_root

    for candidate in search_root.rglob("images"):
        root = candidate.parent
        if all((root / d).exists() for d in SPLIT_DIRS):
            return root

    raise FileNotFoundError(
        "Could not locate a YOLO-format dataset (expected images/{train,val} and "
        f"labels/{{train,val}}) under {search_root}."
    )


def link_into_place(source_root: Path, target_root: Path) -> None:
    for split in SPLIT_DIRS:
        src = source_root / split
        dst = target_root / split
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"Skipping existing {dst}")
            continue
        print(f"Copying {src} -> {dst}")
        shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the KITTI-format detection dataset from Kaggle into ./yolo."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Kaggle dataset slug, e.g. owner/dataset-name.",
    )
    parser.add_argument(
        "--target",
        default="yolo",
        help="Destination directory for the dataset (default: yolo).",
    )
    parser.add_argument(
        "--use-cli",
        action="store_true",
        help="Use the kaggle CLI instead of kagglehub.",
    )
    args = parser.parse_args()

    if not args.dataset:
        raise SystemExit(
            "No dataset specified. Pass --dataset owner/name or set KAGGLE_DATASET."
        )

    target_root = Path(args.target)

    if args.use_cli:
        download_root = download_with_cli(args.dataset, target_root / ".kaggle_cache")
    else:
        download_root = download_with_kagglehub(args.dataset)

    dataset_root = find_dataset_root(download_root)
    link_into_place(dataset_root, target_root)
    print(f"\nDataset ready under {target_root.resolve()}")


if __name__ == "__main__":
    main()
