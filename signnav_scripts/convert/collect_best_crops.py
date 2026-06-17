#!/usr/bin/env python3
"""
collect_best_crops.py - Copy per-bag best crops into one review folder.

Input:
    signnav_scripts/datasets/detection/<bag_name>/best/best.jpg

Output:
    signnav_scripts/datasets/best_crops_by_bag/<bag_name>__best.jpg
    signnav_scripts/datasets/best_crops_by_bag/best_crops_index.csv
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_detection_root() -> Path:
    return script_dir().parent / "datasets" / "detection"


def default_output_root() -> Path:
    return script_dir().parent / "datasets" / "best_crops_by_bag"


def safe_name(path: Path) -> str:
    return "__".join(path.parts)


def load_selected(best_dir: Path) -> Dict[str, object]:
    meta_path = best_dir / "best_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    selected = metadata.get("selected")
    return selected if isinstance(selected, dict) else {}


def collect(detection_root: Path, output_root: Path, clear: bool) -> int:
    if clear and output_root.exists():
        for path in output_root.iterdir():
            if path.is_file():
                path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for best_path in sorted(detection_root.glob("*/best/best.jpg")):
        bag_dir = best_path.parents[1]
        bag_rel = bag_dir.relative_to(detection_root)
        bag_name = safe_name(bag_rel)
        copied_name = f"{bag_name}__best{best_path.suffix.lower() or '.jpg'}"
        copied_path = output_root / copied_name
        shutil.copy2(best_path, copied_path)

        selected = load_selected(best_path.parent)
        rows.append(
            {
                "bag": bag_rel.as_posix(),
                "copied_best": copied_name,
                "source_best": str(best_path),
                "source_crop": selected.get("source_crop", ""),
                "frame_filename": selected.get("frame_filename", ""),
                "crop_filename": selected.get("crop_filename", ""),
                "score": selected.get("score", ""),
                "class_name": selected.get("class_name", ""),
                "confidence": selected.get("confidence", ""),
            }
        )

    index_path = output_root / "best_crops_index.csv"
    with open(index_path, "w", newline="") as f:
        fieldnames = [
            "bag",
            "copied_best",
            "source_best",
            "source_crop",
            "frame_filename",
            "crop_filename",
            "score",
            "class_name",
            "confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Copied {len(rows)} best crop(s) into {output_root}")
    print(f"Wrote index -> {index_path}")
    return len(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Collect per-bag best crops into one folder.")
    ap.add_argument("--detection-root", type=Path, default=default_detection_root())
    ap.add_argument("--output-root", type=Path, default=default_output_root())
    ap.add_argument("--keep-existing", action="store_true", help="Do not clear old files from output-root first.")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    collect(args.detection_root.resolve(), args.output_root.resolve(), clear=not args.keep_existing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
