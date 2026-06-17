#!/usr/bin/env python3
"""
select_best_detection.py - Stage 4 of the SignNav converter.

Input layout:
    signnav_scripts/datasets/detection/<bag_name>/frames/*.jpg
    signnav_scripts/datasets/detection/<bag_name>/detections.csv

Output layout:
    signnav_scripts/datasets/detection/<bag_name>/best/best.jpg
    signnav_scripts/datasets/detection/<bag_name>/best/best_metadata.json
    signnav_scripts/datasets/detection/<bag_name>/best/best_candidates.csv
    signnav_scripts/datasets/detection/<bag_name>/best_manifest.json

The selector prefers real YOLO detections. If a bag has no detected crops and
fallback crops exist, it can still choose the clearest fallback crop unless
--no-fallback is passed.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_detection_root() -> Path:
    return script_dir().parent / "datasets" / "detection"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def signature_for(data: object) -> str:
    return sha256_bytes(stable_json(data).encode("utf-8"))


def frame_sort_key(path: Path) -> Tuple[int, object]:
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.name


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def discover_bags(detection_root: Path) -> List[Path]:
    if not detection_root.exists():
        return []
    bags = []
    for frames_dir in sorted(detection_root.rglob("frames")):
        if not frames_dir.is_dir():
            continue
        bag_dir = frames_dir.parent
        detections_csv = bag_dir / "detections.csv"
        if detections_csv.exists():
            bags.append(bag_dir)
    return bags


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    fieldnames = [
        "rank",
        "score",
        "frame_filename",
        "crop_filename",
        "detected",
        "confidence",
        "class_name",
        "detection_source",
        "crop_width",
        "crop_height",
        "sharpness",
        "brightness",
        "contrast",
        "sharpness_score",
        "clarity_score",
        "size_score",
        "area_score",
        "center_score",
        "aspect_score",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def detection_state_hash(bag_dir: Path) -> str:
    detect_manifest = bag_dir / "detect_manifest.json"
    if detect_manifest.exists():
        return sha256_file(detect_manifest)

    frames_dir = bag_dir / "frames"
    entries = []
    if frames_dir.exists():
        for path in sorted(frames_dir.iterdir(), key=frame_sort_key):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                st = path.stat()
                entries.append({"filename": path.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    detections_csv = bag_dir / "detections.csv"
    if detections_csv.exists():
        entries.append({"detections_csv_sha256": sha256_file(detections_csv)})
    return signature_for(entries)


def manifest_is_current(manifest_path: Path, signature: str, best_dir: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("signature") != signature:
        return False

    selected = manifest.get("selected")
    if selected is None:
        return best_dir.exists() and (best_dir / "best_candidates.csv").exists()
    output_filename = selected.get("output_filename", "")
    return bool(output_filename) and (best_dir / output_filename).exists()


def import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: opencv-python\n"
            "Install it with:\n"
            "    python3 -m pip install opencv-python"
        ) from exc
    return cv2


def crop_quality(cv2_module: object, crop_path: Path) -> Optional[Dict[str, float]]:
    image = cv2_module.imread(str(crop_path), cv2_module.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None

    gray = cv2_module.cvtColor(image, cv2_module.COLOR_BGR2GRAY)
    sharpness = float(cv2_module.Laplacian(gray, cv2_module.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    bright_threshold = max(105, int(brightness + 0.45 * contrast))
    _, bright = cv2_module.threshold(gray, bright_threshold, 255, cv2_module.THRESH_BINARY)
    bright = cv2_module.morphologyEx(
        bright,
        cv2_module.MORPH_OPEN,
        cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    n_labels, _, stats, _ = cv2_module.connectedComponentsWithStats(bright, 8)
    small_components = 0
    small_component_area = 0
    max_component_area = max(80, int(width * height * 0.006))
    for label_id in range(1, n_labels):
        _, _, comp_w, comp_h, comp_area = stats[label_id]
        if (
            3 <= comp_area <= max_component_area
            and 2 <= comp_w <= width * 0.6
            and 2 <= comp_h <= height * 0.35
        ):
            small_components += 1
            small_component_area += int(comp_area)

    sharpness_score = min(math.log1p(sharpness) / math.log1p(1500.0), 1.0)
    contrast_score = min(contrast / 70.0, 1.0)
    brightness_score = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    darkness_score = max(0.0, min((112.0 - brightness) / 72.0, 1.0))
    component_count_score = min(small_components / 18.0, 1.0)
    component_density_score = min((small_components / max(1.0, (width * height) / 1_000_000.0)) / 260.0, 1.0)
    component_area_score = min((small_component_area / max(1.0, width * height)) / 0.035, 1.0)
    text_score = max(component_count_score, 0.8 * component_density_score, component_area_score)
    clarity_score = 0.65 * sharpness_score + 0.20 * contrast_score + 0.15 * brightness_score
    size_score = min(math.sqrt(width * height) / 900.0, 1.0)

    aspect = max(width / max(1, height), height / max(1, width))
    if aspect <= 2.5:
        aspect_score = 1.0
    else:
        aspect_score = max(0.0, 1.0 - (aspect - 2.5) / 4.0)

    return {
        "width": float(width),
        "height": float(height),
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness_score": sharpness_score,
        "clarity_score": clarity_score,
        "darkness_score": darkness_score,
        "text_score": text_score,
        "small_components": float(small_components),
        "size_score": size_score,
        "aspect_score": aspect_score,
    }


def center_score(row: Dict[str, str]) -> float:
    image_width = safe_float(row.get("image_width"), 0.0)
    image_height = safe_float(row.get("image_height"), 0.0)
    if image_width <= 0 or image_height <= 0:
        return 0.5

    crop_x1 = safe_float(row.get("crop_x1"), 0.0)
    crop_y1 = safe_float(row.get("crop_y1"), 0.0)
    crop_x2 = safe_float(row.get("crop_x2"), image_width)
    crop_y2 = safe_float(row.get("crop_y2"), image_height)
    cx = (crop_x1 + crop_x2) / 2.0
    cy = (crop_y1 + crop_y2) / 2.0

    dx = abs(cx - image_width / 2.0) / max(1.0, image_width / 2.0)
    dy = abs(cy - image_height / 2.0) / max(1.0, image_height / 2.0)
    distance = math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)
    return max(0.0, 1.0 - distance)


def area_score(row: Dict[str, str]) -> float:
    crop_width = safe_float(row.get("crop_width"), 0.0)
    crop_height = safe_float(row.get("crop_height"), 0.0)
    image_width = safe_float(row.get("image_width"), 0.0)
    image_height = safe_float(row.get("image_height"), 0.0)
    if crop_width <= 0 or crop_height <= 0 or image_width <= 0 or image_height <= 0:
        return 0.0
    area_ratio = (crop_width * crop_height) / max(1.0, image_width * image_height)
    small_ok = min(math.sqrt(area_ratio / 0.04), 1.0)
    large_ok = min((0.22 / max(area_ratio, 0.001)) ** 0.75, 1.0)
    return small_ok * large_ok


def crop_shape_score(row: Dict[str, str]) -> float:
    crop_width = safe_float(row.get("crop_width"), 0.0)
    crop_height = safe_float(row.get("crop_height"), 0.0)
    image_width = safe_float(row.get("image_width"), 0.0)
    image_height = safe_float(row.get("image_height"), 0.0)
    if crop_width <= 0 or crop_height <= 0 or image_width <= 0 or image_height <= 0:
        return 0.5

    area_ratio = (crop_width * crop_height) / max(1.0, image_width * image_height)
    width_ratio = crop_width / image_width
    height_ratio = crop_height / image_height
    aspect = max(crop_width / max(1.0, crop_height), crop_height / max(1.0, crop_width))

    score = 1.0
    if area_ratio > 0.24:
        score *= max(0.18, 1.0 - (area_ratio - 0.24) / 0.34)
    if width_ratio > 0.82:
        score *= 0.58
    if height_ratio > 0.88:
        score *= 0.58
    if aspect > 4.0:
        score *= max(0.25, 1.0 - (aspect - 4.0) / 3.0)

    crop_x1 = safe_float(row.get("crop_x1"), 0.0)
    crop_y1 = safe_float(row.get("crop_y1"), 0.0)
    crop_x2 = safe_float(row.get("crop_x2"), image_width)
    crop_y2 = safe_float(row.get("crop_y2"), image_height)
    touches = 0
    touches += int(crop_x1 <= 1)
    touches += int(crop_y1 <= 1)
    touches += int(crop_x2 >= image_width - 1)
    touches += int(crop_y2 >= image_height - 1)
    if touches == 1:
        score *= 0.82
    elif touches >= 2:
        score *= 0.48

    return max(0.05, score)


def class_usefulness(class_name: str) -> float:
    name = class_name.lower()
    if "heuristic_dark-panel" in name:
        return 1.10
    if "heuristic_edge-panel" in name:
        return 1.04
    if "stop sign" in name:
        return 1.04
    if "sign" in name:
        return 1.03
    if "traffic light" in name:
        return 0.96
    return 1.0


def score_candidate(row: Dict[str, str], quality: Dict[str, float]) -> Dict[str, object]:
    detected = str_to_bool(row.get("detected", "false"))
    confidence = safe_float(row.get("confidence"), 0.0)
    confidence_score = min(max(confidence, 0.0), 1.0)
    detection_source = row.get("detection_source", "")
    row_area_score = area_score(row)
    row_center_score = center_score(row)
    row_shape_score = crop_shape_score(row)
    usefulness = class_usefulness(row.get("class_name", ""))

    if detected:
        score = (
            0.42 * confidence_score
            + 0.27 * quality["clarity_score"]
            + 0.14 * quality["size_score"]
            + 0.08 * row_area_score
            + 0.05 * row_center_score
            + 0.04 * quality["aspect_score"]
        )
        score *= usefulness
    elif detection_source == "heuristic":
        score = (
            0.38 * confidence_score
            + 0.27 * quality["clarity_score"]
            + 0.12 * quality["size_score"]
            + 0.12 * quality["text_score"]
            + 0.08 * quality["darkness_score"]
            + 0.08 * row_area_score
            + 0.06 * row_center_score
            + 0.03 * quality["aspect_score"]
        )
        if row.get("class_name", "") == "heuristic_dark-panel" and quality["brightness"] > 84.0:
            score *= max(0.45, 1.0 - (quality["brightness"] - 84.0) / 45.0)
        crop_width = safe_float(row.get("crop_width"), quality["width"])
        crop_height = safe_float(row.get("crop_height"), quality["height"])
        horizontal_aspect = crop_width / max(1.0, crop_height)
        if (
            row.get("class_name", "") == "heuristic_dark-panel"
            and horizontal_aspect > 1.45
            and quality["sharpness_score"] < 0.45
            and row_area_score > 0.82
        ):
            score *= 0.70
        score *= usefulness * row_shape_score
    else:
        # Full-frame fallback is only a last resort. Large hallway crops can look
        # sharp/clear, but they are not useful sign crops.
        score = 0.28 * quality["clarity_score"] + 0.07 * quality["size_score"] + 0.10 * row_center_score
        score = min(score, 0.42)

    return {
        "score": score,
        "frame_filename": row.get("frame_filename", ""),
        "crop_filename": row.get("crop_filename", ""),
        "detected": str(detected).lower(),
        "confidence": confidence,
        "class_name": row.get("class_name", ""),
        "detection_source": detection_source,
        "crop_width": safe_int(row.get("crop_width"), int(quality["width"])),
        "crop_height": safe_int(row.get("crop_height"), int(quality["height"])),
        "sharpness": quality["sharpness"],
        "brightness": quality["brightness"],
        "contrast": quality["contrast"],
        "sharpness_score": quality["sharpness_score"],
        "clarity_score": quality["clarity_score"],
        "size_score": quality["size_score"],
        "area_score": row_area_score,
        "center_score": row_center_score,
        "aspect_score": quality["aspect_score"],
    }


def clean_best_dir(best_dir: Path) -> None:
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)


def select_for_bag(
    args: argparse.Namespace,
    cv2_module: object,
    bag_dir: Path,
    runtime_config: Dict[str, object],
) -> str:
    frames_dir = bag_dir / "frames"
    detections_csv = bag_dir / "detections.csv"
    best_dir = bag_dir / args.best_dir_name
    manifest_path = bag_dir / "best_manifest.json"

    state_hash = detection_state_hash(bag_dir)
    bag_rel = bag_dir.relative_to(args.detection_root)
    bag_signature = signature_for(
        {
            "runtime_config": runtime_config,
            "detection_state_hash": state_hash,
            "bag_relative_path": bag_rel.as_posix(),
        }
    )

    if not args.force and manifest_is_current(manifest_path, bag_signature, best_dir):
        print(f"SKIP {bag_rel}: best crop already selected with this code/config")
        return "skipped"

    rows = load_rows(detections_csv)
    crop_rows = []
    for row in rows:
        crop_filename = row.get("crop_filename", "")
        if not crop_filename:
            continue
        crop_path = frames_dir / crop_filename
        if crop_path.exists() and crop_path.suffix.lower() in IMAGE_EXTS:
            row = dict(row)
            row["_crop_path"] = str(crop_path)
            crop_rows.append(row)

    detected_rows = [
        row
        for row in crop_rows
        if str_to_bool(row.get("detected", "false")) and safe_float(row.get("confidence"), 0.0) >= args.min_confidence
    ]
    heuristic_rows = [
        row
        for row in crop_rows
        if row.get("detection_source", "") == "heuristic"
        and safe_float(row.get("confidence"), 0.0) >= args.min_confidence
    ]
    dark_panel_rows = [row for row in heuristic_rows if row.get("class_name", "") == "heuristic_dark-panel"]
    edge_panel_rows = [row for row in heuristic_rows if row.get("class_name", "") == "heuristic_edge-panel"]
    if detected_rows:
        candidate_rows = detected_rows
        candidate_source = "detected"
    elif dark_panel_rows:
        candidate_rows = dark_panel_rows
        candidate_source = "heuristic_dark-panel"
    elif edge_panel_rows:
        candidate_rows = edge_panel_rows
        candidate_source = "heuristic_edge-panel"
    elif args.allow_fallback:
        candidate_rows = crop_rows
        candidate_source = "fallback"
    else:
        candidate_rows = []
        candidate_source = "none"

    clean_best_dir(best_dir)
    scored = []
    for row in candidate_rows:
        crop_path = Path(row["_crop_path"])
        quality = crop_quality(cv2_module, crop_path)
        if quality is None:
            continue
        scored_row = score_candidate(row, quality)
        scored_row["_crop_path"] = str(crop_path)
        scored.append(scored_row)

    scored.sort(key=lambda item: float(item["score"]), reverse=True)

    candidate_rows_for_csv = []
    for rank, item in enumerate(scored, start=1):
        candidate_rows_for_csv.append(
            {
                "rank": rank,
                "score": f"{float(item['score']):.6f}",
                "frame_filename": item["frame_filename"],
                "crop_filename": item["crop_filename"],
                "detected": item["detected"],
                "confidence": f"{float(item['confidence']):.6f}",
                "class_name": item["class_name"],
                "detection_source": item["detection_source"],
                "crop_width": item["crop_width"],
                "crop_height": item["crop_height"],
                "sharpness": f"{float(item['sharpness']):.3f}",
                "brightness": f"{float(item['brightness']):.3f}",
                "contrast": f"{float(item['contrast']):.3f}",
                "sharpness_score": f"{float(item['sharpness_score']):.6f}",
                "clarity_score": f"{float(item['clarity_score']):.6f}",
                "size_score": f"{float(item['size_score']):.6f}",
                "area_score": f"{float(item['area_score']):.6f}",
                "center_score": f"{float(item['center_score']):.6f}",
                "aspect_score": f"{float(item['aspect_score']):.6f}",
            }
        )
    write_rows(best_dir / "best_candidates.csv", candidate_rows_for_csv)

    selected = None
    if scored:
        best = scored[0]
        source_path = Path(best["_crop_path"])
        suffix = source_path.suffix.lower() or ".jpg"
        output_filename = f"{args.output_basename}{suffix}"
        output_path = best_dir / output_filename
        shutil.copy2(source_path, output_path)
        selected = {
            "output_filename": output_filename,
            "source_crop": str(source_path),
            "frame_filename": best["frame_filename"],
            "crop_filename": best["crop_filename"],
            "detected": best["detected"],
            "confidence": float(best["confidence"]),
            "class_name": best["class_name"],
            "score": float(best["score"]),
            "candidate_source": candidate_source,
        }

        metadata = {
            "selected_at_utc": utc_now(),
            "bag_relative_path": bag_rel.as_posix(),
            "selected": selected,
            "scoring": {
                "min_confidence": args.min_confidence,
                "allow_fallback": args.allow_fallback,
                "candidate_count": len(scored),
            },
        }
        with open(best_dir / "best_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")

    manifest = {
        "pipeline": "select_best_detection",
        "created_at_utc": utc_now(),
        "signature": bag_signature,
        "runtime_config": runtime_config,
        "detection_state_hash": state_hash,
        "bag_relative_path": bag_rel.as_posix(),
        "candidate_source": candidate_source,
        "candidate_count": len(scored),
        "selected": selected,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    if selected is None:
        print(f"DONE {bag_rel}: no usable crop found")
    else:
        print(
            f"DONE {bag_rel}: {selected['output_filename']} "
            f"from {selected['crop_filename']} score={selected['score']:.3f}"
        )
    return "processed"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Select the best detected sign crop for each bag.")
    ap.add_argument("--detection-root", type=Path, default=default_detection_root())
    ap.add_argument("--best-dir-name", default="best")
    ap.add_argument("--output-basename", default="best")
    ap.add_argument("--min-confidence", type=float, default=0.15)
    ap.add_argument("--no-fallback", dest="allow_fallback", action="store_false")
    ap.set_defaults(allow_fallback=True)
    ap.add_argument("--force", action="store_true", help="Rerun even when manifests say outputs are current.")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    args.detection_root = args.detection_root.resolve()

    bags = discover_bags(args.detection_root)
    if not bags:
        print(f"No detection bags found under: {args.detection_root}")
        print("Run detect_sign_crops.py first.")
        return 0

    cv2_module = import_cv2()
    runtime_config = {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "best_dir_name": args.best_dir_name,
        "output_basename": args.output_basename,
        "min_confidence": args.min_confidence,
        "allow_fallback": args.allow_fallback,
    }

    counts = {"processed": 0, "skipped": 0}
    for bag_dir in bags:
        status = select_for_bag(args, cv2_module, bag_dir, runtime_config)
        counts[status] = counts.get(status, 0) + 1

    print(f"Finished: {counts.get('processed', 0)} processed, {counts.get('skipped', 0)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
