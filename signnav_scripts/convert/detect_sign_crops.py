#!/usr/bin/env python3
"""
detect_sign_crops.py - Stage 3 of the SignNav converter.

Input layout:
    signnav_scripts/datasets/extracted/<bag_name>/frames/*.jpg

Output layout:
    signnav_scripts/datasets/detection/<bag_name>/frames/*.jpg
    signnav_scripts/datasets/detection/<bag_name>/detections.csv
    signnav_scripts/datasets/detection/<bag_name>/detect_manifest.json

The script produces at most one crop per source frame, with the same filename as
the source frame. By default, if YOLO finds no sign-like object in a frame, a
deterministic high-contrast panel heuristic tries to crop a likely indoor sign.
Rows with no actual YOLO detection are marked detected=false in detections.csv.
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
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTO_CLASS_KEYWORDS = (
    "sign",
    "traffic light",
    "wayfinding",
    "marker",
    "arrow",
    "direction",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_extracted_root() -> Path:
    return script_dir().parent / "datasets" / "extracted"


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


def image_files(frames_dir: Path) -> List[Path]:
    return sorted(
        [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=frame_sort_key,
    )


def discover_bags(extracted_root: Path) -> List[Tuple[Path, Path, Path, List[Path]]]:
    bags = []
    if not extracted_root.exists():
        return bags

    for frames_dir in sorted(extracted_root.rglob("frames")):
        if not frames_dir.is_dir():
            continue
        frames = image_files(frames_dir)
        if not frames:
            continue
        bag_dir = frames_dir.parent
        bag_rel = bag_dir.relative_to(extracted_root)
        bags.append((bag_rel, bag_dir, frames_dir, frames))
    return bags


def source_inventory(frames: Sequence[Path], frames_dir: Path) -> Tuple[List[Dict[str, object]], str]:
    entries = []
    for path in frames:
        st = path.stat()
        entries.append(
            {
                "filename": path.relative_to(frames_dir).as_posix(),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        )
    return entries, signature_for(entries)


def normalize_model_names(names: object) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def resolve_allowed_classes(names: Dict[int, str], class_spec: str) -> Optional[Set[int]]:
    spec = class_spec.strip()
    if not spec or spec.lower() == "all":
        return None

    if spec.lower() == "auto":
        allowed = {
            class_id
            for class_id, class_name in names.items()
            if any(keyword in class_name.lower() for keyword in AUTO_CLASS_KEYWORDS)
        }
        return allowed or None

    by_name = {class_name.lower(): class_id for class_id, class_name in names.items()}
    allowed = set()
    missing = []
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token.isdigit():
            allowed.add(int(token))
            continue
        class_id = by_name.get(token.lower())
        if class_id is None:
            missing.append(token)
        else:
            allowed.add(class_id)

    if missing:
        available = ", ".join(f"{i}:{name}" for i, name in sorted(names.items()))
        raise SystemExit(
            "Could not resolve class name(s): "
            + ", ".join(missing)
            + "\nAvailable YOLO classes: "
            + available
        )
    return allowed or None


def expand_bbox(
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    dx = box_w * margin
    dy = box_h * margin
    crop_x1 = max(0, int(x1 - dx))
    crop_y1 = max(0, int(y1 - dy))
    crop_x2 = min(image_width, int(x2 + dx + 0.9999))
    crop_y2 = min(image_height, int(y2 + dy + 0.9999))
    return crop_x1, crop_y1, crop_x2, crop_y2


def save_crop(cv2_module: object, out_path: Path, crop: object) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        ok = cv2_module.imwrite(
            str(out_path),
            crop,
            [
                int(cv2_module.IMWRITE_JPEG_QUALITY),
                100,
                int(cv2_module.IMWRITE_JPEG_OPTIMIZE),
                1,
            ],
        )
    elif suffix == ".png":
        ok = cv2_module.imwrite(str(out_path), crop, [int(cv2_module.IMWRITE_PNG_COMPRESSION), 1])
    else:
        ok = cv2_module.imwrite(str(out_path), crop)

    if not ok:
        raise RuntimeError(f"Failed to write crop: {out_path}")


def choose_detection(
    result: object,
    names: Dict[int, str],
    allowed_classes: Optional[Set[int]],
    image_width: int,
    image_height: int,
) -> Optional[Dict[str, object]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)

    image_area = max(1, image_width * image_height)
    candidates = []
    for box, conf, class_id in zip(xyxy, confs, classes):
        if allowed_classes is not None and int(class_id) not in allowed_classes:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        box_area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / image_area)
        area_bonus = min(box_area_ratio / 0.08, 1.0) * 0.12
        score = float(conf) + area_bonus
        candidates.append(
            {
                "bbox": (x1, y1, x2, y2),
                "confidence": float(conf),
                "class_id": int(class_id),
                "class_name": names.get(int(class_id), str(class_id)),
                "score": score,
                "box_area_ratio": box_area_ratio,
            }
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def choose_heuristic_sign(cv2_module: object, image: object) -> Optional[Dict[str, object]]:
    """Find a likely sign panel when YOLO has no matching sign-like class."""
    image_height, image_width = image.shape[:2]
    scale = min(1.0, 960.0 / max(image_width, image_height))
    if scale < 1.0:
        small = cv2_module.resize(
            image,
            (int(image_width * scale), int(image_height * scale)),
            interpolation=cv2_module.INTER_AREA,
        )
    else:
        small = image

    small_height, small_width = small.shape[:2]
    gray = cv2_module.cvtColor(small, cv2_module.COLOR_BGR2GRAY)
    blur = cv2_module.GaussianBlur(gray, (3, 3), 0)
    edges = cv2_module.Canny(blur, 50, 150)

    edge_kernel = cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (9, 9))
    edge_mask = cv2_module.morphologyEx(edges, cv2_module.MORPH_CLOSE, edge_kernel, iterations=2)

    _, dark_mask = cv2_module.threshold(blur, 82, 255, cv2_module.THRESH_BINARY_INV)
    dark_kernel = cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (7, 7))
    dark_mask = cv2_module.morphologyEx(dark_mask, cv2_module.MORPH_OPEN, dark_kernel, iterations=1)
    dark_mask = cv2_module.morphologyEx(dark_mask, cv2_module.MORPH_CLOSE, dark_kernel, iterations=2)

    candidates = []

    def add_candidates(mask: object, source: str) -> None:
        found = cv2_module.findContours(mask, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        image_area = max(1, small_width * small_height)

        for contour in contours:
            x, y, w, h = cv2_module.boundingRect(contour)
            if w < 24 or h < 24:
                continue

            area = w * h
            area_ratio = area / image_area
            if area_ratio < 0.003 or area_ratio > 0.45:
                continue

            aspect = w / max(1, h)
            if aspect < 0.18 or aspect > 6.0:
                continue

            crop_gray = gray[y : y + h, x : x + w]
            crop_edges = edges[y : y + h, x : x + w]
            if crop_gray.size == 0:
                continue

            edge_density = float(cv2_module.countNonZero(crop_edges)) / max(1, area)
            contrast = float(crop_gray.std())
            if edge_density < 0.008 and contrast < 20.0:
                continue

            contour_area = float(cv2_module.contourArea(contour))
            rectangularity = min(contour_area / max(1.0, float(area)), 1.0)
            area_small_ok = min(math.sqrt(area_ratio / 0.04), 1.0)
            area_large_ok = min(math.sqrt(0.32 / max(area_ratio, 0.001)), 1.0)
            area_score = area_small_ok * area_large_ok
            edge_score = min(edge_density / 0.10, 1.0)
            contrast_score = min(contrast / 72.0, 1.0)

            if 0.35 <= aspect <= 4.5:
                aspect_score = 1.0
            else:
                aspect_score = max(0.0, 1.0 - abs(math.log(max(aspect, 0.01))) / 2.5)

            center_y = (y + h / 2.0) / max(1, small_height)
            upper_score = 1.0 if center_y <= 0.68 else max(0.0, 1.0 - (center_y - 0.68) / 0.32)
            source_bonus = 0.08 if source == "dark-panel" else 0.0
            score = (
                0.32 * edge_score
                + 0.25 * contrast_score
                + 0.20 * area_score
                + 0.10 * rectangularity
                + 0.05 * aspect_score
                + 0.08 * upper_score
                + source_bonus
            )

            inv_scale = 1.0 / scale
            candidates.append(
                {
                    "bbox": (
                        x * inv_scale,
                        y * inv_scale,
                        (x + w) * inv_scale,
                        (y + h) * inv_scale,
                    ),
                    "score": min(score, 1.0),
                    "source": source,
                    "box_area_ratio": (w * h) / max(1, small_width * small_height),
                }
            )

    add_candidates(edge_mask, "edge-panel")
    add_candidates(dark_mask, "dark-panel")

    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def load_existing_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def manifest_is_current(
    manifest_path: Path,
    signature: str,
    detections_csv: Path,
    output_frames_dir: Path,
) -> bool:
    if not manifest_path.exists() or not detections_csv.exists() or not output_frames_dir.exists():
        return False
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("signature") != signature:
        return False

    rows = load_existing_rows(detections_csv)
    expected_crop_count = int(manifest.get("output", {}).get("crop_count", -1))
    actual_crop_count = 0
    for row in rows:
        crop_filename = row.get("crop_filename", "")
        if not crop_filename:
            continue
        if not (output_frames_dir / crop_filename).exists():
            return False
        actual_crop_count += 1
    return actual_crop_count == expected_crop_count


def clean_previous_outputs(output_frames_dir: Path, detections_csv: Path, manifest_path: Path) -> None:
    if output_frames_dir.exists():
        shutil.rmtree(output_frames_dir)
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    for path in (detections_csv, manifest_path):
        if path.exists():
            path.unlink()


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


def import_yolo():
    try:
        from ultralytics import YOLO  # type: ignore
        import ultralytics  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics\n"
            "Install it with:\n"
            "    python3 -m pip install ultralytics opencv-python"
        ) from exc
    return YOLO, getattr(ultralytics, "__version__", "unknown")


def write_rows(csv_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    fieldnames = [
        "frame_filename",
        "crop_filename",
        "detected",
        "confidence",
        "class_id",
        "class_name",
        "selected_score",
        "image_width",
        "image_height",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "crop_width",
        "crop_height",
        "box_area_ratio",
        "detection_source",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_bag(
    args: argparse.Namespace,
    model: object,
    names: Dict[int, str],
    allowed_classes: Optional[Set[int]],
    runtime_config: Dict[str, object],
    cv2_module: object,
    bag_rel: Path,
    bag_dir: Path,
    frames_dir: Path,
    frames: Sequence[Path],
) -> str:
    output_bag_dir = args.detection_root / bag_rel
    output_frames_dir = output_bag_dir / "frames"
    detections_csv = output_bag_dir / "detections.csv"
    manifest_path = output_bag_dir / "detect_manifest.json"

    inventory_entries, inventory_hash = source_inventory(frames, frames_dir)
    bag_signature = signature_for(
        {
            "runtime_config": runtime_config,
            "source_inventory_hash": inventory_hash,
            "bag_relative_path": bag_rel.as_posix(),
        }
    )

    if not args.force and manifest_is_current(manifest_path, bag_signature, detections_csv, output_frames_dir):
        print(f"SKIP {bag_rel}: already processed with this code/config")
        return "skipped"

    print(f"RUN  {bag_rel}: {len(frames)} frame(s)")
    output_bag_dir.mkdir(parents=True, exist_ok=True)
    clean_previous_outputs(output_frames_dir, detections_csv, manifest_path)

    predict_kwargs = {
        "source": [str(p.resolve()) for p in frames],
        "stream": True,
        "verbose": False,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "batch": args.batch,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    rows = []
    crop_count = 0
    detection_count = 0

    for result_index, result in enumerate(model.predict(**predict_kwargs)):
        if result_index >= len(frames):
            break
        frame_path = frames[result_index]
        frame_filename = frame_path.name

        row: Dict[str, object] = {
            "frame_filename": frame_filename,
            "crop_filename": "",
            "detected": "false",
            "confidence": "0.000000",
            "class_id": "",
            "class_name": "",
            "selected_score": "0.000000",
            "image_width": "",
            "image_height": "",
            "bbox_x1": "",
            "bbox_y1": "",
            "bbox_x2": "",
            "bbox_y2": "",
            "crop_x1": "",
            "crop_y1": "",
            "crop_x2": "",
            "crop_y2": "",
            "crop_width": "",
            "crop_height": "",
            "box_area_ratio": "0.000000",
            "detection_source": "",
            "error": "",
        }

        image = cv2_module.imread(str(frame_path), cv2_module.IMREAD_COLOR)
        if image is None:
            row["error"] = "could_not_read_image"
            rows.append(row)
            continue

        image_height, image_width = image.shape[:2]
        row["image_width"] = image_width
        row["image_height"] = image_height

        selected = choose_detection(result, names, allowed_classes, image_width, image_height)
        detection_source = "yolo" if selected is not None else ""
        heuristic = None
        if selected is None and args.fallback == "heuristic":
            heuristic = choose_heuristic_sign(cv2_module, image)

        if selected is None and heuristic is None and args.fallback == "skip":
            rows.append(row)
            continue

        if selected is None and heuristic is None:
            crop_box = (0, 0, image_width, image_height)
            bbox = ("", "", "", "")
            class_id = ""
            class_name = ""
            confidence = 0.0
            selected_score = 0.0
            box_area_ratio = 0.0
            detection_source = "full-frame"
        elif selected is None:
            bbox = heuristic["bbox"]
            crop_box = expand_bbox(bbox, image_width, image_height, args.margin)
            class_id = -1
            class_name = f"heuristic_{heuristic['source']}"
            confidence = float(heuristic["score"])
            selected_score = float(heuristic["score"])
            box_area_ratio = float(heuristic["box_area_ratio"])
            detection_source = "heuristic"
        else:
            bbox = selected["bbox"]
            crop_box = expand_bbox(bbox, image_width, image_height, args.margin)
            class_id = selected["class_id"]
            class_name = selected["class_name"]
            confidence = float(selected["confidence"])
            selected_score = float(selected["score"])
            box_area_ratio = float(selected["box_area_ratio"])
            detection_count += 1
            row["detected"] = "true"

        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            row["error"] = "empty_crop"
            rows.append(row)
            continue

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_path = output_frames_dir / frame_filename
        save_crop(cv2_module, crop_path, crop)
        crop_count += 1

        row.update(
            {
                "crop_filename": frame_filename,
                "confidence": f"{confidence:.6f}",
                "class_id": class_id,
                "class_name": class_name,
                "selected_score": f"{selected_score:.6f}",
                "bbox_x1": "" if bbox[0] == "" else f"{float(bbox[0]):.3f}",
                "bbox_y1": "" if bbox[1] == "" else f"{float(bbox[1]):.3f}",
                "bbox_x2": "" if bbox[2] == "" else f"{float(bbox[2]):.3f}",
                "bbox_y2": "" if bbox[3] == "" else f"{float(bbox[3]):.3f}",
                "crop_x1": crop_x1,
                "crop_y1": crop_y1,
                "crop_x2": crop_x2,
                "crop_y2": crop_y2,
                "crop_width": crop_x2 - crop_x1,
                "crop_height": crop_y2 - crop_y1,
                "box_area_ratio": f"{box_area_ratio:.6f}",
                "detection_source": detection_source,
            }
        )
        rows.append(row)

    write_rows(detections_csv, rows)

    manifest = {
        "pipeline": "detect_sign_crops",
        "created_at_utc": utc_now(),
        "signature": bag_signature,
        "runtime_config": runtime_config,
        "source": {
            "bag_relative_path": bag_rel.as_posix(),
            "bag_dir": str(bag_dir),
            "frames_dir": str(frames_dir),
            "frame_count": len(frames),
            "inventory_hash": inventory_hash,
            "inventory": inventory_entries,
        },
        "output": {
            "bag_dir": str(output_bag_dir),
            "frames_dir": str(output_frames_dir),
            "detections_csv": str(detections_csv),
            "crop_count": crop_count,
            "detected_crop_count": detection_count,
            "fallback_crop_count": crop_count - detection_count,
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        f"DONE {bag_rel}: {detection_count} detected crop(s), "
        f"{crop_count - detection_count} fallback crop(s)"
    )
    return "processed"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Detect and crop one sign candidate per extracted frame.")
    ap.add_argument("--extracted-root", type=Path, default=default_extracted_root())
    ap.add_argument("--detection-root", type=Path, default=default_detection_root())
    ap.add_argument("--model", default="yolov8n.pt", help="Ultralytics model name or local .pt path.")
    ap.add_argument(
        "--classes",
        default="auto",
        help="Comma-separated YOLO class names/ids, 'auto' for sign-like classes, or 'all'.",
    )
    ap.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU threshold.")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    ap.add_argument("--batch", type=int, default=8, help="YOLO batch size.")
    ap.add_argument("--device", default="", help="Ultralytics device string, e.g. '0' or 'cpu'.")
    ap.add_argument("--margin", type=float, default=0.10, help="Crop expansion in every direction.")
    ap.add_argument(
        "--fallback",
        choices=("heuristic", "full-frame", "skip"),
        default="heuristic",
        help="What to do when YOLO finds no sign-like object in a frame.",
    )
    ap.add_argument("--force", action="store_true", help="Rerun even when manifests say outputs are current.")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    args.extracted_root = args.extracted_root.resolve()
    args.detection_root = args.detection_root.resolve()

    bags = discover_bags(args.extracted_root)
    if not bags:
        print(f"No extracted bags with frames found under: {args.extracted_root}")
        print("Expected layout: <extracted-root>/<bag_name>/frames/*.jpg")
        return 0

    cv2_module = import_cv2()
    YOLO, ultralytics_version = import_yolo()
    model = YOLO(args.model)
    names = normalize_model_names(getattr(model, "names", {}))
    allowed_classes = resolve_allowed_classes(names, args.classes)

    if allowed_classes is None:
        class_description = "all classes"
    else:
        class_description = ", ".join(
            f"{class_id}:{names.get(class_id, class_id)}" for class_id in sorted(allowed_classes)
        )
    print(f"Model: {args.model}")
    print(f"Class filter: {class_description}")
    print(f"Extracted root: {args.extracted_root}")
    print(f"Detection root: {args.detection_root}")

    runtime_config = {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "model": args.model,
        "ultralytics_version": ultralytics_version,
        "classes": args.classes,
        "allowed_class_ids": None if allowed_classes is None else sorted(allowed_classes),
        "allowed_class_names": None
        if allowed_classes is None
        else [names.get(class_id, str(class_id)) for class_id in sorted(allowed_classes)],
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "margin": args.margin,
        "fallback": args.fallback,
    }

    counts = {"processed": 0, "skipped": 0}
    for bag_rel, bag_dir, frames_dir, frames in bags:
        status = process_bag(
            args,
            model,
            names,
            allowed_classes,
            runtime_config,
            cv2_module,
            bag_rel,
            bag_dir,
            frames_dir,
            frames,
        )
        counts[status] = counts.get(status, 0) + 1

    print(f"Finished: {counts.get('processed', 0)} processed, {counts.get('skipped', 0)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
