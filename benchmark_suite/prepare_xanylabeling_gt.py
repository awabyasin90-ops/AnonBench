#!/usr/bin/env python3
"""
prepare_xanylabeling_gt.py
==========================
Generates ONLY V.1 and V.2 Ground-Truth workspaces with Strict Non-Duplication IoU/IoMin Fusion.

Workspaces Created:
  1. V.1 (Visual Fusion Mode):
     - Output: `3_Systems/outputs/V.1`
     - Bounding boxes and badges burned directly onto image pixels.
     - Strict non-duplication IoU / IoMin fusion applied so no duplicate boxes appear in the same place.
     - Single master `annotations.json` file updated accordingly.
  
  2. V.2 (Native X-AnyLabeling / LabelMe Mode):
     - Output: `3_Systems/outputs/V.2`
     - Clean images without burned-in pixels.
     - Individual `<frame_stem>.json` file paired 1-to-1 with each `<frame_stem>.jpg`.
     - Strict non-duplication IoU / IoMin fusion applied (Zero duplicate boxes in the same place).
     - Standard LabelMe format (`shapes` with `label`, `points: [[x1, y1], [x2, y2]]`, `shape_type: "rectangle"`).
     - Stripped of confidence and bin clutter (Strict Simplicity Rule).
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np


PIPELINE_KEYS = [
    "1_FAST_V5",
    "2_EgoBlur",
    "3_DashcamCleaner",
    "4_FusionPortable",
    "5_Generative_Anonymizer"
]


def calculate_overlap(boxA: List[float], boxB: List[float]) -> Tuple[float, float]:
    """Computes IoU and Intersection-over-Minimum-Area (IoMin)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    if inter_area <= 0.0 or areaA <= 0.0 or areaB <= 0.0:
        return 0.0, 0.0

    union_area = areaA + areaB - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    iomin = inter_area / min(areaA, areaB)

    return iou, iomin


def is_duplicate(boxA: List[float], boxB: List[float], iou_thresh: float = 0.35, iomin_thresh: float = 0.50) -> bool:
    """Returns True if two boxes represent the same object (either high IoU or nested containment)."""
    iou, iomin = calculate_overlap(boxA, boxB)
    return (iou >= iou_thresh) or (iomin >= iomin_thresh)


def fuse_boxes_strict_non_dup(boxes: List[List[float]], iou_thresh: float = 0.35, iomin_thresh: float = 0.50) -> List[List[float]]:
    """
    Iteratively merges any overlapping or nested boxes on the same object into consensus boxes.
    Guarantees ZERO duplicate boxes in the same place.
    """
    if not boxes:
        return []

    current = [list(map(float, b)) for b in boxes]
    changed = True

    while changed:
        changed = False
        n = len(current)
        merged = False
        for i in range(n):
            for j in range(i + 1, n):
                if is_duplicate(current[i], current[j], iou_thresh, iomin_thresh):
                    b1 = current[i]
                    b2 = current[j]
                    # Robust consensus bounding coordinates
                    fused = [
                        round((b1[0] + b2[0]) / 2.0, 1),
                        round((b1[1] + b2[1]) / 2.0, 1),
                        round((b1[2] + b2[2]) / 2.0, 1),
                        round((b1[3] + b2[3]) / 2.0, 1)
                    ]
                    current.pop(j)
                    current[i] = fused
                    changed = True
                    merged = True
                    break
            if merged:
                break

    # Final filter: ensure strictly positive dimensions
    valid_fused = []
    for b in current:
        if b[2] > b[0] and b[3] > b[1]:
            valid_fused.append(b)

    return valid_fused


def select_frames(
    input_dir: Path | None,
    results_dir: Path,
    num_frames: int = 10,
    selected_frames_json: Path | None = None
) -> List[str]:
    """Selects frame filenames based on selected_frames.json or detections."""
    candidates = []
    if selected_frames_json and selected_frames_json.exists():
        with open(selected_frames_json, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    elif (results_dir / "blurred" / "telemetry" / "selected_frames.json").exists():
        with open(results_dir / "blurred" / "telemetry" / "selected_frames.json", "r", encoding="utf-8") as f:
            candidates = json.load(f)
    elif (results_dir / "telemetry" / "selected_frames.json").exists():
        with open(results_dir / "telemetry" / "selected_frames.json", "r", encoding="utf-8") as f:
            candidates = json.load(f)

    if not candidates and input_dir and input_dir.exists():
        raw_files = sorted([p.name for p in input_dir.glob("*.jpg")] + [p.name for p in input_dir.glob("*.png")])
        if raw_files:
            candidates = raw_files

    if not candidates:
        fast_dir = results_dir / "blurred" / "1_FAST_V5"
        if fast_dir.exists():
            candidates = sorted([p.name for p in fast_dir.glob("*.jpg")] + [p.name for p in fast_dir.glob("*.png")])

    if not candidates:
        raise FileNotFoundError(f"Could not locate evaluation frames in {results_dir} or {input_dir}")

    total = len(candidates)
    if total <= num_frames:
        return candidates

    indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    return [candidates[idx] for idx in indices]


def load_model_detections(results_dir: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Loads detections.json for all 5 anonymization systems."""
    blurred_dir = results_dir / "blurred" if (results_dir / "blurred").exists() else results_dir
    models_dets = {}

    for pipe_key in PIPELINE_KEYS:
        det_file = blurred_dir / pipe_key / "detections.json"
        if not det_file.exists() and "FusionPortable" in pipe_key:
            det_file = blurred_dir / "4_UnderstandAI" / "detections.json"

        if det_file.exists():
            with open(det_file, "r", encoding="utf-8") as f:
                models_dets[pipe_key] = json.load(f)
        else:
            models_dets[pipe_key] = {}

    return models_dets


def draw_detected_objects_on_frame(img: np.ndarray, shapes: List[Dict[str, Any]]) -> np.ndarray:
    """Draws detected bounding boxes and labels cleanly onto frame pixels for V.1."""
    canvas = img.copy()
    h, w = canvas.shape[:2]

    for sh in shapes:
        lbl = sh.get("label", "face")
        pts = sh.get("points", [])
        if len(pts) < 2:
            continue

        x1 = int(round(min(pts[0][0], pts[1][0])))
        y1 = int(round(min(pts[0][1], pts[1][1])))
        x2 = int(round(max(pts[0][0], pts[1][0])))
        y2 = int(round(max(pts[0][1], pts[1][1])))

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if "plate" in lbl:
            color = (255, 220, 0) # Cyan/Yellow for plate
            tag = "Plate"
        else:
            color = (0, 255, 0)   # Green for face
            tag = "Face"

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label badge
        font_scale = 0.40
        thickness = 1
        (lw, lh), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        lbl_y1 = max(0, y1 - lh - 4)
        lbl_y2 = y1
        lbl_x1 = x1
        lbl_x2 = min(w - 1, x1 + lw + 6)

        cv2.rectangle(canvas, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
        cv2.putText(canvas, tag, (lbl_x1 + 3, lbl_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return canvas


def build_fused_candidates_per_frame(
    selected_frames: List[str],
    models_dets: Dict[str, Dict[str, List[Dict[str, Any]]]],
    img_shapes_dict: Dict[str, Tuple[int, int]],
    iou_thresh: float = 0.35,
    iomin_thresh: float = 0.50
) -> Dict[str, Tuple[List[List[float]], List[List[float]]]]:
    """Computes strictly deduplicated fused boxes for each selected frame."""
    fused_per_frame = {}
    for frame_name in selected_frames:
        h, w = img_shapes_dict.get(frame_name, (1024, 1040))
        face_boxes, plate_boxes = [], []

        for p_key, frame_dict in models_dets.items():
            for d in frame_dict.get(frame_name, []):
                cls_name = str(d.get("class", "face")).lower().strip()
                b = d.get("box", [])
                if len(b) != 4:
                    continue

                x1 = max(0.0, min(float(w - 1), float(b[0])))
                y1 = max(0.0, min(float(h - 1), float(b[1])))
                x2 = max(0.0, min(float(w - 1), float(b[2])))
                y2 = max(0.0, min(float(h - 1), float(b[3])))

                if x2 <= x1 or y2 <= y1:
                    continue

                if "plate" in cls_name or "license" in cls_name:
                    plate_boxes.append([x1, y1, x2, y2])
                else:
                    face_boxes.append([x1, y1, x2, y2])

        fused_faces = fuse_boxes_strict_non_dup(face_boxes, iou_thresh=iou_thresh, iomin_thresh=iomin_thresh)
        fused_plates = fuse_boxes_strict_non_dup(plate_boxes, iou_thresh=iou_thresh, iomin_thresh=iomin_thresh)
        fused_per_frame[frame_name] = (fused_faces, fused_plates)

    return fused_per_frame


def generate_v1(
    output_dir: Path,
    selected_frames: List[str],
    fused_data: Dict[str, Tuple[List[List[float]], List[List[float]]]],
    source_img_paths: Dict[str, Path]
):
    """
    V.1 Generator:
      - Burns non-duplicated fused objects directly on each frame canvas.
      - Saves updated single master annotations.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.glob("*"):
        if f.is_file():
            f.unlink()

    coco_dataset = {
        "info": {
            "description": "Multi-Model Fused Ground Truth Annotations (V.1 Strict Non-Duplication)",
            "models_fused": ["FAST_V5", "EgoBlur", "DashcamCleaner", "FusionPortable", "Generative_Anonymizer"]
        },
        "categories": [
            {"id": 1, "name": "face", "supercategory": "privacy"},
            {"id": 2, "name": "plate", "supercategory": "privacy"}
        ],
        "images": [],
        "annotations": [],
        "frames": {}
    }

    ann_id = 1
    total_faces, total_plates = 0, 0

    for idx, frame_name in enumerate(selected_frames, 1):
        src_path = source_img_paths[frame_name]
        img = cv2.imread(str(src_path))
        h, w = img.shape[:2]

        fused_faces, fused_plates = fused_data[frame_name]
        total_faces += len(fused_faces)
        total_plates += len(fused_plates)

        shapes = []
        coco_dataset["images"].append({
            "id": idx,
            "file_name": frame_name,
            "width": int(w),
            "height": int(h)
        })

        frame_box_list = []
        for fb in fused_faces:
            bw = round(float(fb[2] - fb[0]), 1)
            bh = round(float(fb[3] - fb[1]), 1)
            shapes.append({
                "label": "face",
                "points": [[float(fb[0]), float(fb[1])], [float(fb[2]), float(fb[3])]],
                "shape_type": "rectangle"
            })
            coco_dataset["annotations"].append({
                "id": ann_id,
                "image_id": idx,
                "file_name": frame_name,
                "category_id": 1,
                "category_name": "face",
                "bbox": [float(fb[0]), float(fb[1]), bw, bh],
                "area": round(float(bw * bh), 1),
                "iscrowd": 0
            })
            frame_box_list.append({
                "box_id": ann_id,
                "label": "face",
                "box": [float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])],
                "points": [[float(fb[0]), float(fb[1])], [float(fb[2]), float(fb[3])]]
            })
            ann_id += 1

        for pb in fused_plates:
            bw = round(float(pb[2] - pb[0]), 1)
            bh = round(float(pb[3] - pb[1]), 1)
            shapes.append({
                "label": "plate",
                "points": [[float(pb[0]), float(pb[1])], [float(pb[2]), float(pb[3])]],
                "shape_type": "rectangle"
            })
            coco_dataset["annotations"].append({
                "id": ann_id,
                "image_id": idx,
                "file_name": frame_name,
                "category_id": 2,
                "category_name": "plate",
                "bbox": [float(pb[0]), float(pb[1]), bw, bh],
                "area": round(float(bw * bh), 1),
                "iscrowd": 0
            })
            frame_box_list.append({
                "box_id": ann_id,
                "label": "plate",
                "box": [float(pb[0]), float(pb[1]), float(pb[2]), float(pb[3])],
                "points": [[float(pb[0]), float(pb[1])], [float(pb[2]), float(pb[3])]]
            })
            ann_id += 1

        coco_dataset["frames"][frame_name] = frame_box_list

        # Burn strictly non-duplicated detected boxes onto frame pixels
        burned_img = draw_detected_objects_on_frame(img, shapes)
        cv2.imwrite(str(output_dir / frame_name), burned_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save single master annotations.json
    with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco_dataset, f, indent=2)

    print(f"[+] V.1 generated in {output_dir}: 10 burned frames + single annotations.json ({total_faces} faces, {total_plates} plates)")


def generate_v2(
    output_dir: Path,
    selected_frames: List[str],
    fused_data: Dict[str, Tuple[List[List[float]], List[List[float]]]],
    source_img_paths: Dict[str, Path]
):
    """
    V.2 Generator (Native X-AnyLabeling / LabelMe mode):
      - Clean images (NO burned pixels).
      - Generates separate matching <frame_stem>.json for each frame in LabelMe format.
      - Zero duplicate boxes in the same place.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.glob("*"):
        if f.is_file():
            f.unlink()

    total_faces, total_plates = 0, 0

    for idx, frame_name in enumerate(selected_frames, 1):
        src_path = source_img_paths[frame_name]
        dest_img_path = output_dir / frame_name

        # Copy clean frame (zero burned pixels)
        shutil.copy2(src_path, dest_img_path)

        img = cv2.imread(str(src_path))
        h, w = img.shape[:2]

        fused_faces, fused_plates = fused_data[frame_name]
        total_faces += len(fused_faces)
        total_plates += len(fused_plates)

        shapes = []

        # Add faces
        for fb in fused_faces:
            shapes.append({
                "label": "face",
                "points": [
                    [round(float(fb[0]), 1), round(float(fb[1]), 1)],
                    [round(float(fb[2]), 1), round(float(fb[3]), 1)]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })

        # Add plates
        for pb in fused_plates:
            shapes.append({
                "label": "plate",
                "points": [
                    [round(float(pb[0]), 1), round(float(pb[1]), 1)],
                    [round(float(pb[2]), 1), round(float(pb[3]), 1)]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })

        # LabelMe format JSON
        labelme_data = {
            "version": "2.4.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": frame_name,
            "imageData": None,
            "imageHeight": int(h),
            "imageWidth": int(w)
        }

        json_dest_path = output_dir / (Path(frame_name).stem + ".json")
        with open(json_dest_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, indent=2)

    print(f"[+] V.2 generated in {output_dir}: 10 clean frames + 10 individual LabelMe JSONs ({total_faces} faces, {total_plates} plates)")


def main():
    parser = argparse.ArgumentParser(description="Prepare frames for X-AnyLabeling review supporting ONLY V.1 and V.2.")
    parser.add_argument("--input-dir", type=str, default="", help="Optional path to source images.")
    parser.add_argument("--results-dir", type=str, default="3_Systems/outputs/results_pendik_100", help="Directory where models' outputs and detections.json reside.")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to sample for review (default: 10).")
    parser.add_argument("--iou-thresh", type=float, default=0.35, help="IoU threshold for non-duplication fusion (default: 0.35).")
    parser.add_argument("--iomin-thresh", type=float, default=0.50, help="IoMin threshold for containment non-duplication (default: 0.50).")
    parser.add_argument("--demo", action="store_true", help="Run consensus fusion on demo_dataset (5 frames).")
    args = parser.parse_args()

    if args.demo:
        base_dir = Path(__file__).resolve().parent.parent
        demo_dir = base_dir / "demo_dataset"
        results_dir = demo_dir / "raw_detections"
        input_dir = demo_dir / "images"
        outputs_base = demo_dir / "fused_candidates"
        args.num_frames = 5
        print("[*] Running in --demo mode on reproducible 5-frame demo_dataset.")
    else:
        results_dir = Path(args.results_dir).resolve()
        input_dir = Path(args.input_dir).resolve() if args.input_dir else None
        outputs_base = results_dir.parent

    print("=" * 80)
    print("  X-ANYLABELING GROUND TRUTH PREPARATION PIPELINE (ONLY V.1 & V.2)")
    print("=" * 80)
    print(f"[*] Results Directory : {results_dir}")
    print(f"[*] Num Frames Target : {args.num_frames}")
    print(f"[*] IoU Fusion Thresh : {args.iou_thresh} (IoMin Thresh: {args.iomin_thresh})")

    # 1. Frame Selection
    selected_frames = select_frames(input_dir, results_dir, args.num_frames)
    print(f"[+] Selected {len(selected_frames)} frames:")
    for idx, fn in enumerate(selected_frames, 1):
        print(f"    {idx:02d}. {fn}")

    # 2. Source image location
    blurred_base = results_dir / "blurred" if (results_dir / "blurred").exists() else results_dir
    search_keys = ["2_EgoBlur", "1_FAST_V5", "3_DashcamCleaner", "4_FusionPortable", "5_Generative_Anonymizer"]

    source_img_paths = {}
    img_shapes = {}

    for fn in selected_frames:
        src = None
        if input_dir and (input_dir / fn).exists():
            src = input_dir / fn
        else:
            for k in search_keys:
                cand = blurred_base / k / fn
                if cand.exists():
                    src = cand
                    break
        if not src or not src.exists():
            raise FileNotFoundError(f"Could not locate image file {fn}")
        source_img_paths[fn] = src
        img = cv2.imread(str(src))
        img_shapes[fn] = (img.shape[0], img.shape[1])

    # 3. Load detections & perform strict non-duplication fusion
    models_dets = load_model_detections(results_dir)
    fused_data = build_fused_candidates_per_frame(
        selected_frames,
        models_dets,
        img_shapes,
        iou_thresh=args.iou_thresh,
        iomin_thresh=args.iomin_thresh
    )

    # 4. Clean up any stale directories so ONLY V.1 and V.2 remain
    for old_dir_name in ["xanylabeling_review", "xanylabeling_review_v1", "xanylabeling_review_v2", "v.2"]:
        old_path = outputs_base / old_dir_name
        if old_path.is_symlink() or old_path.is_file():
            try:
                old_path.unlink()
            except Exception:
                pass
        elif old_path.is_dir() and old_dir_name not in ["V.1", "V.2"]:
            try:
                shutil.rmtree(old_path)
            except Exception:
                pass

    # Ensure V.1 and V.2 are real directories (remove symlinks if any)
    v1_dir = outputs_base / "V.1"
    if v1_dir.is_symlink():
        v1_dir.unlink()
    v1_dir.mkdir(parents=True, exist_ok=True)

    v2_dir = outputs_base / "V.2"
    if v2_dir.is_symlink():
        v2_dir.unlink()
    v2_dir.mkdir(parents=True, exist_ok=True)

    # 5. Generate V.1 and V.2
    generate_v1(v1_dir, selected_frames, fused_data, source_img_paths)
    generate_v2(v2_dir, selected_frames, fused_data, source_img_paths)

    print("=" * 80)
    print("  COMPLETED: ONLY V.1 AND V.2 CREATED WITH ZERO DUPLICATIONS")
    print("=" * 80)


if __name__ == "__main__":
    main()
