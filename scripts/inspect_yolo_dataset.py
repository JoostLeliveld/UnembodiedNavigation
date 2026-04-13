#!/usr/bin/env python3
"""
Interactive dataset inspector tool.
Explore training data, labels, and visualizations.

Usage:
  python inspect_yolo_dataset.py <dataset_dir>
  python inspect_yolo_dataset.py logs/yolo_seg_datasets/yolo_dataset_20260410_...
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str):
    """Print formatted section."""
    print(f"\n▶ {text}")
    print("-" * 70)


def inspect_config(dataset_dir: Path):
    """Inspect data.yaml configuration."""
    print_section("Configuration (data.yaml)")
    
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        print("❌ data.yaml not found")
        return
    
    config = yaml.safe_load(data_yaml.read_text())
    
    print(f"Task: {config.get('task', 'unknown')}")
    print(f"Classes: {config.get('nc', 0)}")
    print(f"Class names: {config.get('names', {})}")
    
    source_root = Path(config.get("path", dataset_dir))
    for split in ("train", "val", "test"):
        images_dir = source_root / "images" / split
        if images_dir.exists():
            count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
            print(f"{split.upper():6} images: {count:3}")


def inspect_labels(dataset_dir: Path):
    """Analyze label statistics."""
    print_section("Label Statistics")
    
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        print("❌ data.yaml not found")
        return
    
    config = yaml.safe_load(data_yaml.read_text())
    source_root = Path(config.get("path", dataset_dir))
    
    for split in ("train", "val"):
        labels_dir = source_root / "labels" / split
        if not labels_dir.exists():
            continue
        
        print(f"\n{split.upper()} Labels:")
        
        boxes_per_image = []
        total_coords = []
        
        for label_file in sorted(labels_dir.glob("*.txt")):
            lines = [l.strip() for l in label_file.read_text().split("\n") if l.strip()]
            boxes_per_image.append(len(lines))
            
            for line in lines:
                values = line.split()
                if values:
                    total_coords.append(len(values))
        
        if boxes_per_image:
            print(f"  Total annotations: {sum(boxes_per_image)}")
            print(f"  Boxes per image: min={min(boxes_per_image)}, max={max(boxes_per_image)}, "
                  f"avg={np.mean(boxes_per_image):.2f}")
            print(f"  Empty labels: {sum(1 for n in boxes_per_image if n == 0)}/{len(boxes_per_image)}")
            if total_coords:
                print(f"  Coords per annotation: min={min(total_coords)}, max={max(total_coords)}, "
                      f"avg={np.mean(total_coords):.1f}")
            else:
                print("  Coords per annotation: none (all labels empty)")


def show_label_examples(dataset_dir: Path, count: int = 3):
    """Show example labels."""
    print_section(f"Label Examples (first {count})")
    
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return
    
    config = yaml.safe_load(data_yaml.read_text())
    source_root = Path(config.get("path", dataset_dir))
    labels_dir = source_root / "labels" / "train"
    
    if not labels_dir.exists():
        print("❌ No train labels")
        return
    
    for i, label_file in enumerate(sorted(labels_dir.glob("*.txt"))[:count]):
        print(f"\n{label_file.name}:")
        content = label_file.read_text()
        if not content.strip():
            print("  (empty)")
        else:
            for line in content.split("\n")[:5]:
                if line.strip():
                    parts = line.split()
                    if len(parts) > 5:
                        print(f"  Class {parts[0]}: {len(parts)-1} coordinates")
                        print(f"    {' '.join(parts[:6])}...")
                    else:
                        print(f"  {line}")


def show_conversion_manifest(dataset_dir: Path):
    """Show SAM conversion details (if available)."""
    manifest_file = dataset_dir / "conversion_manifest.json"
    if not manifest_file.exists():
        return
    
    print_section("SAM Conversion Manifest")
    
    manifest = json.loads(manifest_file.read_text())

    if "label_model" in manifest:
        print(f"Method: {manifest['label_model']['method']}")
        print(f"SAM Model: {manifest['label_model']['sam_model']}")
        print(f"Device: {manifest['label_model']['device']}")
        print(f"\nResults:")
        print(f"  Total images: {manifest['n_images']}")
        print(f"  Images with prompts: {manifest['n_prompted_images']}")
        print(f"  Total prompts: {manifest['n_prompts']}")
        print(f"  Total masks generated: {manifest['n_masks']}")
        if manifest["n_prompted_images"] > 0:
            avg_masks = manifest["n_masks"] / manifest["n_prompted_images"]
            print(f"  Avg masks per image: {avg_masks:.2f}")
        return

    print(f"Method: {manifest.get('method', 'unknown')}")
    print(f"SAM Model: {manifest.get('sam_model', 'unknown')}")
    print(f"Description: {manifest.get('description', '')}")
    settings = manifest.get('settings', {})
    if settings:
        print("\nSettings:")
        for key in ('prompt_mode', 'prompt_modes_tried', 'prompt_pad_px', 'min_mask_area_px', 'min_bbox_overlap_ratio'):
            if key in settings:
                print(f"  {key}: {settings[key]}")
    results = manifest.get('results', {})
    if results:
        print("\nResults:")
        for key in ('total_images', 'total_prompts', 'total_masks_accepted', 'total_masks_rejected', 'acceptance_rate', 'prompt_used_counts'):
            if key in results:
                print(f"  {key}: {results[key]}")


def show_capture_manifest(dataset_dir: Path):
    """Show capture details (if available)."""
    manifest_file = dataset_dir / "capture_manifest.json"
    if not manifest_file.exists():
        return
    
    print_section("Capture Manifest")
    
    manifest = json.loads(manifest_file.read_text())
    
    print(f"World: {manifest.get('world_name', 'unknown')}")
    print(f"Total images: {manifest.get('n_images', manifest.get('n_frames', 0))}")
    print(f"Sampled positions: {manifest.get('n_sampled_positions', manifest.get('n_frames', 0))}")
    
    if "robot_dimensions" in manifest:
        dims = manifest["robot_dimensions"]
        print(f"Robot size: {dims.get('half_x', 0)*2:.3f}m × {dims.get('half_y', 0)*2:.3f}m "
              f"× {dims.get('height', 0):.3f}m")
    elif "label_model" in manifest:
        dims = manifest["label_model"]
        print(f"Robot size: {dims.get('robot_half_x', 0)*2:.3f}m × {dims.get('robot_half_y', 0)*2:.3f}m "
              f"× {dims.get('robot_height_m', 0):.3f}m")
    
    if "camera_config" in manifest:
        cam = manifest["camera_config"]
        print(f"Camera: resolution {cam.get('img_width', 0)}×{cam.get('img_height', 0)}, "
              f"FOV {cam.get('fov_h_deg', 0):.1f}°")
    elif "camera" in manifest:
        cam = manifest["camera"]
        print(f"Camera: resolution {cam.get('img_width', 0)}×{cam.get('img_height', 0)}, "
              f"fov_h_rad {cam.get('fov_h_rad', 0):.4f}")


def list_images(dataset_dir: Path):
    """List available images."""
    print_section("Available Images")
    
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        print("❌ data.yaml not found")
        return
    
    config = yaml.safe_load(data_yaml.read_text())
    source_root = Path(config.get("path", dataset_dir))
    
    for split in ("train", "val", "test"):
        images_dir = source_root / "images" / split
        if not images_dir.exists():
            continue
        
        images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        if images:
            print(f"\n{split.upper()} ({len(images)} images):")
            for img in images[:10]:
                size_kb = img.stat().st_size / 1024
                print(f"  • {img.name} ({size_kb:.1f} KB)")
            if len(images) > 10:
                print(f"  ... and {len(images) - 10} more")


def show_preview(dataset_dir: Path, image_name: str = None):
    """Show preview image with overlay."""
    print_section("Preview Images")
    
    candidate_dirs = [dataset_dir / "previews" / "train", dataset_dir / "previews"]
    previews_dir = next((d for d in candidate_dirs if d.exists()), None)
    if previews_dir is None:
        print("❌ No previews directory")
        return
    
    previews = sorted(previews_dir.glob("*.jpg"))
    if not previews:
        print("❌ No preview images found")
        return
    
    if image_name:
        preview = previews_dir / image_name
    else:
        preview = previews[0]
    
    if not preview.exists():
        print(f"❌ Preview not found: {preview.name}")
        return
    
    print(f"Preview: {preview.name}")
    print(f"Path: {preview}")
    print(f"\nTo view: display {preview}")
    print(f"Or: eog {preview}  # alternative image viewer")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect YOLO dataset structure, labels, and metadata"
    )
    parser.add_argument("dataset", help="Path to dataset directory (containing data.yaml)")
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset).expanduser().resolve()
    
    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        return 1
    
    if not (dataset_dir / "data.yaml").exists():
        print(f"❌ Not a YOLO dataset (missing data.yaml): {dataset_dir}")
        return 1
    
    print_header(f"DATASET INSPECTOR: {dataset_dir.name}")
    
    # Run all inspections
    inspect_config(dataset_dir)
    inspect_labels(dataset_dir)
    show_label_examples(dataset_dir)
    show_conversion_manifest(dataset_dir)
    show_capture_manifest(dataset_dir)
    list_images(dataset_dir)
    show_preview(dataset_dir)
    
    print("\n" + "=" * 70)
    print("  Dataset inspection complete!")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
