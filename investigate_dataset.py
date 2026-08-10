import json
import os
import statistics

# Load annotations
anno_path = r"c:\Users\prajn\OneDrive\Desktop\iitp\news.coco\train\_annotations.coco.json"
with open(anno_path) as f:
    coco = json.load(f)

images = coco.get("images", [])
annotations = coco.get("annotations", [])
img_dict = {img["id"]: img for img in images}

# 1. PROVENANCE
# The extraction script named files "frame_0000.jpg" incrementally across ALL videos.
# So "frame_0007" means the 7th extracted frame overall, but we don't inherently know which video it belongs to
# UNLESS we reconstruct it by looking at the video durations or the original extract_frames.py logs.

# Let's check the original extraction logic:
# `for video_file in VIDEO_DIR.glob("*.mp4"):`
#    `cap.read()` ... `filename = FRAME_DIR / f"frame_{frame_count:04d}.jpg"`
# It increments a global `frame_count`. So frame IDs are sequential across the entire batch of videos.
# Since we can't easily map frame ID to video without re-running or looking at logs, it's mixed.

# Let's inspect the 5 invalid boxes
invalid_boxes_report = []
for ann in annotations:
    img = img_dict[ann["image_id"]]
    x, y, w, h = ann["bbox"]
    iw, ih = img["width"], img["height"]
    
    exceeds = []
    if x < 0: exceeds.append(("left", abs(x)))
    if y < 0: exceeds.append(("top", abs(y)))
    if x + w > iw: exceeds.append(("right", (x + w) - iw))
    if y + h > ih: exceeds.append(("bottom", (y + h) - ih))
    
    if exceeds:
        invalid_boxes_report.append({
            "image_filename": img["file_name"],
            "annotation_id": ann["id"],
            "bbox": ann["bbox"],
            "img_dims": (iw, ih),
            "exceeds": exceeds
        })

print("INVALID BOXES:")
for b in invalid_boxes_report:
    print(b)
print("---")

# 3. DISTRIBUTION
from collections import defaultdict
ann_per_img = defaultdict(int)
for img in images:
    ann_per_img[img["id"]] = 0
for ann in annotations:
    ann_per_img[ann["image_id"]] += 1

counts = list(ann_per_img.values())
print("ANNOTATIONS PER IMAGE:")
print(f"Min: {min(counts)}")
print(f"Max: {max(counts)}")
print(f"Mean: {sum(counts)/len(counts):.2f}")
print(f"Median: {statistics.median(counts)}")

areas = []
aspect_ratios = []

small_count = 0
medium_count = 0
large_count = 0

wide_count = 0
tall_count = 0

for ann in annotations:
    x, y, w, h = ann["bbox"]
    area = w * h
    areas.append(area)
    aspect_ratios.append(w / h if h > 0 else 0)
    
    # Relative area
    img = img_dict[ann["image_id"]]
    rel_area = area / (img["width"] * img["height"])
    
    # COCO standard: small < 32^2, medium 32^2-96^2, large > 96^2
    if area < 1024:
        small_count += 1
    elif area < 9216:
        medium_count += 1
    else:
        large_count += 1
        
    # Aspect ratio rules (heuristic)
    # horizontal: aspect ratio > 2
    # vertical: aspect ratio < 0.5
    ar = w / h if h > 0 else 1
    if ar > 3:
        wide_count += 1
    elif ar < 0.33:
        tall_count += 1

print("\nBOX AREA:")
print(f"Min: {min(areas):.2f}, Max: {max(areas):.2f}, Mean: {sum(areas)/len(areas):.2f}")
print(f"Small (<32^2): {small_count} ({small_count/len(annotations)*100:.1f}%)")
print(f"Medium (32^2-96^2): {medium_count} ({medium_count/len(annotations)*100:.1f}%)")
print(f"Large (>96^2): {large_count} ({large_count/len(annotations)*100:.1f}%)")

print("\nASPECT RATIO:")
print(f"Min AR: {min(aspect_ratios):.2f}, Max AR: {max(aspect_ratios):.2f}")
print(f"Very wide (AR > 3): {wide_count} ({wide_count/len(annotations)*100:.1f}%)")
print(f"Tall (AR < 0.33): {tall_count} ({tall_count/len(annotations)*100:.1f}%)")
