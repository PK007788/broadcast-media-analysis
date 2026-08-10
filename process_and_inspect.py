import json
import os
import shutil
import re

original_dir = r"c:\Users\prajn\OneDrive\Desktop\iitp\news.coco"
processed_dir = r"c:\Users\prajn\OneDrive\Desktop\iitp\data\coco_processed"
original_copy_dir = r"c:\Users\prajn\OneDrive\Desktop\iitp\data\coco_original"

# Phase 1: Create processed dataset
os.makedirs(os.path.join(processed_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(original_copy_dir, "train"), exist_ok=True)

# Copy images to processed
for root, dirs, files in os.walk(os.path.join(original_dir, "train")):
    for f in files:
        if f.endswith(".jpg"):
            shutil.copy(os.path.join(root, f), os.path.join(processed_dir, "train", f))

# Process JSON
with open(os.path.join(original_dir, "train", "_annotations.coco.json")) as f:
    coco = json.load(f)

img_dict = {img["id"]: img for img in coco["images"]}

negative_coords = 0
exceeds_w = 0
exceeds_h = 0
zero_w_h = 0

for ann in coco["annotations"]:
    img = img_dict[ann["image_id"]]
    iw, ih = img["width"], img["height"]
    
    x, y, w, h = ann["bbox"]
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # Clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(iw, x2)
    y2 = min(ih, y2)
    
    w_new = x2 - x1
    h_new = y2 - y1
    
    ann["bbox"] = [x1, y1, w_new, h_new]
    
    # Verification
    if x1 < 0 or y1 < 0: negative_coords += 1
    if x2 > iw: exceeds_w += 1
    if y2 > ih: exceeds_h += 1
    if w_new <= 0 or h_new <= 0: zero_w_h += 1

with open(os.path.join(processed_dir, "train", "_annotations.coco.json"), "w") as f:
    json.dump(coco, f)

print("PHASE 1 VERIFICATION:")
print(f"Negative coords: {negative_coords}")
print(f"Exceeds width: {exceeds_w}")
print(f"Exceeds height: {exceeds_h}")
print(f"Zero width/height: {zero_w_h}")
print("---")

# Phase 2: Frame gaps
frame_numbers = []
for img in coco["images"]:
    # filename e.g. frame_0007_jpg.rf...
    match = re.search(r"frame_(\d+)", img["file_name"])
    if match:
        frame_numbers.append((int(match.group(1)), img["file_name"]))
    
frame_numbers.sort(key=lambda x: x[0])

groups = []
if frame_numbers:
    current_group = [frame_numbers[0]]
    for i in range(1, len(frame_numbers)):
        prev_num = frame_numbers[i-1][0]
        curr_num = frame_numbers[i][0]
        
        # If the gap is suspiciously large (e.g. > 20), it's probably a new video
        # Let's see what the gaps actually are
        if curr_num - prev_num > 5:
            groups.append(current_group)
            current_group = [frame_numbers[i]]
        else:
            current_group.append(frame_numbers[i])
    groups.append(current_group)

print("PHASE 2 GROUPS:")
for idx, g in enumerate(groups):
    print(f"Group {idx+1}: First {g[0][0]:04d}, Last {g[-1][0]:04d}, Count {len(g)}")
print("---")

# Phase 4: SSDLite Anchor Config
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

model = ssdlite320_mobilenet_v3_large(pretrained=False)
ag = model.anchor_generator

print("PHASE 4 ANCHOR GEN:")
print("Number of feature maps:", len(ag.aspect_ratios))
for i, (scales, ratios) in enumerate(zip(ag.scales, ag.aspect_ratios)):
    print(f"Feature map {i+1}:")
    print(f"  Scales: {scales}")
    print(f"  Aspect ratios (default): {ratios}")
    # Number of anchors per location = len(scales) * len(ratios) roughly,
    # actually ssd anchors are 2 + 2 * len(extra_ratios) or something, let's just print len of returned anchors later if needed
