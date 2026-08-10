import json
import os
import shutil
import re
import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from fvcore.nn import FlopCountAnalysis
import tempfile

processed_dir = r"c:\Users\prajn\OneDrive\Desktop\iitp\data\coco_processed"
train_src_dir = os.path.join(processed_dir, "train")
anno_file = os.path.join(train_src_dir, "_annotations.coco.json")

# 1. VERIFY BOUNDARY CLAMPING
with open(anno_file) as f:
    coco = json.load(f)

img_dict = {img["id"]: img for img in coco["images"]}
invalid_boxes = 0
for ann in coco["annotations"]:
    x, y, w, h = ann["bbox"]
    img = img_dict[ann["image_id"]]
    iw, ih = img["width"], img["height"]
    if x < 0 or y < 0 or x+w > iw or y+h > ih:
        invalid_boxes += 1
print(f"Verified clamped boxes. Remaining out-of-bounds: {invalid_boxes}")

# 2. CREATE TEMPORAL-GROUPED SPLITS
frame_numbers = []
for img in coco["images"]:
    match = re.search(r"frame_(\d+)", img["file_name"])
    if match:
        frame_numbers.append((int(match.group(1)), img))

frame_numbers.sort(key=lambda x: x[0])
groups = []
if frame_numbers:
    current_group = [frame_numbers[0]]
    for i in range(1, len(frame_numbers)):
        prev_num = frame_numbers[i-1][0]
        curr_num = frame_numbers[i][0]
        if curr_num - prev_num > 5:
            groups.append(current_group)
            current_group = [frame_numbers[i]]
        else:
            current_group.append(frame_numbers[i])
    groups.append(current_group)

# Splits: Train 1-29, Val 30-34, Test 35-41
splits = {"train": [], "val": [], "test": []}
for idx, g in enumerate(groups):
    imgs = [item[1] for item in g]
    if idx < 29: splits["train"].extend(imgs)
    elif idx < 34: splits["val"].extend(imgs)
    else: splits["test"].extend(imgs)

print(f"Proposed split verification - No groups cross boundaries by design.")
print(f"Train: {len(splits['train'])} imgs, Val: {len(splits['val'])} imgs, Test: {len(splits['test'])} imgs")

ann_by_img = {}
for ann in coco["annotations"]:
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

for split_name, imgs in splits.items():
    split_dir = os.path.join(processed_dir, f"split_{split_name}")
    os.makedirs(split_dir, exist_ok=True)
    
    split_json = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": imgs,
        "annotations": []
    }
    for img in imgs:
        split_json["annotations"].extend(ann_by_img.get(img["id"], []))
        src_img = os.path.join(train_src_dir, img["file_name"])
        dst_img = os.path.join(split_dir, img["file_name"])
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
    with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
        json.dump(split_json, f)


# 3. INSTANTIATE & MEASURE MODELS
def measure_model(name, model, dummy_input):
    model.eval()
    
    # 1. Forward pass
    with torch.no_grad():
        out = model(dummy_input)
    
    # 2. Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 3. GFLOPs
    flops = FlopCountAnalysis(model, dummy_input)
    gflops = flops.total() / 1e9
    
    # 4. Model size
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(model.state_dict(), f.name)
        f.seek(0, os.SEEK_END)
        size_mb = f.tell() / (1024 * 1024)
    os.remove(f.name)
    
    ag = model.anchor_generator
    num_anchors = ag.num_anchors_per_location()
    
    print(f"\n--- MODEL: {name} ---")
    print(f"Total Params: {total_params / 1e6:.2f} M")
    print(f"Trainable Params: {trainable_params / 1e6:.2f} M")
    print(f"GFLOPs: {gflops:.3f}")
    print(f"Model Size: {size_mb:.2f} MB")
    print(f"Anchors per location: {num_anchors}")
    print(f"Output keys: {list(out[0].keys())}")

dummy_x = [torch.rand(3, 320, 320)]
print("\nCreating BASELINE...")
model_base = ssdlite320_mobilenet_v3_large(num_classes=2, weights=None, weights_backbone=None)
measure_model("BASELINE", model_base, dummy_x)

print("\nCreating DATASET-ADAPTED...")
anchor_generator = DefaultBoxGenerator(
    [[2, 3, 5, 8]] * 6,
    min_ratio=20, 
    max_ratio=95
)
model_adapted = ssdlite320_mobilenet_v3_large(num_classes=2, weights=None, weights_backbone=None)
in_channels = [m[0][0].in_channels for m in model_adapted.head.classification_head.module_list]

from torchvision.models.detection.ssdlite import SSDLiteHead
import torch.nn as nn
model_adapted.anchor_generator = anchor_generator
model_adapted.head = SSDLiteHead(in_channels, anchor_generator.num_anchors_per_location(), 2, norm_layer=nn.BatchNorm2d)
measure_model("DATASET-ADAPTED", model_adapted, dummy_x)
