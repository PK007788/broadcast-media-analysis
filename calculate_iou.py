import os
# pyrefly: ignore [missing-import]
import torch
import sys

# Ensure the local directory is in path for imports
sys.path.insert(0, r"c:\Users\prajn\OneDrive\Desktop\iitp")
from train_baseline import OverlayCOCODataset, DetectionTransform, compute_iou
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

def load_model(weights_path):
    # num_classes is 2 (background + overlay)
    model = ssdlite320_mobilenet_v3_large(num_classes=2, weights=None, weights_backbone=None)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def calculate_mean_iou(model, dataset, device, conf_threshold=0.3):
    model.to(device)
    model.eval()
    
    total_iou = 0.0
    matched_boxes = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            img, target = dataset[i]
            img = img.unsqueeze(0).to(device)
            gt_boxes = target["boxes"].to(device)
            
            if len(gt_boxes) == 0:
                continue
                
            preds = model(img)[0]
            pred_boxes = preds["boxes"]
            scores = preds["scores"]
            labels = preds["labels"]
            
            mask = (scores >= conf_threshold) & (labels == 1)
            pred_boxes = pred_boxes[mask]
            
            if len(pred_boxes) == 0:
                continue
                
            ious = compute_iou(pred_boxes, gt_boxes)
            
            matched_gt = set()
            for p_idx, pb in enumerate(pred_boxes):
                max_iou, max_gt = ious[p_idx].max(dim=0)
                if max_iou >= 0.5 and max_gt.item() not in matched_gt:
                    total_iou += max_iou.item()
                    matched_boxes += 1
                    matched_gt.add(max_gt.item())

    if matched_boxes == 0:
        return 0.0
    return total_iou / matched_boxes

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_dir = r"c:\Users\prajn\OneDrive\Desktop\iitp\data\coco_processed\split_test"
    dataset = OverlayCOCODataset(test_dir, transforms=DetectionTransform(train=False))
    
    baseline_weights = r"c:\Users\prajn\OneDrive\Desktop\iitp\runs\ssdlite_mobilenetv3\baseline\checkpoints\best_model.pt"
    adapted_weights = r"c:\Users\prajn\OneDrive\Desktop\iitp\runs\ssdlite_mobilenetv3\dataset_adapted\checkpoints\best_model.pt"
    
    print("Loading Baseline model...")
    baseline_model = load_model(baseline_weights)
    baseline_iou = calculate_mean_iou(baseline_model, dataset, device)
    
    print("Loading Dataset-Adapted model...")
    from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
    from torchvision.models.detection.ssdlite import SSDLiteHead
    import torch.nn as nn
    from functools import partial
    
    adapted_model = ssdlite320_mobilenet_v3_large(num_classes=2, weights=None, weights_backbone=None)
    in_channels = [c[0][0].in_channels for c in adapted_model.head.classification_head.module_list]
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    custom_anchor_gen = DefaultBoxGenerator(
        [[2, 3, 5, 8]] * 6,
        min_ratio=20,
        max_ratio=95
    )
    adapted_model.anchor_generator = custom_anchor_gen
    num_anchors = custom_anchor_gen.num_anchors_per_location()
    adapted_model.head = SSDLiteHead(in_channels, num_anchors, 2, norm_layer=norm_layer)
    
    adapted_ckpt = torch.load(adapted_weights, map_location="cpu", weights_only=True)
    if "model_state_dict" in adapted_ckpt:
        adapted_model.load_state_dict(adapted_ckpt["model_state_dict"])
    else:
        adapted_model.load_state_dict(adapted_ckpt)
    adapted_model.eval()
    adapted_iou = calculate_mean_iou(adapted_model, dataset, device)
    
    print("\n--- Results ---")
    print(f"Baseline Mean IoU: {baseline_iou:.4f}")
    print(f"Dataset-Adapted Mean IoU: {adapted_iou:.4f}")

