# Broadcast Media Analysis: Rectangular Text Overlay Detection

## Overview
This project trains a custom computer vision model to detect **important text overlays** in broadcast news frames. While often referred to as "lower-thirds," the model goes beyond just the bottom banners; it is designed to capture *every* critical piece of information presented as text on the screen. These informational graphics—names, titles, tickers, and other rectangular text boxes—are bounded and recognized by the model. Detecting these overlays is a critical first step in automated broadcast media analysis and text extraction.

## Model Details
We custom trained and fine-tuned an **SSDLite320 with a MobileNetV3-Large backbone**. This architecture was chosen because it provides an extremely efficient, lightweight object detection solution that can run in real-time environments on CPU without requiring heavy GPU compute.

### Architecture & Computational Complexity
- **Base Architecture**: SSDLite320
- **Backbone**: MobileNetV3-Large
- **Parameters**: 2.21M (2,210,000)
- **Computation**: 0.427 GFLOPs
- **Transfer Learning**: Fine-tuned from pretrained COCO weights

The model footprint is extremely low, allowing fast inference while still maintaining adequate accuracy for bounding box prediction on structured broadcast graphical overlays.

## Training Approaches
We evaluated the model using two distinct fine-tuning strategies:

### 1. Model A — Baseline
**Script**: `train_baseline.py`
The baseline approach fine-tunes the pretrained `ssdlite320_mobilenet_v3_large` model using standard COCO default anchor box configurations. It acts as the control to see how well out-of-the-box object detection handles broadcast overlays.

### 2. Model B — Dataset-Adapted
**Script**: `train_dataset_adapted.py`
This approach modifies the base model by introducing **custom wide anchors**. Because text boxes and lower-thirds in news broadcasts are almost always extremely wide and short rectangles (spanning large portions of the screen width), replacing the default anchor generators with custom aspect ratios specific to this domain significantly improves the intersection-over-union (IoU) matching during training. This allows the model to converge faster and predict bounding boxes that better fit the text overlays.

## Dataset Information
The dataset used for training is sourced from a Roboflow export.
- **Dataset Name**: `news-ffe2h`
- **Total Images**: 259
- **Total Annotations**: ~500+ text overlays
- **Number of Classes**: 1 (`lower-third` / `text-overlay`)
- **Format**: COCO / YOLO format

## Project Structure & Scripts
- **`train_baseline.py`**: Trains the baseline Model A. Checkpoints are saved under `runs/ssdlite_mobilenetv3/baseline/`.
- **`train_dataset_adapted.py`**: Trains the adapted Model B with custom anchor generation. Checkpoints are saved under `runs/ssdlite_mobilenetv3/dataset_adapted/`.
- **`process_and_inspect.py` / `investigate_dataset.py`**: Dataset investigation and preprocessing tools.
- **`split_and_measure.py`**: Script to measure and verify model complexity (FLOPs/Params).
- **`app/`**: Contains the code for a web-based demo application, split into a `backend/` (`app.py` - FastAPI based) and `frontend/` (`streamlit_app.py` - Streamlit UI).

## Performance Measures
To evaluate the effectiveness of the models in detecting text overlays, we track several performance metrics. The most critical metric for our use-case is **Intersection over Union (IoU)**.

- **Intersection over Union (IoU)**: This measures the spatial overlap between the model's predicted bounding box and the ground-truth text overlay box. An IoU of 1.0 means perfect alignment. Since broadcast text overlays are strict rectangles containing vital text, achieving high IoU ensures that the entire text is bounded accurately without cropping off words or including too much background noise.
- **Mean Average Precision (mAP)**: Evaluated at various IoU thresholds (e.g., mAP@0.5, mAP@0.5:0.95), this gives a comprehensive view of how well the model localizes overlays at different strictness levels.

### Evaluation Results (Test Set)
We have a custom script `calculate_iou.py` to evaluate the exact Mean IoU (mIoU) on our test split for correctly identified text overlays:
- **Baseline (Model A)**: 0.7477 Mean IoU
- **Dataset-Adapted (Model B)**: 0.6899 Mean IoU

## Limitations & Future Work
**Current Data Limitations:** 
Right now, the dataset is quite small, containing approximately 250 images and just over 500 bounding box overlays. While the model is working and recognizing the rectangular boxes, its robustness is limited by the lack of diverse training examples.

**Future Improvements:**
In the future, we plan to collect and annotate significantly more data from various broadcast sources. With a larger, more diverse dataset, the model will become much more proper and generalized, improving detection accuracy across different broadcasting styles and screen layouts.

## Getting Started

1. **Install Dependencies**:
   Ensure you have PyTorch and Torchvision installed.
   ```bash
   pip install torch torchvision
   ```

2. **Train the Models**:
   To train the baseline:
   ```bash
   python train_baseline.py
   ```
   To train the dataset-adapted version:
   ```bash
   python train_dataset_adapted.py
   ```

3. **Launch the Demo Web App**:
   You can run the full frontend and backend via the application files located in `app/`.
