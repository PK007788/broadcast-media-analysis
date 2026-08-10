"""
Flask Backend — Broadcast Overlay Detection API
================================================
Loads the validated Model A (SSDLite320-MobileNetV3-Large) once at startup.
Provides endpoints for both image and video inference.
"""

import io
import os
import sys
import time
import uuid
import logging
import traceback
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteHead
from torchvision import transforms as T
from flask import Flask, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "runs", "ssdlite_mobilenetv3", "baseline", "checkpoints", "best_model.pt")
TEMP_DIR = os.path.join(BASE_DIR, "app", "temp")
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "webp"}
NUM_CLASSES = 2  # background + overlay
INPUT_SIZE = 320
MODEL_PARAMS = "2.21M"
MODEL_GFLOPS = 0.427
MODEL_SIZE_MB = 8.71

os.makedirs(TEMP_DIR, exist_ok=True)

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# MODEL LOADING (once at startup)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def load_model():
    """Load Model A checkpoint. Called once at startup."""
    global model

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    logger.info(f"Loading model on device: {device}")

    # Build the SAME architecture used during training:
    # 1) Load with COCO pretrained weights (sets correct backbone channels)
    # 2) Replace head for our 2-class problem
    model = ssdlite320_mobilenet_v3_large(
        weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
        num_classes=91
    )
    in_channels = [c[0][0].in_channels for c in model.head.classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head = SSDLiteHead(in_channels, num_anchors, NUM_CLASSES, norm_layer=norm_layer)

    # Load trained weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Verify with a dummy forward pass
    with torch.inference_mode():
        dummy = [torch.rand(3, INPUT_SIZE, INPUT_SIZE).to(device)]
        out = model(dummy)
        assert "boxes" in out[0], "Model output missing 'boxes'"

    logger.info(f"Model loaded successfully. Classes: {NUM_CLASSES}, Device: {device}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}, Val loss: {checkpoint.get('val_loss', 'N/A')}")
    return model


# ============================================================
# INFERENCE HELPERS
# ============================================================
def preprocess_frame(frame_bgr):
    """Convert an OpenCV BGR frame to a normalized tensor for the model."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (INPUT_SIZE, INPUT_SIZE))
    tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    for c in range(3):
        tensor[c] = (tensor[c] - MEAN[c]) / STD[c]
    return tensor


def detect_on_tensor(tensor, orig_w, orig_h, conf_threshold=0.4):
    """
    Run detection on a preprocessed tensor.
    Returns list of dicts with boxes in ORIGINAL image coordinates.
    """
    with torch.inference_mode():
        predictions = model([tensor.to(device)])

    pred = predictions[0]
    boxes = pred["boxes"].cpu()
    scores = pred["scores"].cpu()
    labels = pred["labels"].cpu()

    mask = (scores >= conf_threshold) & (labels == 1)
    boxes = boxes[mask]
    scores = scores[mask]

    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE

    detections = []
    for box, score in zip(boxes, scores):
        x1 = max(0, min(int(box[0].item() * scale_x), orig_w))
        y1 = max(0, min(int(box[1].item() * scale_y), orig_h))
        x2 = max(0, min(int(box[2].item() * scale_x), orig_w))
        y2 = max(0, min(int(box[3].item() * scale_y), orig_h))
        detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": round(score.item(), 3)})

    return detections


def draw_detections_cv2(frame_bgr, detections):
    """Draw bounding boxes and labels on a BGR frame (OpenCV)."""
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        score = det["score"]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"overlay {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return frame_bgr


def draw_detections_pil(image_pil, detections):
    """Draw bounding boxes and labels on a PIL Image."""
    draw = ImageDraw.Draw(image_pil)
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        score = det["score"]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        label = f"overlay {score:.2f}"
        # Label background
        bbox = draw.textbbox((x1, y1 - 16), label)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill="lime")
        draw.text((x1, y1 - 16), label, fill="black")
    return image_pil


# ============================================================
# VIDEO PROCESSING
# ============================================================
def process_video(input_path, output_path, conf_threshold=0.4):
    """Process a full video: detect overlays per frame and write annotated output."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video properties (fps/width/height)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Cannot create output video writer")

    frame_count = 0
    total_detections = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tensor = preprocess_frame(frame)
        detections = detect_on_tensor(tensor, width, height, conf_threshold)
        total_detections += len(detections)
        frame = draw_detections_cv2(frame, detections)
        writer.write(frame)
        frame_count += 1

    elapsed = time.time() - start_time
    cap.release()
    writer.release()

    return {
        "frames_processed": frame_count,
        "total_frames": total_frames,
        "total_detections": total_detections,
        "processing_time_s": round(elapsed, 2),
        "processing_fps": round(frame_count / elapsed, 1) if elapsed > 0 else 0,
        "input_resolution": f"{width}x{height}",
        "input_fps": round(fps, 1),
    }


# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "SSDLite320-MobileNetV3-Large",
        "model_class": "overlay",
        "parameters": MODEL_PARAMS,
        "gflops": MODEL_GFLOPS,
        "model_size_mb": MODEL_SIZE_MB,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    })


# ============================================================
# IMAGE ENDPOINT
# ============================================================
@app.route("/predict/image", methods=["POST"])
def predict_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided. Use key 'image'."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_IMAGE_EXT:
            return jsonify({"error": f"Unsupported image format. Allowed: {ALLOWED_IMAGE_EXT}"}), 400

        conf_threshold = float(request.form.get("confidence", 0.4))
        conf_threshold = max(0.05, min(0.95, conf_threshold))

        # Decode image
        img_bytes = file.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = img_pil.size

        # Convert to OpenCV for preprocessing
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Inference
        start_time = time.time()
        tensor = preprocess_frame(img_bgr)
        detections = detect_on_tensor(tensor, orig_w, orig_h, conf_threshold)
        elapsed = time.time() - start_time

        # Draw detections on PIL image
        annotated = draw_detections_pil(img_pil.copy(), detections)

        # Encode annotated image to PNG bytes
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        buf.seek(0)

        # Build metadata header
        import json
        metadata = {
            "detections": len(detections),
            "boxes": detections,
            "confidence_threshold": conf_threshold,
            "processing_time_ms": round(elapsed * 1000, 2),
            "image_size": f"{orig_w}x{orig_h}",
            "device": str(device),
        }

        logger.info(f"Image inference: {orig_w}x{orig_h}, {len(detections)} detections, {elapsed*1000:.1f}ms")

        response = send_file(buf, mimetype="image/png", download_name="detected.png")
        response.headers["X-Detection-Metadata"] = json.dumps(metadata)
        return response

    except Exception as e:
        logger.error(f"Image inference error: {traceback.format_exc()}")
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500


# ============================================================
# VIDEO ENDPOINT
# ============================================================
@app.route("/predict/video", methods=["POST"])
def predict_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided. Use key 'video'."}), 400

        file = request.files["video"]
        if file.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_VIDEO_EXT:
            return jsonify({"error": f"Unsupported format. Allowed: {ALLOWED_VIDEO_EXT}"}), 400

        conf_threshold = float(request.form.get("confidence", 0.4))
        conf_threshold = max(0.05, min(0.95, conf_threshold))

        job_id = str(uuid.uuid4())[:8]
        input_path = os.path.join(TEMP_DIR, f"input_{job_id}.{ext}")
        output_path = os.path.join(TEMP_DIR, f"output_{job_id}.mp4")

        file.save(input_path)
        logger.info(f"[{job_id}] Saved upload: {input_path} ({os.path.getsize(input_path) / 1e6:.1f} MB)")
        logger.info(f"[{job_id}] Processing with confidence={conf_threshold}")

        stats = process_video(input_path, output_path, conf_threshold)
        logger.info(f"[{job_id}] Done: {stats['frames_processed']} frames, {stats['total_detections']} detections, {stats['processing_fps']} FPS")

        try:
            os.remove(input_path)
        except OSError:
            pass

        import json
        response = send_file(
            output_path, mimetype="video/mp4", as_attachment=True,
            download_name=f"detected_{file.filename.rsplit('.', 1)[0]}.mp4"
        )
        response.headers["X-Video-Stats"] = json.dumps(stats)
        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Video inference error: {traceback.format_exc()}")
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500


# ============================================================
# STARTUP
# ============================================================
if __name__ == "__main__":
    load_model()
    logger.info("Flask backend ready.")
    app.run(host="0.0.0.0", port=5000, debug=False)
