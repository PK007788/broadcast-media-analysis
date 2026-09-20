"""
Streamlit Frontend — Broadcast Overlay Detection
=================================================
IMAGE-FIRST interface with VIDEO as secondary mode.
Connects to the Flask backend API.
"""

# pyrefly: ignore [missing-import]
import streamlit as st
import requests
import tempfile
import os
import io
import json
import time
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Broadcast Overlay Detection",
    page_icon="📺",
    layout="wide",
)

BACKEND_URL = "http://localhost:5000"

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        font-family: 'Inter', sans-serif;
    }
    .subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #888;
        margin-bottom: 1.5rem;
        font-family: 'Inter', sans-serif;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stat-card h3 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .stat-card p {
        margin: 0;
        font-size: 0.8rem;
        opacity: 0.85;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #444;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stRadio"] > label {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-title">📺 Broadcast Overlay Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect rectangular broadcast overlays in news frames &nbsp;·&nbsp; SSDLite320 + MobileNetV3-Large &nbsp;·&nbsp; 0.427 GFLOPs &nbsp;·&nbsp; 2.21M params</p>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Controls")

    mode = st.radio(
        "Detection Mode",
        ["🖼️ Image Detection", "🎬 Video Detection"],
        index=0,
        help="Image detection is the primary feature."
    )

    st.divider()

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Only detections above this confidence will be drawn."
    )

    st.divider()
    st.header("📊 Model Information")

    backend_ok = False
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        backend_ok = True
        st.success("Backend: **Online**")
        st.markdown(f"""
| Property | Value |
|:---|:---|
| **Model** | SSDLite320-MNv3 |
| **Class** | `overlay` |
| **Parameters** | {health.get('parameters', 'N/A')} |
| **GFLOPs** | {health.get('gflops', 'N/A')} |
| **Size** | {health.get('model_size_mb', 'N/A')} MB |
| **Device** | `{health.get('device', 'N/A')}` |
| **GPU** | {health.get('gpu_name', 'N/A')} |
        """)
    except Exception:
        st.error("Backend: **Offline**")
        st.info("Start the Flask backend first:\n```\npython app/backend/app.py\n```")

    st.divider()
    st.caption("IIT Patna · Broadcast Overlay Detection")

# ============================================================
# IMAGE DETECTION MODE
# ============================================================
if "🖼️" in mode:
    st.markdown('<p class="section-header">Upload News Frame</p>', unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "Select a broadcast news frame",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a news screenshot or extracted video frame.",
        key="img_upload"
    )

    if uploaded_image is not None:
        # Display original
        original_img = Image.open(uploaded_image)
        orig_w, orig_h = original_img.size

        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown('<p class="section-header">Original Image</p>', unsafe_allow_html=True)
            st.image(original_img, use_container_width=True, caption=f"Original ({orig_w}×{orig_h})")

        # Run Detection button
        run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
        with run_col2:
            run_btn = st.button(
                "🔍 Run Detection",
                use_container_width=True,
                type="primary",
                disabled=not backend_ok,
                key="img_run"
            )

        if run_btn:
            if not backend_ok:
                st.error("Cannot run detection — Flask backend is offline.")
            else:
                with st.spinner("Running overlay detection..."):
                    try:
                        uploaded_image.seek(0)
                        files = {"image": (uploaded_image.name, uploaded_image.read(), "image/png")}
                        data = {"confidence": str(conf_threshold)}

                        start = time.time()
                        response = requests.post(
                            f"{BACKEND_URL}/predict/image",
                            files=files,
                            data=data,
                            timeout=30
                        )
                        total_time = time.time() - start

                        if response.status_code == 200:
                            # Parse metadata from header
                            meta_str = response.headers.get("X-Detection-Metadata", "{}")
                            metadata = json.loads(meta_str)

                            # Display annotated image
                            annotated_img = Image.open(io.BytesIO(response.content))
                            with col_det:
                                st.markdown('<p class="section-header">Detection Result</p>', unsafe_allow_html=True)
                                st.image(annotated_img, use_container_width=True,
                                         caption=f"Detected: {metadata.get('detections', 0)} overlay(s)")

                            # Statistics
                            st.markdown('<p class="section-header">Detection Statistics</p>', unsafe_allow_html=True)
                            s1, s2, s3, s4 = st.columns(4)
                            with s1:
                                st.markdown(f"""<div class="stat-card">
                                    <h3>{metadata.get('detections', 0)}</h3>
                                    <p>Detections</p>
                                </div>""", unsafe_allow_html=True)
                            with s2:
                                st.markdown(f"""<div class="stat-card">
                                    <h3>{metadata.get('processing_time_ms', 0):.1f} ms</h3>
                                    <p>Inference Time</p>
                                </div>""", unsafe_allow_html=True)
                            with s3:
                                proc_ms = metadata.get('processing_time_ms', 1)
                                fps_val = 1000.0 / proc_ms if proc_ms > 0 else 0
                                st.markdown(f"""<div class="stat-card">
                                    <h3>{fps_val:.0f}</h3>
                                    <p>Inference FPS</p>
                                </div>""", unsafe_allow_html=True)
                            with s4:
                                st.markdown(f"""<div class="stat-card">
                                    <h3>{conf_threshold:.2f}</h3>
                                    <p>Threshold</p>
                                </div>""", unsafe_allow_html=True)

                            # Per-detection table
                            boxes = metadata.get("boxes", [])
                            if boxes:
                                st.markdown("**Detected Overlays:**")
                                for i, b in enumerate(boxes):
                                    st.text(f"  #{i+1}  overlay {b['score']:.3f}  [{b['x1']}, {b['y1']}, {b['x2']}, {b['y2']}]")

                            # Download
                            st.divider()
                            buf = io.BytesIO()
                            annotated_img.save(buf, format="PNG")
                            st.download_button(
                                label="⬇️ Download Annotated Image",
                                data=buf.getvalue(),
                                file_name=f"detected_{uploaded_image.name.rsplit('.', 1)[0]}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        else:
                            err = response.json().get("error", "Unknown error")
                            st.error(f"Detection failed: {err}")

                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to Flask backend. Is it running on port 5000?")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Upload a news frame to detect broadcast overlays (tickers, banners, info panels).")


# ============================================================
# VIDEO DETECTION MODE
# ============================================================
elif "🎬" in mode:
    st.markdown('<p class="section-header">Upload News Video</p>', unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Select a broadcast news video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Upload a news video clip for overlay detection.",
        key="vid_upload"
    )

    if uploaded_video is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-header">Original Video</p>', unsafe_allow_html=True)
            st.video(uploaded_video)

        run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
        with run_col2:
            run_btn = st.button(
                "🚀 Run Video Detection",
                use_container_width=True,
                type="primary",
                disabled=not backend_ok,
                key="vid_run"
            )

        if run_btn:
            if not backend_ok:
                st.error("Cannot run detection — Flask backend is offline.")
            else:
                with st.spinner("Processing video... This may take a few minutes depending on video length."):
                    progress_bar = st.progress(0, text="Uploading video...")
                    try:
                        progress_bar.progress(10, text="Uploading to backend...")
                        uploaded_video.seek(0)
                        files = {"video": (uploaded_video.name, uploaded_video.read(), "video/mp4")}
                        data = {"confidence": str(conf_threshold)}

                        progress_bar.progress(20, text="Running overlay detection on all frames...")
                        start_time = time.time()
                        response = requests.post(
                            f"{BACKEND_URL}/predict/video",
                            files=files,
                            data=data,
                            timeout=1200
                        )
                        elapsed = time.time() - start_time

                        if response.status_code == 200:
                            progress_bar.progress(90, text="Saving result...")

                            stats_str = response.headers.get("X-Video-Stats", "{}")
                            stats = json.loads(stats_str)

                            output_path = os.path.join(tempfile.gettempdir(), f"detected_{uploaded_video.name}")
                            with open(output_path, "wb") as f:
                                f.write(response.content)

                            progress_bar.progress(100, text="Done!")

                            with col2:
                                st.markdown('<p class="section-header">Detected Overlays</p>', unsafe_allow_html=True)
                                st.video(output_path)

                            # Statistics
                            st.markdown('<p class="section-header">Processing Statistics</p>', unsafe_allow_html=True)
                            s1, s2, s3, s4 = st.columns(4)
                            with s1:
                                st.markdown(f"""<div class="stat-card">
                                    <h3>{stats.get('frames_processed', '?')}</h3>
                                    <p>Frames Processed</p>
                                </div>""", unsafe_allow_html=True)
                            with s2:
                                st.markdown(f"""<div class="stat-card">
                                    <h3>{stats.get('total_detections', '?')}</h3>
                                    <p>Total Detections</p>
                                </div>""", unsafe_allow_html=True)
                            with s3:
                                st.markdown(f"""<div class="stat-card">
                                    <h3>{stats.get('processing_fps', '?')}</h3>
                                    <p>Processing FPS</p>
                                </div>""", unsafe_allow_html=True)
                            with s4:
                                st.markdown(f"""<div class="stat-card">
                                    <h3>{elapsed:.1f}s</h3>
                                    <p>Total Time</p>
                                </div>""", unsafe_allow_html=True)

                            st.divider()
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download Processed Video",
                                    data=f,
                                    file_name=f"detected_{uploaded_video.name}",
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                        else:
                            progress_bar.empty()
                            err = response.json().get("error", "Unknown error")
                            st.error(f"Detection failed: {err}")

                    except requests.exceptions.ConnectionError:
                        progress_bar.empty()
                        st.error("Could not connect to Flask backend. Is it running on port 5000?")
                    except requests.exceptions.Timeout:
                        progress_bar.empty()
                        st.error("Request timed out. The video may be too large.")
                    except Exception as e:
                        progress_bar.empty()
                        st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Upload a news video to detect broadcast overlays across all frames.")
