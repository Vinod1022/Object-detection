"""
YOLOv8 Object Detection Streamlit App
- Upload an image in the sidebar
- Choose a model (pretrained or upload your own .pt)
- Adjust confidence and IoU thresholds
- Run detection and view results (annotated image + table + download)

Requirements:
    pip install streamlit ultralytics opencv-python-headless pillow numpy pandas
"""

# pip install streamlit ultralytics opencv-python-headless pillow numpy pandas



import io
from pathlib import Path
from typing import Tuple, List, Dict

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

# Try import ultralytics (YOLOv8). If unavailable, instruct user to install.
try:
    from ultralytics import YOLO
except Exception as e:
    st.error("Missing dependency: ultralytics. Install with `pip install ultralytics` and restart the app.")
    st.stop()

# Small helper functions
@st.cache_resource
def load_model(model_path: str = "yolov8n.pt"):
    """
    Loads the YOLO model. This is cached so it won't reload on every interaction.
    If model_path points to a file that does not exist, ultralytics will try to download if available.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model '{model_path}': {e}")
        raise

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def run_detection_on_pil(model: YOLO, pil_img: Image.Image, conf: float, iou: float, imgsz: int = 640):
    """
    Run detection and return:
      - annotated PIL image
      - pandas DataFrame of detections
      - raw results object
    """
    # Convert to numpy (BGR expected by some libs, but ultralytics accepts PIL/np)
    img_np = np.array(pil_img)
    # ultralytics accepts list of images
    results = model.predict(source=[img_np], conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    # results is a list-like (one entry per image)
    res = results[0]

    # Use ultralytics built-in plotting to get annotated image (returns np.array in RGB)
    try:
        annotated = res.plot()  # returns np array RGB
        annotated_pil = Image.fromarray(annotated)
    except Exception:
        # fallback: show original if plotting fails
        annotated_pil = pil_img.copy()

    # Build dataframe of detections
    detections = []
    boxes = getattr(res, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        # boxes.xyxyn or boxes.xyxy; boxes.cls; boxes.conf
        for i, box in enumerate(boxes):
            # safe extraction
            xyxy = None
            try:
                xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
            except Exception:
                try:
                    xyxy = box.xywh[0].tolist()
                except Exception:
                    xyxy = [None, None, None, None]
            conf_score = float(box.conf[0]) if hasattr(box, "conf") else float(getattr(box, "confidence", 0.0))
            cls = int(box.cls[0]) if hasattr(box, "cls") else int(getattr(box, "class_id", -1))
            name = model.names[cls] if cls in model.names else str(cls)
            detections.append({
                "index": i,
                "class_id": cls,
                "class_name": name,
                "confidence": round(conf_score, 4),
                "x1": round(xyxy[0], 2) if xyxy and xyxy[0] is not None else None,
                "y1": round(xyxy[1], 2) if xyxy and xyxy[1] is not None else None,
                "x2": round(xyxy[2], 2) if xyxy and xyxy[2] is not None else None,
                "y2": round(xyxy[3], 2) if xyxy and xyxy[3] is not None else None,
            })
    df = pd.DataFrame(detections)
    return annotated_pil, df, res

# --- Streamlit UI ---
st.set_page_config(page_title="YOLOv8 Object Detection App", layout="wide")
st.sidebar.title("Upload an Image")
st.sidebar.markdown("Choose an image to detect objects. Supported: JPG, JPEG, PNG. Limit ~20MB.")

uploaded_image = st.sidebar.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])
st.sidebar.write("")  # spacing

st.sidebar.markdown("---")
st.sidebar.title("Model")
st.sidebar.markdown("Choose a YOLOv8 model or upload your custom `.pt` file.")
use_pretrained = st.sidebar.selectbox("Pretrained model", ["yolov8n.pt (fast, small)", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt", "Custom upload (.pt)"])
custom_model_file = None
if use_pretrained == "Custom upload (.pt)":
    custom_model_file = st.sidebar.file_uploader("Upload model (.pt)", type=["pt"])

# Model config
st.sidebar.markdown("---")
st.sidebar.markdown("Detection parameters")
conf_thr = st.sidebar.slider("Confidence threshold", min_value=0.01, max_value=1.0, value=0.35, step=0.01)
iou_thr = st.sidebar.slider("IoU threshold (NMS)", min_value=0.01, max_value=1.0, value=0.45, step=0.01)
imgsz = st.sidebar.selectbox("Inference image size (px)", [320, 480, 640, 800, 1024], index=2)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: use `yolov8n.pt` for quick tests and a custom `.pt` if you trained one in your notebook.")

# Main layout
st.title("YOLOv8 Object Detection App")
st.markdown("Please upload an image to start detection.")

# Model selection & load
model_path = "yolov8n.pt"
if use_pretrained != "Custom upload (.pt)":
    model_choice_map = {
        "yolov8n.pt (fast, small)": "yolov8n.pt",
        "yolov8s.pt": "yolov8s.pt",
        "yolov8m.pt": "yolov8m.pt",
        "yolov8l.pt": "yolov8l.pt",
        "yolov8x.pt": "yolov8x.pt",
    }
    model_path = model_choice_map.get(use_pretrained, "yolov8n.pt")
else:
    # save uploaded custom model temporarily to a file and use that path
    if custom_model_file is None:
        st.sidebar.warning("Upload your custom .pt model file to use it.")
    else:
        tmp_model_path = Path("custom_uploaded_model.pt")
        with open(tmp_model_path, "wb") as f:
            f.write(custom_model_file.read())
        model_path = str(tmp_model_path)

# Load model (cached)
with st.spinner(f"Loading model `{model_path}` ..."):
    model = load_model(model_path)

# If image uploaded, show preview and run detect
if uploaded_image is None:
    st.info("Waiting for image upload in the left sidebar.")
else:
    try:
        pil_img = Image.open(uploaded_image).convert("RGB")
    except Exception as e:
        st.error(f"Error reading image: {e}")
        st.stop()

    # Show two columns: original + annotated
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Input image")
        st.image(pil_img, use_column_width=True)

    # Detect button
    if st.button("Run detection"):
        with st.spinner("Running detection..."):
            annotated_pil, detections_df, raw = run_detection_on_pil(model, pil_img, conf=conf_thr, iou=iou_thr, imgsz=imgsz)

        with col2:
            st.subheader("Detection result")
            st.image(annotated_pil, use_column_width=True)
            # allow download of annotated image
            img_bytes = pil_to_bytes(annotated_pil, fmt="PNG")
            st.download_button("Download annotated image", data=img_bytes, file_name="detection.png", mime="image/png")

        # show detections table
        st.subheader("Detections")
        if detections_df.empty:
            st.write("No objects detected — try lowering the confidence threshold.")
        else:
            st.dataframe(detections_df)

        # show a small summary
        if not detections_df.empty:
            counts = detections_df['class_name'].value_counts().rename_axis('class_name').reset_index(name='counts')
            st.markdown("**Detected classes:**")
            st.table(counts)
