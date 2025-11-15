"""
Helmet Detection Streamlit App (YOLOv8)
- Upload image
- Uses your trained best.pt model (with_helmet / without_helmet)
- Shows detections, summary, counts & downloadable results
"""

import io
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

# Try importing YOLOv8
try:
    from ultralytics import YOLO
except:
    st.error("Ultralytics YOLO not installed. Run: pip install ultralytics")
    st.stop()

# ---------------------------
# Model loader
# ---------------------------
@st.cache_resource
def load_model(model_path: str):
    try:
        model = YOLO(model_path)

        # 🔥 Force your class names (important fix)
        model.names = {
            0: "with_helmet",
            1: "without_helmet"
        }

        return model
    except Exception as e:
        st.error(f"Failed to load model: {model_path}\n{e}")
        raise

def pil_to_bytes(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def run_yolo(model, pil_img, conf, iou, imgsz=640):
    img_np = np.array(pil_img)
    results = model.predict(
        source=[img_np],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False
    )
    res = results[0]

    # Annotated image
    annotated_np = res.plot()
    annotated_pil = Image.fromarray(annotated_np)

    # Build detections table
    detections = []
    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf_score = float(box.conf[0])
            name = model.names[cls]  # forced mapping
            detections.append({
                "class_name": name,
                "confidence": round(conf_score, 3),
                "x1": round(xyxy[0], 2),
                "y1": round(xyxy[1], 2),
                "x2": round(xyxy[2], 2),
                "y2": round(xyxy[3], 2),
            })

    df = pd.DataFrame(detections)
    return annotated_pil, df


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Helmet Detection App", layout="wide")
st.title("🛵 Helmet Detection App")
st.write("Detect **with_helmet** or **without_helmet** using your YOLOv8 model.")

# Sidebar upload
uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Sidebar model selection
st.sidebar.markdown("---")
st.sidebar.subheader("Model Selection")

model_source = st.sidebar.selectbox(
    "Choose model",
    [
        "Use my best.pt (default)",
        "Upload custom .pt"
    ]
)

custom_model = None
if model_source == "Upload custom .pt":
    custom_model = st.sidebar.file_uploader("Upload .pt model", type=["pt"])

# Sidebar parameters
st.sidebar.markdown("---")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.01)
iou = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.01)
imgsz = st.sidebar.selectbox("Image Size", [320, 480, 640, 800], index=2)

# Decide model path
if model_source == "Use my best.pt (default)":
    model_path = "best.pt"     # Your helmet model must be here
else:
    if custom_model is None:
        st.warning("Upload your custom .pt model.")
        st.stop()
    temp_path = Path("uploaded_model.pt")
    with open(temp_path, "wb") as f:
        f.write(custom_model.read())
    model_path = str(temp_path)

# Load model
with st.spinner(f"Loading model: {model_path}"):
    model = load_model(model_path)

# No image uploaded yet
if uploaded_image is None:
    st.info("Upload an image from the left sidebar.")
    st.stop()

# Load image
pil_img = Image.open(uploaded_image).convert("RGB")

# Columns for input/output
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    st.image(pil_img, use_column_width=True)

if st.button("Run Detection"):
    with st.spinner("Detecting helmets..."):
        annotated_pil, df = run_yolo(model, pil_img, conf, iou, imgsz)

    with col2:
        st.subheader("Detection Result")
        st.image(annotated_pil, use_column_width=True)
        st.download_button(
            "Download Annotated Image",
            data=pil_to_bytes(annotated_pil),
            file_name="helmet_detection.png",
            mime="image/png"
        )

    # Display results
    st.subheader("Detected Objects")
    if df.empty:
        st.warning("🚫 No detections found!")
    else:
        st.dataframe(df)

        # Compute counts
        class_counts = df["class_name"].value_counts().to_dict()

        # ----------------------------
        # 🎯 FINAL FIXED SUMMARY BLOCK
        # ----------------------------
        st.subheader("Helmet Summary")

        with_helmet = class_counts.get("with_helmet", 0)
        without_helmet = class_counts.get("without_helmet", 0)

        st.success(f"🟢 People WITH Helmet: {with_helmet}")
        st.error(f"🔴 People WITHOUT Helmet: {without_helmet}")



