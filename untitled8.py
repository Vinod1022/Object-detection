import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# Load your custom YOLOv8 helmet detection model
model = YOLO("/content/runs/detect/train/weights/best.pt")

# Streamlit page settings
st.set_page_config(page_title="Helmet Detection App", layout="wide")

st.title("🚴 Helmet Detection App")
st.write("Upload an image to detect if the person is wearing a helmet.")

# Sidebar upload
st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose image (JPG/JPEG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Read image
    original_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(original_img)

    # Predict
    results = model.predict(img_array, imgsz=640)

    # Annotated output
    annotated_img = results[0].plot()
    annotated_img = Image.fromarray(annotated_img)

    # Get class names
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = results[0].names

    # Count detections
    class_counts = {}
    for cid in class_ids:
        name = class_names[cid]
        class_counts[name] = class_counts.get(name, 0) + 1

    # Two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input image")
        st.image(original_img, use_column_width=True)

    with col2:
        st.subheader("Detection result")
        st.image(annotated_img, use_column_width=True)

    # Summary Box
    st.subheader("Detection Summary")

    if "with_helmet" in class_counts:
        st.success(f"🟢 Helmet Detected: {class_counts.get('with_helmet', 0)} person(s)")

    if "without_helmet" in class_counts:
        st.error(f"🔴 No Helmet: {class_counts.get('without_helmet', 0)} person(s)")

    if not class_counts:
        st.info("No person detected.")

    # Show table
    st.subheader("Detected classes:")
    st.write(class_counts)

    # Download button
    img_bytes = io.BytesIO()
    annotated_img.save(img_bytes, format="PNG")
    st.download_button(
        label="Download annotated image",
        data=img_bytes.getvalue(),
        file_name="helmet_detection.png",
        mime="image/png"
    )

