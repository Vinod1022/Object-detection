
import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import time

st.set_page_config(page_title="Helmet Detection App", layout="wide")

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH =r"C:\Users\VINODKUMAR\Downloads\Helmet Detection\Helmet Detection_1\best.pt"

@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# ----------------------------------------
# Helper: Run detection + draw colored boxes
# ----------------------------------------
def detect_and_draw(image, model):
    results = model(image)[0]

    annotated = image.copy()
    with_helmet = 0
    without_helmet = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.model.names[cls]
        x1, y1, x2, y2 = box.xyxy[0]

        # choose box color
        if label.lower() == "with_helmet":
            color = (0, 255, 0)      # green
            with_helmet += 1
        else:
            color = (255, 0, 0)      # red
            without_helmet += 1

        # draw
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(
            annotated, label, (int(x1), int(y1)-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )

    total = with_helmet + without_helmet
    return annotated, total, with_helmet, without_helmet

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🪖 Helmet Detection (YOLO) – Full App")
st.write("Upload a video or use your webcam.")

mode = st.radio("Select Input Source", ["Webcam", "Upload Video"])

# -------------------------------
# 🚀 Webcam Mode
# -------------------------------
if mode == "Webcam":
    run = st.checkbox("Start Webcam")

    if run:
        cam = cv2.VideoCapture(0)
        frame_display = st.empty()

        while True:
            ret, frame = cam.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            annotated, total, wh, woh = detect_and_draw(frame_rgb, model)

            frame_display.image(annotated, channels="RGB")

            st.sidebar.markdown("### 📊 Summary (Live)")
            st.sidebar.write(f"👥 Total Persons: **{total}**")
            st.sidebar.write(f"🟩 With Helmet: **{wh}**")
            st.sidebar.write(f"🟥 Without Helmet: **{woh}**")

            if not run:
                break

        cam.release()

# -------------------------------
# 🎬 Video Upload Mode
# -------------------------------
else:
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save temporary video
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp.name)
        frame_display = st.empty()

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated, total, wh, woh = detect_and_draw(frame_rgb, model)

            frame_display.image(annotated, channels="RGB")

            st.sidebar.markdown("### 📊 Summary (Video)")
            st.sidebar.write(f"👥 Total Persons: **{total}**")
            st.sidebar.write(f"🟩 With Helmet: **{wh}**")
            st.sidebar.write(f"🟥 Without Helmet: **{woh}**")

            time.sleep(0.02)

        cap.release()


