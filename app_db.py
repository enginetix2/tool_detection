import streamlit as st
import cv2
from ultralytics import YOLO
import time

# -----------------------------
# CONFIGURATIONS
# -----------------------------
MODEL_PATH = "runs/detect/train/weights/best.pt"  # Adjust to your model path
CAMERA_INDEX = 1  # 0 = default camera, 1 = external, etc.

# Mapping of physical tool to YOLO labels (in, out)
TOOL_LABELS = {
    "cable_a": ("cable_a_in", "cable_a_out"),
    "battery": ("battery_in", "battery_out"),
    "cable_b": ("cable_b_in", "cable_b_out"),
    "gpu":     ("gpu_in", "gpu_out")
}

# Build sets of in/out labels for color-coding
IN_LABELS = set(pair[0] for pair in TOOL_LABELS.values())   # e.g. {"cable_a_in", "battery_in", ...}
OUT_LABELS = set(pair[1] for pair in TOOL_LABELS.values())  # e.g. {"cable_a_out", "battery_out", ...}

CHECKMARK = "✅"
CROSSMARK = "❌"

# -----------------------------
# STREAMLIT PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Shadowbox Tool Tracking", layout="centered")
st.title("🔧 Shadowbox Tool Tracking — Live Update")

"""
**Instructions**  
1. Click **Start Detection** to open your camera and begin continuous detection.  
2. Click **Stop Detection** to stop the camera loop.  
"""

# We’ll store whether or not we’re running detection in session state.
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

# Buttons to start/stop the detection loop
start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection")

if start_button:
    st.session_state.run_detection = True
if stop_button:
    st.session_state.run_detection = False

# Placeholders for the video frame and the tool status
video_placeholder = st.empty()
status_placeholder = st.empty()

# Load the YOLO model once (in session_state or globally)
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = YOLO(MODEL_PATH)

model = st.session_state.yolo_model

# -----------------------------
# CONTINUOUS DETECTION LOOP
# -----------------------------
def run_camera_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.warning(f"Failed to open camera index {CAMERA_INDEX}. Check your device/camera settings.")
        return

    while st.session_state.run_detection:
        # 1) Read a frame
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from camera.")
            break

        # 2) Run YOLO detection
        results = model.predict(frame, conf=0.50, iou=0.)

        # We'll build a set of detected class labels each iteration
        detected_classes = set()
        annotated_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf     = float(box.conf[0])

                # Add to our set of detected classes
                detected_classes.add(cls_name)

                # Pick color for bounding box:
                #  - Green if "in" label
                #  - Red if "out" label
                #  - White otherwise
                if cls_name in IN_LABELS:
                    color = (0, 255, 0)
                elif cls_name in OUT_LABELS:
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 255)

                # Draw bounding box + label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{cls_name} ({conf:.2f})"
                cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 3) Convert BGR->RGB for Streamlit and display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame, caption="Live Camera (Annotated)")

        # 4) Build the tool status display
        status_lines = []
        for tool_name, (in_label, out_label) in TOOL_LABELS.items():
            if in_label in detected_classes:
                status_lines.append(f"{CHECKMARK} **{tool_name}** : IN")
            elif out_label in detected_classes:
                status_lines.append(f"{CROSSMARK} **{tool_name}** : OUT")
            else:
                status_lines.append(f"{CROSSMARK} **{tool_name}** : OUT (not detected)")

        status_placeholder.markdown("\n".join(status_lines))

        # 5) Tiny sleep to avoid busy looping
        time.sleep(0.25)

    cap.release()
    # Clear the placeholders (optional)
    video_placeholder.empty()
    status_placeholder.write("Detection stopped.")

# If user pressed 'Start Detection', run the loop
if st.session_state.run_detection:
    run_camera_loop()
