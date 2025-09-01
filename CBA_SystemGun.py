import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from pathlib import Path
from datetime import datetime
import os  # Import os for handling file paths

# Load the YOLO model for gun detection
try:
    gun_model = YOLO('/Users/apple/Documents/NewYOLOGun/Guns & Knives/weights/best.pt')
    st.success("Gun Detection Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading Gun Detection model: {e}")
    st.stop()

# Function to process and annotate an uploaded image for gun detection
def detect_guns_in_image(image_file):
    try:
        # Read image from uploaded file
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Not able to detect image. Please check the file format.")
            return None, None

        # Run the gun detection model
        gun_results = gun_model(img, verbose=False)

        # Draw bounding boxes for detected guns
        annotated_img = img.copy()
        for result in gun_results:
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    class_name = gun_model.names[cls]
                    if conf > 0.5:  # Confidence threshold
                        color = (0, 0, 255)  # Red for guns
                        label = f"{class_name} {conf:.2f}"
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_img, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"gun_detected_{timestamp}.jpg"
        output_path = Path(f"./results/{output_filename}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated_img)

        st.success(f"Image processed. Output saved: {output_path}")
        return annotated_img, str(output_path.resolve())
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# Streamlit app for gun detection
st.title("Gun Detection System")
st.write("Upload an image to detect guns using the YOLO model.")

# File uploader for image
image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if image_file is not None:
    annotated_img, output_path = detect_guns_in_image(image_file)
    if annotated_img is not None:
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, caption="Detected Image", use_container_width=True)

        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Detected Image",
                data=file,
                file_name=os.path.basename(output_path),  # Corrected to extract the file name
                mime="image/jpeg"
            )