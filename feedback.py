import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import os
from uuid import uuid4

st.set_page_config(page_title="Hand Gesture Feedback", page_icon="🔍", layout="centered")
st.title("💡 Hand Gesture Feedback System")

st.markdown("""
This app allows users to:
- Upload an image of a hand gesture
- Predict the gesture (placeholder)
- Provide feedback if the prediction is incorrect
""")

# Directory to save feedback images
FEEDBACK_DIR = "feedback_data"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Upload image
uploaded_file = st.file_uploader("Upload a hand gesture image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Gesture", use_column_width=True)

    # Convert image to NumPy array (OpenCV format)
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # --- Placeholder for prediction ---
    predicted_label = "3"  # Dummy label for demo
    st.success(f"Predicted Gesture: {predicted_label}")

    # Feedback Section
    st.markdown("### ✉️ Provide Feedback")
    feedback = st.radio("Was this prediction correct?", ["Yes", "No"])

    if feedback == "No":
        correct_label = st.selectbox("Choose the correct label:", [str(i) for i in range(1, 11)])
        if st.button("Submit Correction"):
            label_dir = os.path.join(FEEDBACK_DIR, correct_label)
            os.makedirs(label_dir, exist_ok=True)
            save_path = os.path.join(label_dir, f"{uuid4()}.jpg")
            cv2.imwrite(save_path, img_bgr)
            st.success(f"🚀 Feedback saved under label {correct_label}!")
    else:
        st.info("Thank you for confirming the prediction!")
