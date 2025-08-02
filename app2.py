import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import joblib

# Load model
clf = joblib.load("model/hand_cricket1.pkl")

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.title("🤚🏏 Hand Cricket Game - ML Edition")

if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.last_capture_time = 0
    st.session_state.frame_count = 0

video_placeholder = st.empty()
message_placeholder = st.empty()

def predict(frame):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten().reshape(1, -1)
            return clf.predict(data)[0]
    return None


if st.button("🎮 Start Game", key="start_btn"):
    st.session_state.running = True
    st.session_state.last_capture_time = time.monotonic()
stop = st.button("🛑 Stop", key="stop_btn")
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected")
            break

        # Show live feed
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Countdown Logic
        elapsed = time.monotonic() - st.session_state.last_capture_time
        remaining = 3 - int(elapsed)
        if remaining > 0:
            message_placeholder.markdown(f"### ⏳ Get ready: {remaining}")
        else:
            message_placeholder.markdown("### 📸 Capturing your move...")
            time.sleep(0.5)
            prediction = predict(frame)
            if prediction:
                st.success(f"🧠 You showed: {prediction}")
                # You can call `play_turn(prediction)` here
            else:
                st.warning("❓ Couldn't detect hand")

            st.session_state.last_capture_time = time.monotonic()

        # Break if user clicks stop (optional)
        if stop:
            st.session_state.running = False
            break

    cap.release()
    cv2.destroyAllWindows()