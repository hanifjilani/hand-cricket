import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import mediapipe as mp
import joblib
from datetime import datetime
import tempfile
from supabase import create_client
import os

# -------------------------
# Model & MediaPipe Loading
# -------------------------
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, '..', 'model', 'hand_cricket1.pkl')
MODEL_PATH = os.path.abspath(model_path)

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load model ({path}): {e}")
            return None
    return None

@st.cache_resource
def get_mp_hands():
    return mp.solutions.hands, mp.solutions.drawing_utils

clf = load_model(MODEL_PATH)
mp_hands, mp_drawing = get_mp_hands()

# -------------------------
# Supabase Client
# -------------------------
SUPABASE_URL = st.secrets["supabase_url"]
SUPABASE_KEY = st.secrets["supabase_key"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------
# App State
# -------------------------
# Initialize session state variables
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# -------------------------
# Video Processor
# -------------------------
class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.pred = None
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame = img.copy()

        with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)
                data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten().reshape(1, -1)
                self.pred = int(clf.predict(data)[0])
            else:
                self.pred = None

        # Overlay text
        if self.pred is not None:
            cv2.putText(img, f"Pred: {self.pred}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------
# UI Layout
# -------------------------
st.title("🖐 Hand Cricket - Feedback")
st.write("Live detection + feedback storage to Supabase")

col1, col2 = st.columns([3, 1])

with col1:
    ctx = webrtc_streamer(
        key="hand-feedback",
        video_processor_factory=HandProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

with col2:
    st.markdown("### Controls")
    if ctx.video_processor:
        if st.button("📸 Capture Frame"):
            st.session_state.last_frame = ctx.video_processor.frame
            st.session_state.last_prediction = ctx.video_processor.pred


# -------------------------
# Feedback Section (Below)
# -------------------------
st.markdown("---")
st.subheader("📤 Submit Feedback")

if st.session_state.last_frame is not None:
    st.image(st.session_state.last_frame, channels="BGR", caption=f"Predicted: {st.session_state.last_prediction}")
    correct_label = st.selectbox("Correct Label (if wrong)", list(range(1, 11)))

    if st.button("Submit Feedback"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"feedback_{ts}.jpg"
        tmp_path = os.path.join(tempfile.gettempdir(), fname)
        cv2.imwrite(tmp_path, st.session_state.last_frame)

        # Upload image
        with open(tmp_path, "rb") as f:
            supabase.storage.from_("feedback-images").upload(fname, f)

        # Upload metadata
        supabase.table("feedback").insert({
            "timestamp": datetime.now().isoformat(),
            "predicted_label": st.session_state.last_prediction,
            "correct_label": correct_label,
            "image_filename": fname
        }).execute()

        st.success("✅ Feedback submitted to Supabase!")
        last_frame = None