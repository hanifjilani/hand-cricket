import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import joblib

# Load model
clf = joblib.load("model/hand_cricket1.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Game state initialization
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.last_capture_time = 0
    st.session_state.frame_count = 0
    st.session_state.player_score = 0
    st.session_state.bot_score = 0
    st.session_state.batting = "player"
    st.session_state.out = False

st.set_page_config(page_title="Hand Cricket ML", page_icon="🏏", layout="wide")

st.markdown("""
<style>
/* Background gradient for the whole page */
body {
    background: linear-gradient(135deg, #fceabb, #f8b500);
    margin: 0;
    padding: 0;
}


.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    font-family: 'Segoe UI', sans-serif;
    margin-top: 30px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #444;
    font-style: italic;
    margin-top: -10px;
    margin-bottom: 20px;
    font-family: 'Segoe UI', sans-serif;
}
</style>

<div class="main">
    <h1 class="title">🤚🏏 Hand Cricket Game - ML Edition</h1>
    <p class="subtitle">Play your favorite childhood game using the power of AI 🎯</p>
</div>
""", unsafe_allow_html=True)

# Prediction function
def predict(frame):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten().reshape(1, -1)
            return clf.predict(data)[0]
    return None

# Game logic
def play_turn(player_num):
    bot_num = np.random.randint(1, 11)
    bot_move_placeholder.markdown(f"🤖 Bot showed: **{bot_num}**")

    if player_num == bot_num:
        st.session_state.out = True
        st.warning(f"🏏 OUT! Final Score: {st.session_state.player_score}")
        st.session_state.batting = "bot" if st.session_state.batting == "player" else "end"
    else:
        if st.session_state.batting == "player":
            st.session_state.player_score += player_num
        elif st.session_state.batting == "bot":
            st.session_state.bot_score += bot_num

#Empty space
st.text("")
st.text("")

# Start and Stop buttons
# col1_btn, col2_btn = st.columns(2)  # Adjust width ratios
# with col1_btn:
#     if st.button("🎮 Start Game"):
#         st.session_state.running = True
#         st.session_state.last_capture_time = time.monotonic()
# with col2_btn:
#     if st.button("🛑 Stop Game"):
#         st.session_state.running = False

# Layout for Player | Divider | Bot
left_space, col1, col2 = st.columns([0.4,1,1], gap="medium")

with col1:
    if st.button("🎮 Start Game"):
        st.session_state.running = True
        st.session_state.last_capture_time = time.monotonic()
    st.subheader("👤 Player")
    player_video_placeholder = st.empty()
    player_prediction_placeholder = st.empty()
    st.metric("Your Score", st.session_state.player_score)

with col2:
    if st.button("🛑 Stop Game"):
        st.session_state.running = False
    st.subheader("🤖 Bot")
    bot_move_placeholder = st.empty()
    st.metric("Bot Score", st.session_state.bot_score)

# Webcam logic
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    while st.session_state.running and not st.session_state.out:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to access webcam")
            break

        player_video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        elapsed = time.monotonic() - st.session_state.last_capture_time
        remaining = 3 - int(elapsed)

        if remaining > 0:
            player_prediction_placeholder.markdown(f"⏳ Show your number in: **{remaining}**")
        else:
            player_prediction_placeholder.markdown("📸 Capturing...")
            time.sleep(0.5)
            player_num = predict(frame)
            if player_num:
                player_prediction_placeholder.success(f"🧠 You showed: {player_num}")
                play_turn(player_num)
            else:
                player_prediction_placeholder.warning("❓ Couldn't detect hand")
            st.session_state.last_capture_time = time.monotonic()

        # Game end condition
        if st.session_state.batting == "end":
            st.session_state.running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# Game End Messages
if st.session_state.batting == "end":
    st.markdown("---")
    if st.session_state.player_score > st.session_state.bot_score:
        st.success("🎉 You Won!")
    elif st.session_state.player_score < st.session_state.bot_score:
        st.error("😢 Bot Won!")
    else:
        st.warning("🤝 It's a Tie!")
    if st.button("🔁 Play Again"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]