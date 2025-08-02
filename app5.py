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

# Page Config
st.set_page_config(page_title="Hand Cricket ML", page_icon="🏏", layout="wide")

# CSS for styling
st.markdown("""
<style>
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

# Game State
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.last_capture_time = 0
    st.session_state.frame_count = 0
    st.session_state.player_score = 0
    st.session_state.bot_score = 0
    st.session_state.batting = "player"
    st.session_state.out = False
    st.session_state.last_player_num = None
    st.session_state.last_bot_num = None

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

# Game Logic
def play_turn(player_num):
    bot_num = np.random.randint(1, 11)
    st.session_state.last_player_num = player_num
    st.session_state.last_bot_num = bot_num

    if player_num == bot_num:
        st.session_state.out = True
        st.warning(f"🏏 OUT! Final Score: {st.session_state.player_score}")
        st.session_state.batting = "bot" if st.session_state.batting == "player" else "end"
    else:
        if st.session_state.batting == "player":
            st.session_state.player_score += player_num
        elif st.session_state.batting == "bot":
            st.session_state.bot_score += bot_num

def update_score():
    player_score_placeholder.metric("Your Score", st.session_state.player_score)
    bot_move_placeholder.markdown(
            f"<h2 style='color:#FF5733;'>🤖 Bot showed: <b>{st.session_state.last_bot_num}</b></h2>",
            unsafe_allow_html=True
        )

# Empty space
st.text("")
st.text("")

# Layout: Left space | Player | Bot
left_space, col1, col2 = st.columns([0.4, 1, 1], gap="medium")

with col1:
    if st.button("🎮 Start Game"):
        st.session_state.running = True
        st.session_state.last_capture_time = time.monotonic()

    st.subheader("👤 Player")
    player_video_placeholder = st.empty()
    player_prediction_placeholder = st.empty()
    player_score_placeholder = st.empty()

    if st.session_state.last_player_num:
        player_prediction_placeholder.markdown(
            f"<h3 style='color:green;'>🧠 You showed: {st.session_state.last_player_num}</h3>",
            unsafe_allow_html=True
        )

    player_score_placeholder.metric("Your Score", st.session_state.player_score)

with col2:
    if st.button("🛑 Stop Game"):
        st.session_state.running = False

    st.subheader("🤖 Bot")
    bot_move_placeholder = st.empty()

    if st.session_state.last_bot_num:
        bot_move_placeholder.markdown(
            f"<h2 style='color:#FF5733;'>🤖 Bot showed: <b>{st.session_state.last_bot_num}</b></h2>",
            unsafe_allow_html=True
        )

    st.metric("Bot Score", st.session_state.bot_score)

# Webcam logic
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    while st.session_state.running and not st.session_state.out:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to access webcam")
            break

        # player_video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        elapsed = time.monotonic() - st.session_state.last_capture_time
        remaining = 3 - int(elapsed)

        if remaining > 0:
            countdown_text = f"{remaining}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 5
            thickness = 10
            text_size = cv2.getTextSize(countdown_text, font, font_scale, thickness)[0]
            text_x = int((frame.shape[1] - text_size[0]) / 2)
            text_y = int((frame.shape[0] + text_size[1]) / 2)
            cv2.putText(frame, countdown_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            player_video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        else:
            cv2.putText(frame, "Capturing...", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 3, cv2.LINE_AA)

            # # Draw camera body (black rectangle)
            # cv2.rectangle(frame, (30, 30), (130, 90), (0, 0, 0), -1)  # Black camera body (BGR)
            # # Draw lens (circle)
            # cv2.circle(frame, (80, 60), 15, (50, 50, 50), -1)        # Dark gray lens
            # cv2.circle(frame, (80, 60), 7, (255, 255, 255), -1)      # White reflection highlight
            # # Draw flash (bright yellow rectangle)
            # cv2.rectangle(frame, (100, 20), (115, 35), (0, 255, 255), -1)  # Yellow flash (BGR)

            player_video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            time.sleep(0.5)
            player_num = predict(frame)
            if player_num:
                play_turn(player_num)
                update_score()
            else:
                st.warning("❓ Couldn't detect hand")
            st.session_state.last_capture_time = time.monotonic()

        # player_video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if st.session_state.batting == "end":
            st.session_state.running = False
            break
        
    cap.release()
    cv2.destroyAllWindows()

# Game End Results
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