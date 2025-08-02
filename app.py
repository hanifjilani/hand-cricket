import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# Load trained model
clf = joblib.load("model/hand_cricket1.pkl")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Game State
if "player_score" not in st.session_state:
    st.session_state.player_score = 0
    st.session_state.bot_score = 0
    st.session_state.batting = "player"
    st.session_state.out = False

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
    bot_num = np.random.randint(1, 10)
    st.write(f"🤖 Bot played: {bot_num}")
    if player_num == bot_num:
        st.session_state.out = True
        st.write(f"🏏 OUT! Total score: {st.session_state.player_score if st.session_state.batting == 'player' else st.session_state.bot_score}")
        st.session_state.batting = "bot" if st.session_state.batting == "player" else "end"
    else:
        if st.session_state.batting == "player":
            st.session_state.player_score += player_num
        else:
            st.session_state.bot_score += bot_num

# Streamlit UI
st.title("🤚🏏 Hand Cricket Game - ML Edition")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
- Show hand gesture (1-10) to webcam
- Press `Play` to take your shot
- If your number ≠ bot's → runs added
- Same number = you're OUT!
- Then bot bats and tries to beat your score.
""")

# Webcam Input
run_game = st.button("🎮 Play")
frame_placeholder = st.empty()

if run_game and not st.session_state.out and st.session_state.batting != "end":
    cap = cv2.VideoCapture(0)
    captured = False
    while not captured:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected")
            break
        # frame = cv2.flip(frame, 1)
        player_num = predict(frame)

        if player_num:
            st.write(f"🧠 Detected: {player_num}")
            play_turn(player_num)
            captured = True
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    cap.release()

# Scoreboard
st.markdown("---")
st.metric("👤 Your Score", st.session_state.player_score)
st.metric("🤖 Bot Score", st.session_state.bot_score)
st.write(f"Batting: `{st.session_state.batting}`")

if st.session_state.batting == "end":
    if st.session_state.player_score > st.session_state.bot_score:
        st.success("🎉 You Won!")
    elif st.session_state.player_score < st.session_state.bot_score:
        st.error("😢 Bot Won!")
    else:
        st.warning("🤝 It's a Tie!")
    if st.button("🔁 Play Again"):
        for key in st.session_state.keys():
            del st.session_state[key]