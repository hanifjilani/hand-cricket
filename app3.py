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

# Game State
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.last_capture_time = 0
    st.session_state.player_score = 0
    st.session_state.bot_score = 0
    st.session_state.batting = "player"
    st.session_state.out = False

video_placeholder = st.empty()
message_placeholder = st.empty()
scores_placeholder = st.empty()


def predict(frame):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten().reshape(1, -1)
            return clf.predict(data)[0]
    return None


def play_turn(player_num):
    bot_num = np.random.randint(1, 11)
    message_placeholder.markdown(f"### 🤖 Bot played: `{bot_num}`")
    if player_num == bot_num:
        st.session_state.out = True
        message_placeholder.markdown(
            f"### 🏏 OUT! Total Score: `{st.session_state.player_score}`"
        )
        st.session_state.batting = "bot"
        time.sleep(2)
        st.session_state.running = False
    else:
        st.session_state.player_score += player_num
    
    update_scoreboard()


def update_scoreboard():
    scores_placeholder.markdown(f"""
    ## 👤 Your Score: `{st.session_state.player_score}`
    ## 🤖 Bot Score: `{st.session_state.bot_score}`
    ### Batting: `{st.session_state.batting}`
    """)


if st.button(" 🎮 Start Game", key="start_btn"):
    st.session_state.running = True
    st.session_state.last_capture_time = time.monotonic()
    st.session_state.player_score = 0
    st.session_state.bot_score = 0
    st.session_state.batting = "player"
    st.session_state.out = False
    update_scoreboard()

stop = st.button("🛑 Stop", key="stop_btn")

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected")
            break

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        elapsed = time.monotonic() - st.session_state.last_capture_time
        remaining = 3 - int(elapsed)

        if remaining > 0:
            message_placeholder.markdown(f"### ⏳ Get ready: {remaining}")
        else:
            message_placeholder.markdown("### 📸 Capturing your move...")
            time.sleep(0.5)
            prediction = predict(frame)
            if prediction:
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                message_placeholder.success(f"You played: {prediction}")
                play_turn(prediction)
                update_scoreboard()
            else:
                message_placeholder.warning("Couldn't detect a hand gesture")

            st.session_state.last_capture_time = time.monotonic()

        if stop or st.session_state.batting == "bot":
            st.session_state.running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# Game Over State
if st.session_state.batting == "bot" and st.button("🔄 Bot Play", key="bot_play"):
    while st.session_state.bot_score <= st.session_state.player_score:
        bot_num = np.random.randint(1, 11)
        player_num = np.random.randint(1, 11)  # Simulated defense
        message_placeholder.markdown(f"Bot: `{bot_num}`, Defense: `{player_num}`")
        time.sleep(1)
        if bot_num == player_num:
            message_placeholder.markdown(
                f"### Bot OUT! Final Score: `{st.session_state.bot_score}`"
            )
            break
        else:
            st.session_state.bot_score += bot_num
            update_scoreboard()
            if st.session_state.bot_score > st.session_state.player_score:
                break

    update_scoreboard()
    if st.session_state.bot_score > st.session_state.player_score:
        st.error("😭 Bot Wins!")
    elif st.session_state.bot_score < st.session_state.player_score:
        st.success("🎉 You Win!")
    else:
        st.warning("🤝 It's a Tie!")

    if st.button("🔄 Play Again"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]