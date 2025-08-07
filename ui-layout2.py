import streamlit as st
import cv2
import mediapipe as mp
import time
import joblib
import numpy as np
from PIL import Image

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
/* 🏏 Game Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 600;
    margin-top: 30px;
    font-family: 'Poppins', sans-serif;
}
.orange-text {
    color: rgba(255, 145, 0, 0.8)
}
/* 🎮 Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #444;
    margin-top: -10px;
    margin-bottom: 20px;
}

/* 🧢 Scoreboard */
.scoreboard {
    text-align: center;
    font-size: 22px;
    padding: 8px;
    border-radius: 12px;
    background-color: #ffffffaa;
    margin-top: 10px;
}     
.happy {
    background-color: #ffffffcc; /* semi-transparent white */
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    margin-top: 10px;
    margin-bottom: 20px;
    transition: transform 0.2s;
}
.st-emotion-cache-zbekoy {
    position: relative;
    z-index: 10;
    max-width: 80rem;
    margin: 0 auto;
    border-radius: .5rem;
    overflow: hidden;
    background-color: rgba(255, 255, 255, 0.9); /* optional soft background */
    box-shadow: 0 0 30px rgba(255, 145, 0, 0.3); /* 🟠 soft orange glow */
    transition: 
        transform 0.2s ease,
        background-color 0.2s ease,
        box-shadow 0.2s ease
}

.st-emotion-cache-zbekoy:hover {
    transform: scale(1.02);
    background-color: rgba(255, 245, 235, 0.3);
    box-shadow: 0 0 40px rgba(255, 120, 0, 0.35);  /* slightly deeper glow */
}

.card-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 15px;
    text-align: center;
    color: #333;
}      
</style>

<!-- 🏏 Game Heading -->
<div class="main">
    <h1 class="title">Hand <span class="orange-text">Cricket</span> Game - ML Edition</h1>
    <p class="subtitle">Play your favorite childhood game using the power of AI 🎯</p>
</div>
""", unsafe_allow_html=True)

# Placeholders
overlay_show = st.empty()
overlay_remove = st.empty()

def show_result_screen(result="WIN! 🎉"):
    # Entry + base styles
    overlay_show.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');

        .overlay {{
            visibility: visible;
            opacity: 1;
            pointer-events: auto;
            position: fixed;
            top: 0; left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(255, 255, 255, 0.95);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            animation: fadeIn 0.2s ease-in-out;
        }}

        .result-text {{
            position: relative;
            font-family: 'Bebas Neue', sans-serif;
            font-size: 6rem;
            color: #FFA64E;
            letter-spacing: 4px;
            animation: enterScale 0.6s ease-in-out;
        }}

        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        @keyframes enterScale {{
            0% {{ transform: scale(0.6); opacity: 0; }}
            50% {{ transform: scale(1.05); opacity: 1; }}
            100% {{ transform: scale(1); }}
        }}     
        </style>

        <div class="overlay" id="result-overlay">
            <div class="result-text">{result}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def remove_overlay():
    # Only update CSS to trigger exit animation
    overlay_remove.markdown(
        """
        <style>
        @keyframes exitFade {
            0% { opacity: 1; transform: translateY(0); }
            100% { opacity: 0; transform: translateY(-20px); }
        }
        .overlay {
            animation: exitFade 0.6s ease forwards !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    time.sleep(0.6)
    overlay_remove.markdown(
      """
        <style>
        .overlay {
            visibility: hidden;
            opacity: 0;
            pointer-events: none;
        }
        </style>
        """,
        unsafe_allow_html=True  
    )
    # overlay_show.empty()
    # overlay_remove.empty()


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
    bot_num = np.random.randint(1, 2)
    bot_hand_image = Image.open(f"bot_hands/{2}.png")
    bot_image_placeholder.image(bot_hand_image, caption=f"🤖 Bot showed: {st.session_state.last_bot_num}", width=300)
    st.session_state.last_player_num = player_num
    st.session_state.last_bot_num = bot_num

    if st.session_state.batting == "player":
        if player_num == bot_num:
            st.session_state.out = True
            st.warning(f"🏏 OUT! Final Score: {st.session_state.player_score}")
            # st.session_state.batting = "bot" if st.session_state.batting == "player" else "end"
            show_result_screen("🏏 PLAYER OUT!")
            time.sleep(2.5)
            remove_overlay()
        else:
            st.session_state.player_score += player_num
    else:
        if player_num == bot_num:
            st.session_state.out = True
            if st.session_state.player_score == st.session_state.bot_score:
                st.warning("TIE TIE !!")
                show_result_screen("TIE TIE !!")
                time.sleep(2.5)
                remove_overlay()
            else:
                st.warning("Bot Out! You Win!")
                show_result_screen("YOU WIN! BOT OUT!")
                time.sleep(2.5)
                remove_overlay()
        else:
            st.session_state.bot_score += bot_num

        if st.session_state.player_score < st.session_state.bot_score:
            st.session_state.out = True
            st.warning("Bot Wins! You lost!")
            show_result_screen("Bot Wins! You lost!")
            time.sleep(2.5)
            remove_overlay()

def update_score():
    # if st.session_state.out:
    #     if st.session_state.batting == "player":
    #         player_score_placeholder.metric("Total Runs", value=f"{st.session_state.player_score} Runs",border=True)
    #         st.session_state.batting = "bot"
    #         st.rerun()
    #     else:
    #         bot_score_placeholder.metric("Bot Runs", value=f"{st.session_state.bot_score} / {st.session_state.player_score} Runs", delta=int(st.session_state.last_bot_num),border=True)
    # else:
    #     if st.session_state.batting == "player":
    #         player_score_placeholder.metric("Your Runs", value=f"{st.session_state.player_score} Runs", delta=int(st.session_state.last_player_num),border=True)
    #     else:
    #         bot_score_placeholder.metric("Bot Runs", value=f"{st.session_state.bot_score} / {st.session_state.player_score} Runs", delta=int(st.session_state.last_bot_num),border=True)
        
    if st.session_state.out and st.session_state.batting == "player":
        player_score_placeholder.metric("Total Runs", value=f"{st.session_state.player_score} Runs",border=True)
        st.session_state.batting = "bot"
        st.rerun()
    elif st.session_state.batting == "player":
        player_score_placeholder.metric("Your Runs", value=f"{st.session_state.player_score} Runs", delta=int(st.session_state.last_player_num),border=True)
    else:
        bot_score_placeholder.metric("Bot Runs", value=f"{st.session_state.bot_score} / {st.session_state.player_score} Runs", delta=int(st.session_state.last_bot_num),border=True)
        player_score_placeholder.metric("Your Move", value=f"{st.session_state.last_player_num}",border=True)

    


# Empty space
st.text("")
st.text("")


# .Hero_dashboardContainer__fbpgV {
#     position: relative;
#     z-index: 10;
#     max-width: 80rem;
#     margin: 0 auto;
#     border-radius: .5rem;
#     overflow: hidden;
#     box-shadow: 0 0 50px rgba(248, 180, 4, .2);
# }


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

# Button Layouts: Start Game | Stop Game | Rules
# Icons: 🎮 

# Layout Start Game | Stop Game | Rules | Detection Feedback
left_space, col1_btn, col2_btn, col3_btn, col4_btn, right_space = st.columns(6, gap="small", border=False)

with col1_btn:
    if st.button("Start New Game", icon=':material/play_circle:'):
        st.session_state.running = True
        st.session_state.frame_count = 0
        st.session_state.player_score = 0
        st.session_state.bot_score = 0
        st.session_state.batting = "player"
        st.session_state.out = False
        st.session_state.last_player_num = None
        st.session_state.last_bot_num = None
        st.session_state.last_capture_time = time.monotonic()
with col2_btn:
    if st.button("Stop Game", icon=':material/stop_circle:'):
        st.session_state.running = False
with col3_btn:
    if st.button("Game Rules", icon=":material/gamepad:"):
        st.warning("Work in progress!")
with col4_btn:
    if st.button("Detection Feedback", icon=":material/rate_review:"):
        st.switch_page("pages/feedback.py")

# Empty Space
st.text("")
st.text("")

# Layout: Player | Bot
col1, col2 = st.columns([1, 1], gap="large", border=True)

with col1:
    st.subheader(":orange[:material/person:] Player")
    player_badge = st.empty()
    player_badge.badge(":material/sports_cricket: Batting", color="green")
    player_hand_error = st.empty()
    player_video_placeholder = st.empty()
    player_message_placeholder = st.empty()
    a, b = st.columns(2)
    with a:
        player_score_placeholder = st.empty()
    button_placeholder = st.empty()

with col2:
    st.subheader(":orange[:material/robot_2:] Bot")
    bot_badge = st.empty()
    bot_badge.badge(":material/sports_baseball: Bowling", color="red")
    bot_image_placeholder = st.empty()
    bot_score_placeholder = st.empty()
    c, d = st.columns(2)
    with c:
        bot_score_placeholder = st.empty()
if st.session_state.batting == "bot":
    overlay_image = Image.open(f"overlays/start_bowling.png")
    player_video_placeholder.image(overlay_image)
    if button_placeholder.button("Start Bowling", icon=":material/sports_cricket:"):
        st.session_state.running = True
        st.session_state.frame_count = 0
        st.session_state.bot_score = 0
        st.session_state.out = False
        st.session_state.last_player_num = None
        st.session_state.last_bot_num = None
        st.session_state.last_capture_time = time.monotonic()
        button_placeholder.empty()
        # st.rerun()


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
                player_hand_error.empty()
                play_turn(player_num)
                update_score()
            else:
                player_hand_error.error(":material/hand_gesture: Couldn't detect hand")
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