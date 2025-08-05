import streamlit as st
import cv2
import mediapipe as mp
import time
import joblib
from PIL import Image


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
[data-testid="stColumn"] {
    position: relative;
    z-index: 10;
    max-width: 80rem;
    margin: 0 auto;
    border-radius: .5rem;
    overflow: hidden;
    background-color: rgba(255, 255, 255, 0.9); /* optional soft background */
    box-shadow: 0 0 40px rgba(255, 145, 0, 0.2); /* 🟠 soft orange glow */
    transition: 
        transform 0.2s ease,
        background-color 0.2s ease,
        box-shadow 0.2s ease
}

[data-testid="stColumn"]:hover {
    transform: scale(1.02);
    background-color: rgba(255, 245, 235, 0.3);
    box-shadow: 0 0 50px rgba(255, 120, 0, 0.3);  /* slightly deeper glow */
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

# Layout: Player | Bot
col1, col2 = st.columns([1, 1], gap="large", border=True)

with col1:
    if st.button("Start Game", icon=':material/play_circle:'):
        st.session_state.running = True
        st.session_state.last_capture_time = time.monotonic()

    st.subheader(":orange[:material/person:] Player")
    player_video_placeholder = st.empty()
    player_prediction_placeholder = st.empty()
    player_score_placeholder = st.empty()

    # if st.session_state.last_player_num:
    #     player_prediction_placeholder.markdown(
    #         st.markdown(f"<div class='run-animation'><h4>+{st.session_state.last_player_num} runs!</h4></div>", unsafe_allow_html=True)
    #     )

    player_score_placeholder.metric("Your Score", st.session_state.player_score)

with col2:
    if st.button("Stop Game", icon=':material/stop_circle:'):
        st.session_state.running = False

    st.subheader(":orange[:material/robot_2:] Bot")
    bot_image_placeholder = st.empty()

    # if st.session_state.last_bot_num:
    #     bot_move_placeholder.markdown(
    #         f"<h2 style='color:#FF5733;'>🤖 Bot showed: <b>{st.session_state.last_bot_num}</b></h2>",
    #         unsafe_allow_html=True
    #     )

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