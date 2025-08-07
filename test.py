import streamlit as st
import time

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
    time.sleep(0.3)
    overlay_show.empty()
    overlay_remove.empty()

# Streamlit button logic
if st.button("Start"):
    show_result_screen()
    time.sleep(2)
    remove_overlay()