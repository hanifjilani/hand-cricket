# 🏏 Hand Cricket 

An interactive, **ML-driven prototype** of a hand cricket game built with **Python & Streamlit**. This project demonstrates real-time gesture recognition, gameplay logic, and **automated ML feedback loops**. It also serves as the foundation for an **upcoming mobile app and desktop game** that will bring hand cricket to a wider audience.

## Table of Contents
<ul>
  <a href="#features"><li>Features</li></a>
<li>Architecture & Workflow</li>
<li>Key Components</li>
<li>Setup & Installation</li>
<li>Usage</li>
<li>CI / Continuous Model Updating</li>
<li>Roadmap</li>
<li>Contributing</li>
<li>License & Contact</li>
</ul>

## ✨ Features

🎮 Prototype gameplay for hand cricket, with gestures recognized in real-time.

🤖 Bot AI opponent that plays dynamically against the user.

🖼️ Visual overlays (bot hands, game visuals, game results) for immersive experience.

📊 Data collection pipeline **_(data_collect.py)_** for gathering training data.

🧩 Landmark extraction & preprocessing **_(extract_landmark_from_dataset.py)_** to turn raw data into model-ready features.

🧠 ML model training & retraining **_(train_model.py, retrain_with_feedback.py)_** with feedback integration.

🔄 Feedback loop: users can flag errors, which get stored and used to improve model accuracy.

⚡ Automated retraining via **GitHub Actions**: feedback stored in Supabase triggers workflows to retrain and redeploy the model.

## 🏗️ Architecture & Workflow
### Gameplay / UI (`app.py`)

Runs the Streamlit app for live hand cricket.

Uses the ML model to classify gestures and manage scoring.
