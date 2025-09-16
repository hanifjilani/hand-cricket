# 🏏 Hand Cricket 

An interactive, **ML-driven prototype** of a hand cricket game built with **Python & Streamlit**. This project demonstrates real-time gesture recognition, gameplay logic, and **automated ML feedback loops**. It also serves as the foundation for an **upcoming mobile app and desktop game** that will bring hand cricket to a wider audience.

## Table of Contents
<ul>
  <li>Features</li>
<li>Architecture & Workflow</li>
<li>Key Components</li>
<li>Setup & Installation</li>
<li>Usage</li>
<li>CI / Continuous Model Updating</li>
<li>Roadmap</li>
<li>Contributing</li>
<li>License & Contact</li>
</ul>

## Features
<ol>
  <li>🎮 Prototype gameplay for hand cricket, with gestures recognized in real-time.</li>

 <li>🤖 Bot AI opponent that plays dynamically against the user.</li>

 <li>🖼️ Visual overlays (bot hands, game visuals) for immersive experience.</li>

 <li>📊 Data collection pipeline (_data_collect.py_) for gathering training data.</li>

 <li>🧩 Landmark extraction & preprocessing (_extract_landmark_from_dataset.py_) to turn raw data into model-ready features.</li>

 <li>🧠 ML model training & retraining (train_model.py, retrain_with_feedback.py) with feedback integration.</li>

 <li>🔄 Feedback loop: users can flag errors, which get stored and used to improve model accuracy.</li>

 <li>🌐 Streamlit UI with multiple pages for gameplay and feedback collection.</li>

 <li>⚡ Automated retraining via GitHub Actions: feedback stored in Supabase triggers workflows to retrain and redeploy the model.</li>

 <li>📱 Prototype foundation for mobile app & desktop game – this repo is the experimental stage for scaling into production apps.</li>
</ol>
