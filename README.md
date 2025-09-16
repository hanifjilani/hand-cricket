# 🏏 Hand Cricket 

An interactive, **ML-driven prototype** of a hand cricket game built with **Python & Streamlit**. This project demonstrates real-time gesture recognition, gameplay logic, and **automated ML feedback loops**. It also serves as the foundation for an **upcoming mobile app and desktop game** that will bring hand cricket to a wider audience.

## Demo
https://github.com/user-attachments/assets/74930745-3bfd-43b9-a63c-5209b698607f


<div align="center">
 <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
 <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
 <img src="https://img.shields.io/badge/MediaPipe-4285F4?logo=google&logoColor=white" />
 <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white"/>
 <img src="https://img.shields.io/badge/Machine%20Learning-Model-green?logo=tensorflow&logoColor=white"/>
 <img src="https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white"/>
 <img src="https://img.shields.io/badge/Supabase-3ECF8E?logo=supabase&logoColor=white"/>
</div>

## ✨ Features

🎮 Prototype gameplay for hand cricket, with gestures recognized in real-time using MediaPipe Hands.

🤖 Bot AI opponent that plays dynamically against the user.

🖼️ Visual overlays (bot hands, game visuals, game results) for immersive experience.

📊 Data collection pipeline `data_collect.py` for gathering training data.

🧩 Landmark extraction & preprocessing `extract_landmark_from_dataset.py` to turn raw data into model-ready features.

🧠 ML model training & retraining `train_model.py, retrain_with_feedback.py` with feedback integration.

🔄 Feedback loop: users can flag errors, which get stored and used to improve model accuracy.

⚡ Automated retraining via **GitHub Actions**: feedback stored in Supabase triggers workflows to retrain and redeploy the model.

## 🏗️ Architecture & Workflow
1. Gameplay / UI `app.py`
   <ul>
     <li>Runs the Streamlit app for live hand cricket.</li>
     <li>Uses the ML model to classify gestures and manage scoring.</li>
   </ul>

2. Feedback Generator `feedback_generator.py`
   <ul>
     <li>Collects user feedback when predictions are wrong.</li>
     <li>Saves feedback into a Supabase database, where it can be reviewed and used to refine the model.</li>
   </ul>

3. Data & Processing
   <ul>
     <li><code>data_collect.py</code>: Gathers raw gesture data from webcam. Used to generate seed data for the ML model.</li>
     <li><code>extract_landmark_from_dataset.py</code>: Extracts hand landmarks using MediaPipe Hands to create a npz file.</li>
   </ul>
4. Model Lifecycle
   <ul>
        <li><code>train_model.py</code>: Trains baseline model.</li>
        <li><code>retrain_with_feedback.py</code>: Integrates fresh user feedback to improve ML model accuracy.</li>
   </ul>
5. CI/CD Automation
   <ul>
     <li><code>Github Actions bot</code>: Monitors Supabase updates.</li>
     <li><code>Github Actions bot</code>: Automatically retrains, validates, and redeploys updated models.</li>
   </ul>

## ⚙️ Setup & Installation

### 🔹 1. Clone the repo
```bash
git clone https://github.com/hanifjilani/hand-cricket.git
cd hand-cricket
```
### 🔹 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
### 🔹 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 🔹 4. Add your Supabase Credentials
```bash
Create .streamlit/secrets.toml file and then add
supabase_url = "https://xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
supabase_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
### 🔹 5. Run the Application
```bash
streamlit run app.py
```
<h2>📄 License</h2>
MIT License © Hanif Jilani <br>
Developed to empower learners, job seekers, and researchers with practical AI-driven tools.
