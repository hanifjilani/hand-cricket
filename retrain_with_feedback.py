import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import joblib
from supabase import create_client
import tempfile
from datetime import datetime

# === CONFIG ===
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
BUCKET_NAME = "feedback-images"
MODEL_OUTPUT = "model/handcricket_feedback.pkl"

# === Connect to Supabase ===
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Load original dataset ===
def extract_landmarks(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    return None

X, y = [], []

print("Loading original dataset...")
for label in os.listdir("dataset"):
    for img_file in os.listdir(f"dataset/{label}"):
        path = f"dataset/{label}/{img_file}"
        image = cv2.imread(path)
        features = extract_landmarks(image)
        if features is not None:
            X.append(features)
            y.append(int(label))

# === Load feedback from Supabase ===
print("Fetching feedback records...")
feedback_rows = supabase.table("feedback").select("*").execute().data

for row in feedback_rows:
    correct_label = row.get("correct_label")
    image_filename = row.get("image_filename")
    if not correct_label or not image_filename:
        continue

    # Download feedback image
    print(f"Downloading feedback image: {image_filename}")
    res = supabase.storage.from_(BUCKET_NAME).download(image_filename)
    if res:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(res)
            tmp_path = tmp.name
        image = cv2.imread(tmp_path)
        features = extract_landmarks(image)
        if features is not None:
            X.append(features)
            y.append(int(correct_label))

# === Train new model ===
print(f"Training new model with {len(X)} samples...")
clf = RandomForestClassifier()
clf.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(clf, MODEL_OUTPUT)
print(f"✅ New model saved at {MODEL_OUTPUT}")