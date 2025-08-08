import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import joblib
from supabase import create_client
import tempfile

# === CONFIG ===
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
BUCKET_NAME = "feedback-images"
MODEL_OUTPUT = "model/handcricket_feedback.pkl"
BASE_DATASET_FILE = "data/handcricket_landmarks.npz"  # Preprocessed landmarks

# === Connect to Supabase ===
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Helper: Extract landmarks from image ===
def extract_landmarks(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    return None

# === Load preprocessed original dataset ===
if not os.path.exists(BASE_DATASET_FILE):
    raise FileNotFoundError(f"❌ Base dataset file not found: {BASE_DATASET_FILE}")

print(f"📂 Loading base dataset from {BASE_DATASET_FILE}...")
data = np.load(BASE_DATASET_FILE)
X, y = data["X"].tolist(), data["y"].tolist()

print(f"✅ Loaded base dataset with {len(X)} samples")

# === Load feedback from Supabase ===
print("📡 Fetching feedback records from Supabase...")
feedback_rows = supabase.table("feedback").select("*").execute().data

for row in feedback_rows:
    correct_label = row.get("correct_label")
    image_filename = row.get("image_filename")
    if not correct_label or not image_filename:
        continue

    print(f"⬇️  Downloading feedback image: {image_filename}")
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

print(f"📊 Total samples after adding feedback: {len(X)}")

# === Train new model ===
print("🤖 Training new feedback model...")
clf = RandomForestClassifier()
clf.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(clf, MODEL_OUTPUT)
print(f"✅ New feedback model saved at {MODEL_OUTPUT}")