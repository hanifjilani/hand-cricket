import cv2
import os
import numpy as np
import mediapipe as mp
import joblib

def extract_landmarks(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    return None

X, y = [], []

for label in os.listdir("dataset"):
    for img_file in os.listdir(f"dataset/{label}"):
        path = f"dataset/{label}/{img_file}"
        image = cv2.imread(path)
        features = extract_landmarks(image)
        if features is not None:
            X.append(features)
            y.append(int(label))

X = np.array(X)
y = np.array(y)

# Save preprocessed dataset
os.makedirs("data", exist_ok=True)
np.savez_compressed("data/handcricket_landmarks.npz", X=X, y=y)

print("✅ Landmarks saved to data/handcricket_landmarks.npz")