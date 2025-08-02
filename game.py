import cv2
import mediapipe as mp
import joblib
import numpy as np
import random

clf = joblib.load("model/hand_cricket1.pkl")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

score = 0
cap = cv2.VideoCapture(0)

def predict(frame):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten().reshape(1, -1)
            return clf.predict(data)[0]
    return None

batting = True
player_score = 0
bowler_score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    number = predict(frame)
    if number:
        cv2.putText(frame, f"Detected: {number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Cricket ML", frame)
    key = cv2.waitKey(1)

    if key == ord("p") and number:
        ai = random.randint(1, 10)
        print(f"You: {number} | AI: {ai}")
        if number == ai:
            print("You're OUT!" if batting else "AI is OUT!")
            batting = not batting
            if not batting:
                print(f"Your Score: {player_score}")
            else:
                print(f"AI Score: {bowler_score}")
            if player_score > bowler_score and not batting:
                print("🏆 You Win!")
                break
            elif bowler_score > player_score and batting:
                print("💔 You Lose!")
                break
            elif not batting:
                player_score = 0
                bowler_score = 0
        else:
            if batting:
                player_score += number
                print(f"Player Score: {player_score}")
            else:
                bowler_score += ai
                print(f"AI Score: {bowler_score}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()