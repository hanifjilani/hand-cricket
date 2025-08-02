# Description: saves the video capture frame to dataset folder each time you press s"

import cv2
import os

label = "10"  # Change this for each gesture
save_dir = f"dataset/{label}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
i = 0
while i < 200:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f"{save_dir}/{i}.jpg", frame)
        i += 1
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()