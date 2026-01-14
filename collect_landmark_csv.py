# collect_landmark_csv.py
"""
Collect hand landmarks from webcam using MediaPipe and save as CSV for each class.
Each row: label, x0, y0, x1, y1, ..., x20, y20
"""
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

CLASSES = [chr(ord('A')+i) for i in range(26)]
SAMPLES_PER_CLASS = 10  # You can adjust this
CSV_FILE = 'landmark_data.csv'

mp_hands = mp.solutions.hands


def collect_landmarks():
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
        writer.writerow(header)
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
            for cls in CLASSES:
                print(f"Collecting for class: {cls}")
                count = 0
                while count < SAMPLES_PER_CLASS:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    display = frame.copy()
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        lm = [(l.x, l.y) for l in hand_landmarks.landmark]
                        # Draw landmarks for user feedback
                        mp.solutions.drawing_utils.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        cv2.putText(display, f'Class: {cls}  Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.imshow('Webcam', display)
                        key = cv2.waitKey(1)
                        if key == ord('s'):
                            row = [cls] + [coord for point in lm for coord in point]
                            writer.writerow(row)
                            print(f'Saved sample {count+1} for class {cls}')
                            count += 1
                        elif key == ord('q'):
                            print('Exiting...')
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                    else:
                        cv2.putText(display, f'No hand detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        cv2.imshow('Webcam', display)
                        if cv2.waitKey(1) == ord('q'):
                            print('Exiting...')
                            cap.release()
                            cv2.destroyAllWindows()
                            return
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    collect_landmarks()
