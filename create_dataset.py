import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data1'

data = []
norm_data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    if dir_ == ".DS_Store":
            continue
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        if img_path == ".DS_Store":
            continue

        data_aux = []
        norm_data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    norm_x = x - hand_landmarks.landmark[0].x
                    norm_y = y - hand_landmarks.landmark[0].y

                    data_aux.append(x)
                    data_aux.append(y)

                    norm_data_aux.append(norm_x)
                    norm_data_aux.append(norm_y)

            data.append(data_aux)
            norm_data.append(norm_data_aux)
            labels.append(dir_)

f = open('data1.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

norm_f = open('norm_data1.pickle', 'wb')
pickle.dump({'data': norm_data, 'labels': labels}, norm_f)
norm_f.close()