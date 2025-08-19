import pickle
import cv2
import mediapipe as mp
import numpy as np

model_file = './models/model2.pickle'

model_dict = pickle.load(open(model_file, 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

labels_dict = {0: "closed", 1: "previous", 2: "next", 3: "pointer", 4: "drawer", 5: "erase"}

ESC_key = 27

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - hand_landmarks.landmark[0].x)
                data_aux.append(y - hand_landmarks.landmark[0].y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W) - 30
        y1 = int(min(y_) * H) - 30

        x2 = int(max(x_) * W) + 30
        y2 = int(max(y_) * H) + 30

        probs = model.predict_proba([np.asarray(data_aux)])
        confidence = np.max(probs)
        prediction = np.argmax(probs)

        predicted_gesture = labels_dict[int(prediction)]

        

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 4)
        cv2.putText(frame, f'{predicted_gesture} ({confidence*100:.2f}%)', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ESC_key:
        break


cap.release()
cv2.destroyAllWindows()

