import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        finger_count = 0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm = hand.landmark

            # นิ้วชี้ กลาง นาง ก้อย
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]

            for tip, pip in zip(tips, pips):
                if lm[tip].y < lm[pip].y:
                    finger_count += 1

            # Thumb (แกน x)
            if lm[4].x > lm[3].x:
                finger_count += 1

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Fingers: {finger_count}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3)

        cv2.imshow("Finger Counter", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
