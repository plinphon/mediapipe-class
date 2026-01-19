import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

#ตาซ้าย
'''
159 (บน)
145 (ล่าง)
33  (ซ้าย)
133 (ขวา)
'''

#ตาขวา
'''
386 (บน)
374 (ล่าง)
362 (ซ้าย)
263 (ขวา)
'''

# EAR threshold
EAR_THRESHOLD = 0.25
blink_count = 0
eye_closed = False

def distance(p1, p2):
    return math.dist(p1, p2)

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Eye landmarks
            left_eye = [
                (int(landmarks[159].x * w), int(landmarks[159].y * h)),
                (int(landmarks[145].x * w), int(landmarks[145].y * h)),
                (int(landmarks[33].x * w),  int(landmarks[33].y * h)),
                (int(landmarks[133].x * w), int(landmarks[133].y * h)),
            ]

            right_eye = [
                (int(landmarks[386].x * w), int(landmarks[386].y * h)),
                (int(landmarks[374].x * w), int(landmarks[374].y * h)),
                (int(landmarks[362].x * w), int(landmarks[362].y * h)),
                (int(landmarks[263].x * w), int(landmarks[263].y * h)),
            ]

            left_EAR = distance(left_eye[0], left_eye[1]) / distance(left_eye[2], left_eye[3])
            right_EAR = distance(right_eye[0], right_eye[1]) / distance(right_eye[2], right_eye[3])

            EAR = (left_EAR + right_EAR) / 2

            # Blink logic
            if EAR < EAR_THRESHOLD and not eye_closed:
                eye_closed = True
            elif EAR >= EAR_THRESHOLD and eye_closed:
                blink_count += 1
                eye_closed = False

            # Draw eye points
            for p in left_eye + right_eye:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            cv2.putText(frame, f"Blink Count: {blink_count}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 255), 2)

            cv2.putText(frame, f"EAR: {EAR:.2f}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 0, 0), 2)

        cv2.imshow("Blink Counter", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
