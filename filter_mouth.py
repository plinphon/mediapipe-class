import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

MASK_COLOR = (0, 0, 255) 
ALPHA = 0.4             

LIPS_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]


with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # ดึงจุด mask
                points = []
                for idx in LIPS_LANDMARKS:
                    lm = face_landmarks.landmark[idx]
                    points.append((int(lm.x * w), int(lm.y * h)))

                points = np.array(points, dtype=np.int32)

                overlay = frame.copy()
                cv2.fillPoly(overlay, [points], MASK_COLOR)

                # ผสมภาพ
                frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)

        cv2.imshow("Mini Project: Face Mask Filter", frame) 

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
