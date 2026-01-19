import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  #iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        style = mp_drawing.DrawingSpec(
                color=(0, 255, 0),  
                thickness=1
            )

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=style
                )
            h, w, _ = frame.shape
            for idx, lm in enumerate(face_landmarks.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.putText(frame, str(idx), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (0, 0, 255), 1)

        cv2.imshow("MediaPipe Face Mesh (468 landmarks)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
