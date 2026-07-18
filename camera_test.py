import cv2
import mediapipe as mp

# MediaPipe podešavanje
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Kamera nije dostupna")
        break

    # OpenCV daje BGR, MediaPipe očekuje RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    # Ako je telo pronađeno
    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # prikaz
    cv2.imshow("Gymnastics Pose Detection", frame)

    # ESC za izlaz
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()