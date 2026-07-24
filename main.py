import cv2
import mediapipe as mp

from src.angle_calculator import get_all_angles
from src.stability_checker import StabilityChecker

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

stability = StabilityChecker()

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        if results.pose_world_landmarks:

            landmarks = results.pose_world_landmarks.landmark

            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        angles = get_all_angles(landmarks)

        stable, stable_time = stability.update(angles)

        y = 30

        # Ispis svih uglova
        for name, value in angles.items():

            if value is None:
                text = f"{name}: ---"
            else:
                text = f"{name}: {int(value)}"

            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        y += 25

        # ----------------------------
        # STATUS STABILNOSTI
        # ----------------------------

        status = "MOVING"

        if stable:
            status = "STABLE"

        cv2.putText(
            frame,
            status,
            (20, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if stable else (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Stable: {stable_time:.2f}s",
            (20, y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("Gymnastics Evaluator", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()