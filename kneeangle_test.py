import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)


    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        # leva noga
        hip = [
            landmarks[23].x,
            landmarks[23].y,
            landmarks[23].z
        ]

        knee = [
            landmarks[25].x,
            landmarks[25].y,
            landmarks[25].z
        ]

        ankle = [
            landmarks[27].x,
            landmarks[27].y,
            landmarks[27].z
        ]


        knee_angle = calculate_angle(
            hip,
            knee,
            ankle
        )


        cv2.putText(
            frame,
            f"Knee angle: {int(knee_angle)}",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


    cv2.imshow(
        "Knee angle test",
        frame
    )


    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()