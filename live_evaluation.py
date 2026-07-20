import cv2
import mediapipe as mp

from src.angle_calculator import get_all_angles
from src.stability_checker import StabilityChecker
from src.pose_evaluator import PoseEvaluator


# -----------------------------
# MediaPipe
# -----------------------------

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# Evaluator
# -----------------------------

stability = StabilityChecker()

evaluator = PoseEvaluator()

reference_pose = evaluator.load_pose(
    "data/poses/test_pose.json"
)

if reference_pose is None:
    print("Referentna poza nije pronađena!")
    exit()


# -----------------------------
# Kamera
# -----------------------------

cap = cv2.VideoCapture(0)

print("LIVE EVALUATION")
print("ESC - izlaz")

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    stable = False
    stable_time = 0

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        current_angles = get_all_angles(landmarks)

        stable, stable_time = stability.update(
            current_angles
        )

        # -----------------------------
        # Prikaz uglova
        # -----------------------------

        y = 30

        for name, value in current_angles.items():

            cv2.putText(
                frame,
                f"{name}: {int(value)}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            y += 25

        # -----------------------------
        # Status stabilnosti
        # -----------------------------

        if stable:

            cv2.putText(
                frame,
                f"STABLE ({stable_time:.1f}s)",
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            report, deduction, score = evaluator.evaluate(
                current_angles,
                reference_pose
            )

            cv2.putText(
                frame,
                f"Execution score: {score:.2f}/10",
                (20, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Deduction: {deduction:.2f}",
                (20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            y = 390

            for joint, data in report.items():

                if data["color"] == "green":
                    color = (0, 255, 0)

                elif data["color"] == "yellow":
                    color = (0, 255, 255)

                else:
                    color = (0, 0, 255)

                text = (
                    f"{joint}: "
                    f"{data['deviation']:.1f}°   "
                    f"-{data['deduction']:.2f}"
                )

                cv2.putText(
                    frame,
                    text,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

                y += 24

        else:

            cv2.putText(
                frame,
                "MOVING",
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Stable: {stable_time:.1f}s",
                (20, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    cv2.imshow(
        "Gymnastics Evaluation",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()