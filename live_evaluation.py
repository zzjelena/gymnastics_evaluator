import cv2
import mediapipe as mp

from src.angle_calculator import get_all_angles
from src.stability_checker import StabilityChecker
from src.balance_checker import BalanceChecker
from src.pose_evaluator import PoseEvaluator


# ----------------------------------
# MediaPipe
# ----------------------------------

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ----------------------------------
# Evaluatori
# ----------------------------------

stability = StabilityChecker()
balance = BalanceChecker()
evaluator = PoseEvaluator()


# ----------------------------------
# Učitavanje referentne poze
# ----------------------------------

reference_pose = evaluator.load_pose(
    "data/poses/test_pose.json"
)

if reference_pose is None:
    print("Referentna poza nije pronađena!")
    exit()


# ----------------------------------
# Kamera
# ----------------------------------

cap = cv2.VideoCapture(0)

print("===================================")
print("GYMNASTICS LIVE EVALUATION")
print("ESC - Exit")
print("===================================")


while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    stable = False
    stable_time = 0

    score = None
    report = None

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        # -----------------------------
        # Računanje uglova
        # -----------------------------

        current_angles = get_all_angles(landmarks)

        # -----------------------------
        # Stabilnost uglova
        # -----------------------------

        stable, stable_time = stability.update(current_angles)

        # -----------------------------
        # Stabilnost ravnoteže
        # -----------------------------

        balance_ok = balance.update(landmarks)
        balance_info = balance.debug()

        # -----------------------------
        # Ispis uglova
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
        # Debug balance
        # -----------------------------

        cv2.putText(
            frame,
            f"Hip move: {balance_info['hip']:.3f}",
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Shoulder move: {balance_info['shoulder']:.3f}",
            (20, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Foot move: {balance_info['foot']:.3f}",
            (20, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # -----------------------------
        # Status
        # -----------------------------

        if stable and balance_ok:

            cv2.putText(
                frame,
                "POSE STABLE",
                (20, 340),
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
                f"Score: {score:.2f}",
                (20, 375),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            y = 410

            for joint, data in report.items():

                if data["color"] == "green":
                    color = (0, 255, 0)

                elif data["color"] == "yellow":
                    color = (0, 255, 255)

                else:
                    color = (0, 0, 255)

                text = (
                    f"{joint}: "
                    f"{data['deviation']:.1f} deg   "
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
                "HOLD BALANCE",
                (20, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Stable time: {stable_time:.2f}s",
                (20, 375),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    else:

        cv2.putText(
            frame,
            "No person detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Gymnastics Evaluation",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()