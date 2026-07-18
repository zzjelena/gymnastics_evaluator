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


# učitavanje referentne poze
reference_pose = evaluator.load_pose(
    "data/poses/test_pose.json"
)


if reference_pose is None:
    print("Nema referentne poze!")
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



    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = pose.process(rgb)



    stable = False



    if results.pose_landmarks:


        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        landmarks = results.pose_landmarks.landmark


        # računanje trenutnih uglova

        current_angles = get_all_angles(
            landmarks
        )


        # provera stabilnosti

        stable, stable_time = stability.update(
            current_angles
        )



        # prikaz uglova

        y = 30

        for name, value in current_angles.items():

            cv2.putText(
                frame,
                f"{name}: {int(value)}",
                (20,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0,255,0),
                2
            )

            y += 25



        # -----------------------------
        # Evaluacija samo kada miruje
        # -----------------------------

        if stable:


            score, errors = evaluator.evaluate(
                current_angles,
                reference_pose
            )


            cv2.putText(
                frame,
                f"SCORE: {int(score)}",
                (20,400),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )


            # najveća odstupanja

            y_error = 440


            for joint, error in errors.items():


                if error > 10:

                    color = (0,0,255)

                else:

                    color = (0,255,0)



                cv2.putText(

                    frame,

                    f"{joint}: {int(error)} deg",

                    (20,y_error),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    color,

                    2
                )


                y_error += 22



        else:


            cv2.putText(
                frame,
                "MOVING - HOLD POSE",
                (20,400),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
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