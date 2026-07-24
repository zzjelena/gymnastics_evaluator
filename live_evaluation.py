import cv2
import mediapipe as mp

from src.angle_calculator import get_all_angles
from src.stability_checker import StabilityChecker
from src.balance_checker import BalanceChecker
from src.pose_evaluator import PoseEvaluator


# ----------------------------------
# Izbor poze
# ----------------------------------

POSES = {
    "1": {
        "name": "Passe",
        "file": "data/poses/passe.json"
    },

    "2": {
        "name": "Attitude",
        "file": "data/poses/attitude.json"
    },

    "3": {
        "name": "Vaga",
        "file": "data/poses/vaga.json"
    }
}


print("===================================")
print("GYMNASTICS POSE SELECTION")
print("===================================")
print("1 - Passe")
print("2 - Attitude")
print("3 - Vaga")
print("===================================")


choice = input("Izaberi pozu: ")


if choice not in POSES:

    print("Nepostojeca poza!")
    exit()


selected_pose = POSES[choice]


print(
    f"Izabrana poza: {selected_pose['name']}"
)


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
# Ucitavanje referentne poze
# ----------------------------------

reference_pose = evaluator.load_pose(
    selected_pose["file"]
)


if reference_pose is None:

    print(
        "Referentna poza nije pronadjena:"
    )

    print(
        selected_pose["file"]
    )

    exit()



# ----------------------------------
# Kamera
# ----------------------------------

cap = cv2.VideoCapture(0)


print("===================================")
print("LIVE EVALUATION")
print("ESC - izlaz")
print("===================================")



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
    stable_time = 0



    if results.pose_landmarks:


        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )



        landmarks = results.pose_landmarks.landmark



        # -----------------------------
        # Uglovi
        # -----------------------------

        current_angles = get_all_angles(
            landmarks
        )

        current_angles = {
            key:value
            for key,value in current_angles.items()
            if value is not None
        }


        # -----------------------------
        # Stabilnost
        # -----------------------------

        stable, stable_time = stability.update(
            current_angles
        )



        # -----------------------------
        # Ravnoteza
        # -----------------------------

        balance_ok = balance.update(
            landmarks
        )


        balance_info = balance.debug()



        # -----------------------------
        # Prikaz uglova
        # -----------------------------

        y = 30


    for name, value in current_angles.items():

        if value is None:
            text = f"{name}: N/A"

        else:
            text = f"{name}: {int(value)}"


        cv2.putText(
            frame,
            text,
            (20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,255,0),
            2
        )


        y += 25



        # -----------------------------
        # Debug
        # -----------------------------

        cv2.putText(
            frame,
            f"Pose: {selected_pose['name']}",
            (400,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )


        cv2.putText(
            frame,
            f"Hip: {balance_info['hip']:.3f}",
            (20,260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )


        cv2.putText(
            frame,
            f"Shoulder: {balance_info['shoulder']:.3f}",
            (20,285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )


        cv2.putText(
            frame,
            f"Foot: {balance_info['foot']:.3f}",
            (20,310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )



        # -----------------------------
        # Evaluacija
        # -----------------------------


        if stable and balance_ok:


            cv2.putText(
                frame,
                "POSE STABLE",
                (20,350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )


            report, deduction, score = evaluator.evaluate(
                current_angles,
                reference_pose
            )



            cv2.putText(
                frame,
                f"SCORE: {score:.2f}",
                (20,390),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )



            y = 430


            for joint, data in report.items():


                if data["color"] == "green":

                    color = (0,255,0)


                elif data["color"] == "yellow":

                    color = (0,255,255)


                else:

                    color = (0,0,255)



                cv2.putText(
                    frame,
                    f"{joint}: {data['deviation']:.1f} deg -{data['deduction']:.2f}",
                    (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )


                y += 22



        else:


            cv2.putText(
                frame,
                "HOLD POSE",
                (20,350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )


            cv2.putText(
                frame,
                f"Stable time: {stable_time:.2f}s",
                (20,390),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )



    else:


        cv2.putText(
            frame,
            "No person detected",
            (20,40),
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