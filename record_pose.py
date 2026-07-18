import cv2
import mediapipe as mp
import numpy as np
import json
import os
from collections import deque

from src.angle_calculator import get_all_angles
from src.stability_checker import StabilityChecker


# -----------------------------
# MediaPipe setup
# -----------------------------

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# Podešavanja
# -----------------------------

POSE_NAME = "test_pose"

SAVE_PATH = "data/poses/" + POSE_NAME + ".json"


# koliko poslednjih merenja čuvamo
BUFFER_SIZE = 30


angle_buffer = deque(
    maxlen=BUFFER_SIZE
)


stability = StabilityChecker()



# -----------------------------
# Kamera
# -----------------------------

cap = cv2.VideoCapture(0)


print("------------------------------")
print("RECORD MODE")
print("1. Zauzmite pozu")
print("2. Sačekajte STABLE")
print("3. Pritisnite P za čuvanje")
print("ESC - izlaz")
print("------------------------------")



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


        # računanje uglova
        angles = get_all_angles(
            landmarks
        )


        # provera stabilnosti
        stable, stable_time = stability.update(
            angles
        )


        # ako je stabilno, čuvamo merenje
        if stable:

            angle_buffer.append(
                angles.copy()
            )



        y = 30

        for name, value in angles.items():

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
    # Status
    # -----------------------------

    if stable:

        status = "STABLE"

        color = (0,255,0)

    else:

        status = "MOVING"

        color = (0,0,255)



    cv2.putText(
        frame,
        status,
        (20,400),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )


    cv2.putText(
        frame,
        f"Samples: {len(angle_buffer)}/{BUFFER_SIZE}",
        (20,440),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )


    cv2.imshow(
        "Pose Recorder",
        frame
    )



    key = cv2.waitKey(1)



    # -----------------------------
    # ČUVANJE POZE
    # -----------------------------

    if key == ord("p"):


        if len(angle_buffer) < BUFFER_SIZE:

            print(
                "Nema dovoljno stabilnih merenja!"
            )

            continue



        averaged_angles = {}



        # prolaz kroz sve zglobove
        for joint in angle_buffer[0]:


            values = []


            for sample in angle_buffer:

                values.append(
                    sample[joint]
                )


            averaged_angles[joint] = float(
                np.mean(values)
            )



        os.makedirs(
            "data/poses",
            exist_ok=True
        )


        with open(
            SAVE_PATH,
            "w"
        ) as file:


            json.dump(
                averaged_angles,
                file,
                indent=4
            )


        print("----------------")
        print("POSE SAVED")
        print(SAVE_PATH)
        print("----------------")



    if key == 27:
        break



cap.release()
cv2.destroyAllWindows()