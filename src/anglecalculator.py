import numpy as np


def calculate_angle(a, b, c):
    """
    Računa ugao između tri tačke.

    a - prva tačka
    b - centralna tačka
    c - treća tačka
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return 0

    cosine = np.dot(ba, bc) / denominator

    # zaštita od numeričkih grešaka
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.arccos(cosine)

    return np.degrees(angle)


def point(landmarks, index):
    """
    Vraća koordinatu jednog landmarka.
    """

    return [
        landmarks[index].x,
        landmarks[index].y,
        landmarks[index].z
    ]


def get_all_angles(landmarks):

    angles = {}

    # leva noga
    angles["left_knee"] = calculate_angle(
        point(landmarks, 23),
        point(landmarks, 25),
        point(landmarks, 27)
    )

    # desna noga
    angles["right_knee"] = calculate_angle(
        point(landmarks, 24),
        point(landmarks, 26),
        point(landmarks, 28)
    )

    # levi kuk
    angles["left_hip"] = calculate_angle(
        point(landmarks, 11),
        point(landmarks, 23),
        point(landmarks, 25)
    )

    # desni kuk
    angles["right_hip"] = calculate_angle(
        point(landmarks, 12),
        point(landmarks, 24),
        point(landmarks, 26)
    )

    # levi lakat
    angles["left_elbow"] = calculate_angle(
        point(landmarks, 11),
        point(landmarks, 13),
        point(landmarks, 15)
    )

    # desni lakat
    angles["right_elbow"] = calculate_angle(
        point(landmarks, 12),
        point(landmarks, 14),
        point(landmarks, 16)
    )

    # levo rame
    angles["left_shoulder"] = calculate_angle(
        point(landmarks, 13),
        point(landmarks, 11),
        point(landmarks, 23)
    )

    # desno rame
    angles["right_shoulder"] = calculate_angle(
        point(landmarks, 14),
        point(landmarks, 12),
        point(landmarks, 24)
    )

    return angles