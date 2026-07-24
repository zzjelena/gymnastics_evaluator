import numpy as np


VISIBILITY_THRESHOLD = 0.5


def calculate_angle(a, b, c):
    """
    Računa ugao između tri 3D tačke.
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return None

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))

    return angle


def point(landmarks, index):
    """
    Vraća koordinatu jednog landmarka.
    """

    return [
        landmarks[index].x,
        landmarks[index].y,
        landmarks[index].z
    ]


def visible(landmarks, *indices):
    """
    Proverava da li su svi potrebni landmarki dovoljno vidljivi.
    """

    for i in indices:
        if landmarks[i].visibility < VISIBILITY_THRESHOLD:
            return False

    return True


def safe_angle(landmarks, a, b, c):

    if not visible(landmarks, a, b, c):
        return None

    return calculate_angle(
        point(landmarks, a),
        point(landmarks, b),
        point(landmarks, c)
    )


def get_all_angles(landmarks):

    angles = {}

    angles["left_knee"] = safe_angle(
        landmarks,
        23, 25, 27
    )

    angles["right_knee"] = safe_angle(
        landmarks,
        24, 26, 28
    )

    angles["left_hip"] = safe_angle(
        landmarks,
        11, 23, 25
    )

    angles["right_hip"] = safe_angle(
        landmarks,
        12, 24, 26
    )

    angles["left_elbow"] = safe_angle(
        landmarks,
        11, 13, 15
    )

    angles["right_elbow"] = safe_angle(
        landmarks,
        12, 14, 16
    )

    angles["left_shoulder"] = safe_angle(
        landmarks,
        13, 11, 23
    )

    angles["right_shoulder"] = safe_angle(
        landmarks,
        14, 12, 24
    )

    return angles