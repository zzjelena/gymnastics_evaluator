import numpy as np


def calculate_angle(a, b, c):
    """
    Računa ugao ABC
    a = prva tačka
    b = centralna tačka
    c = treća tačka
    """

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


# test primer
hip = [0.4, 0.6, 0]
knee = [0.4, 0.8, 0]
ankle = [0.4, 1.0, 0]

angle = calculate_angle(hip, knee, ankle)

print("Ugao kolena:", angle)