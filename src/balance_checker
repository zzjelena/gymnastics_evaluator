from collections import deque
import numpy as np


class BalanceChecker:

    def __init__(
        self,
        history_size=30,
        hip_threshold=0.02,
        shoulder_threshold=0.025,
        foot_threshold=0.015
    ):

        """
        history_size = broj frejmova

        pragovi su u normalizovanim MediaPipe koordinatama
        """

        self.history = deque(maxlen=history_size)

        self.hip_threshold = hip_threshold
        self.shoulder_threshold = shoulder_threshold
        self.foot_threshold = foot_threshold

    def update(self, landmarks):

        left_hip = landmarks[23]
        right_hip = landmarks[24]

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        # centar kukova

        hip_center = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2,
            (left_hip.z + right_hip.z) / 2
        ])

        # centar ramena

        shoulder_center = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            (left_shoulder.z + right_shoulder.z) / 2
        ])

        # određivanje potpornog stopala

        if left_ankle.y > right_ankle.y:
            support = np.array([
                left_ankle.x,
                left_ankle.y,
                left_ankle.z
            ])
        else:
            support = np.array([
                right_ankle.x,
                right_ankle.y,
                right_ankle.z
            ])

        frame = {

            "hip": hip_center,

            "shoulder": shoulder_center,

            "support": support

        }

        self.history.append(frame)

        if len(self.history) < self.history.maxlen:

            return False

        return self._is_stable()

    def _movement(self, key):

        values = np.array([frame[key] for frame in self.history])

        minimum = np.min(values, axis=0)

        maximum = np.max(values, axis=0)

        return np.linalg.norm(maximum - minimum)

    def _is_stable(self):

        hip_move = self._movement("hip")

        shoulder_move = self._movement("shoulder")

        foot_move = self._movement("support")

        if hip_move > self.hip_threshold:
            return False

        if shoulder_move > self.shoulder_threshold:
            return False

        if foot_move > self.foot_threshold:
            return False

        return True

    def debug(self):

        if len(self.history) < self.history.maxlen:

            return {
                "hip": 0,
                "shoulder": 0,
                "foot": 0
            }

        return {

            "hip": self._movement("hip"),

            "shoulder": self._movement("shoulder"),

            "foot": self._movement("support")

        }