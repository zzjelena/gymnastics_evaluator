from collections import deque
import time


class StabilityChecker:

    def __init__(
        self,
        angle_threshold=8.0,
        required_time=1.0,
        history_size=30
    ):
        """
        angle_threshold - maksimalna dozvoljena promena ugla (stepeni)

        required_time - koliko dugo poza mora biti stabilna

        history_size - koliko frejmova čuvamo
        """

        self.angle_threshold = angle_threshold
        self.required_time = required_time
        self.history_size = history_size

        self.history = deque(maxlen=history_size)

        self.stable_since = None

    def update(self, current_angles):

        # izbacujemo None uglove
        current_angles = {
            key: value
            for key, value in current_angles.items()
            if value is not None
        }

        if len(current_angles) == 0:

            self.history.clear()
            self.stable_since = None

            return False, 0.0

        # dodaj novi frejm
        self.history.append(current_angles.copy())

        # dok nemamo dovoljno frejmova
        if len(self.history) < self.history_size:

            self.stable_since = None
            return False, 0.0

        stable = True

        joints = self.history[0].keys()

        for joint in joints:

            values = []

            for frame in self.history:

                if joint in frame:

                    values.append(frame[joint])

            if len(values) < 2:
                continue

            movement = max(values) - min(values)

            if movement > self.angle_threshold:

                stable = False
                break

        if stable:

            if self.stable_since is None:

                self.stable_since = time.time()

            stable_time = time.time() - self.stable_since

            if stable_time >= self.required_time:

                return True, stable_time

            return False, stable_time

        self.stable_since = None

        return False, 0.0