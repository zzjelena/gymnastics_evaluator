import time


class StabilityChecker:
    def __init__(self, angle_threshold=8.0, required_time=1.0):
        """
        angle_threshold - maksimalna dozvoljena promena ugla (u stepenima)
        required_time - koliko dugo osoba mora biti mirna
        """

        self.angle_threshold = angle_threshold
        self.required_time = required_time

        self.previous_angles = None
        self.stable_since = None

    def update(self, current_angles):
        """
        current_angles = dictionary svih uglova

        Vraća:
        stable (bool)
        stable_time (float)
        """

        # prvi frejm
        if self.previous_angles is None:

            self.previous_angles = current_angles.copy()
            self.stable_since = None

            return False, 0.0

        # najveća promena ugla između dva frejma
        max_difference = 0

        for key in current_angles:

            difference = abs(
                current_angles[key] -
                self.previous_angles[key]
            )

            if difference > max_difference:
                max_difference = difference

        # ažuriramo prethodne uglove
        self.previous_angles = current_angles.copy()

        # osoba se pomera
        if max_difference > self.angle_threshold:

            self.stable_since = None

            return False, 0.0

        # prvi miran frejm
        if self.stable_since is None:

            self.stable_since = time.time()

            return False, 0.0

        # koliko je dugo mirna
        stable_time = time.time() - self.stable_since

        if stable_time >= self.required_time:

            return True, stable_time

        return False, stable_time