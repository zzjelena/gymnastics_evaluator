import json
import os


class PoseEvaluator:

    def __init__(self, tolerance=10):

        # dozvoljeno odstupanje u stepenima
        self.tolerance = tolerance


    def load_pose(self, filename):

        if not os.path.exists(filename):
            print("Pose file not found!")
            return None

        with open(filename, "r") as file:
            return json.load(file)



    def evaluate(self, current_angles, reference_angles):

        errors = {}

        total_error = 0
        joints = 0


        for joint in reference_angles:

            if joint in current_angles:

                error = abs(
                    current_angles[joint]
                    -
                    reference_angles[joint]
                )

                errors[joint] = error

                total_error += error
                joints += 1



        if joints == 0:
            return 0, errors



        average_error = total_error / joints


        # pretvaranje greške u ocenu

        score = 100 - average_error


        if score < 0:
            score = 0


        return score, errors