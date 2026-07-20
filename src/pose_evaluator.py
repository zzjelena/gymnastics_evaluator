import json
import os


class PoseEvaluator:

    def __init__(self):

        pass


    def load_pose(self, filename):

        if not os.path.exists(filename):
            print("Pose file not found.")
            return None

        with open(filename, "r") as f:
            return json.load(f)


    def joint_penalty(self, deviation):
        """
        Vraća kaznu na osnovu odstupanja ugla.
        """

        if deviation <= 5:
            return 0.0

        elif deviation <= 10:
            return 0.10

        elif deviation <= 20:
            return 0.30

        else:
            return 0.50


    def joint_color(self, deviation):

        if deviation <= 5:
            return "green"

        elif deviation <= 10:
            return "yellow"

        else:
            return "red"


    def evaluate(self, current_angles, reference_angles):

        report = {}

        total_deduction = 0


        for joint in reference_angles:

            if joint not in current_angles:
                continue


            deviation = abs(
                current_angles[joint]
                -
                reference_angles[joint]
            )


            deduction = self.joint_penalty(deviation)

            color = self.joint_color(deviation)


            report[joint] = {

                "current": current_angles[joint],

                "reference": reference_angles[joint],

                "deviation": deviation,

                "deduction": deduction,

                "color": color

            }


            total_deduction += deduction


        execution_score = 10 - total_deduction

        if execution_score < 0:
            execution_score = 0


        return report, total_deduction, execution_score