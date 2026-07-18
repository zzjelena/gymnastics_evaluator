from src.pose_evaluator import PoseEvaluator


evaluator = PoseEvaluator()


reference = evaluator.load_pose(
    "data/poses/test_pose.json"
)


current_pose = {

    "left_knee": 45,
    "right_knee": 170,

    "left_hip": 90,
    "right_hip": 90,

    "left_elbow": 170,
    "right_elbow": 170,

    "left_shoulder": 85,
    "right_shoulder": 85

}



score, errors = evaluator.evaluate(
    current_pose,
    reference
)


print("----------------")
print("SCORE:", round(score,2))
print("----------------")


for joint, error in errors.items():

    print(
        joint,
        "deviation:",
        round(error,2),
        "degrees"
    )