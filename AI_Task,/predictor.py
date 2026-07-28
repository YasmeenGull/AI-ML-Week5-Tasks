# predictor.py

import pandas as pd


def predict_result(model, study_hours, sleep_hours, attendance):
    """
    Predict whether the student will pass.
    """

    input_data = pd.DataFrame({
        "StudyHours": [study_hours],
        "SleepHours": [sleep_hours],
        "Attendance": [attendance]
    })

    prediction = model.predict(input_data)

    return prediction[0]