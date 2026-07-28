# main.py

import os

from data_loader import load_dataset
from preprocessing import preprocess_data
from model import train_model
from predictor import predict_result
from evaluation import evaluate_model
from visualization import show_visualization


def main():

    current_folder = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(current_folder, "dataset.csv")

    # Load Dataset
    df = load_dataset(dataset_path)

    if df is None:
        return

    # Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train Model
    model = train_model(X_train, y_train)

    # Evaluate Model
    evaluate_model(model, X_test, y_test)

    # User Input
    print("\n========== Student Prediction ==========\n")

    study_hours = float(input("Enter Study Hours: "))

    sleep_hours = float(input("Enter Sleep Hours: "))

    attendance = float(input("Enter Attendance (%): "))

    prediction = predict_result(
        model,
        study_hours,
        sleep_hours,
        attendance
    )

    print("\n========== Prediction ==========\n")

    if prediction == 1:
        print("Prediction: Student is likely to PASS.")
    else:
        print("Prediction: Student is likely to FAIL.")

    # Visualization
    show_visualization(model, X_test, y_test)


if __name__ == "__main__":
    main()