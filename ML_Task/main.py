# main.py

import os

from data_loader import load_dataset
from preprocessing import preprocess_data
from models import train_models
from evaluation import evaluate_models
from tuning import tune_model
from visualization import show_accuracy_chart
from report import generate_report


def main():

    current_folder = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(current_folder, "dataset.csv")

    # Load Dataset
    df = load_dataset(dataset_path)

    if df is None:
        return

    # Preprocess Dataset
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train Multiple Models
    models = train_models(X_train, y_train)

    # Evaluate Models
    results = evaluate_models(models, X_test, y_test)

    # Display Report
    generate_report(results)

    # Tune Best Model
    print("\n========== Model Tuning ==========\n")

    best_model = tune_model(X_train, y_train)

    print("\nOptimized Model")

    print(best_model)

    # Show Graph
    show_accuracy_chart(results)


if __name__ == "__main__":
    main()