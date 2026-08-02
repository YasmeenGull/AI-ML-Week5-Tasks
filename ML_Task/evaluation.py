# evaluation.py

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models and compare their performance.
    """

    results = []

    for name, model in models.items():

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(y_test, predictions)

        recall = recall_score(y_test, predictions)

        f1 = f1_score(y_test, predictions)

        results.append({

            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1

        })

    results_df = pd.DataFrame(results)

    return results_df