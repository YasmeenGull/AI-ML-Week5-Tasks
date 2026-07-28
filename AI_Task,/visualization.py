# visualization.py

import matplotlib.pyplot as plt


def show_visualization(model, X_test, y_test):
    """
    Display Actual vs Predicted Results.
    """

    predictions = model.predict(X_test)

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(len(y_test)),
        y_test,
        marker="o",
        label="Actual"
    )

    plt.plot(
        range(len(predictions)),
        predictions,
        marker="x",
        label="Predicted"
    )

    plt.title("Actual vs Predicted Results")

    plt.xlabel("Test Samples")

    plt.ylabel("Pass (0/1)")

    plt.legend()

    plt.grid(True)

    plt.show()