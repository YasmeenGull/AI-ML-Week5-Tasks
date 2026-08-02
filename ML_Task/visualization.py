# visualization.py

import matplotlib.pyplot as plt


def show_accuracy_chart(results_df):
    """
    Display accuracy comparison of all models.
    """

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    plt.title("Model Accuracy Comparison")

    plt.xlabel("Models")

    plt.ylabel("Accuracy")

    plt.ylim(0, 1.1)

    plt.grid(axis="y")

    plt.show()