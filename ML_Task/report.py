# report.py


def generate_report(results_df):
    """
    Display and save evaluation report.
    """

    print("\n========== Evaluation Report ==========\n")

    print(results_df)

    results_df.to_csv(
        "evaluation_report.csv",
        index=False
    )

    print("\nEvaluation report saved as evaluation_report.csv")