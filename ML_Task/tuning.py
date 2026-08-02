# tuning.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def tune_model(X_train, y_train):
    """
    Tune Random Forest model using GridSearchCV.
    """

    model = RandomForestClassifier(random_state=42)

    parameters = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": [2, 4]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Model tuning completed successfully.\n")

    print("Best Parameters:")

    print(grid_search.best_params_)

    print()

    print("Best Accuracy:", round(grid_search.best_score_, 3))

    return grid_search.best_estimator_