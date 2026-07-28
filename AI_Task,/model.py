# model.py

from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Train the Random Forest AI model.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("AI Model Trained Successfully.\n")

    return model