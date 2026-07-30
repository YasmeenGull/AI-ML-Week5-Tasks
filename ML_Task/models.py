# models.py

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_models(X_train, y_train):
    """
    Train multiple machine learning models.
    """

    models = {

        "Logistic Regression": LogisticRegression(max_iter=500),

        "Decision Tree": DecisionTreeClassifier(random_state=42),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} trained successfully.")

    return trained_models