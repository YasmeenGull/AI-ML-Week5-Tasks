# preprocessing.py

from sklearn.model_selection import train_test_split


def preprocess_data(df):
    """
    Split dataset into training and testing sets.
    """

    X = df[["StudyHours", "SleepHours", "Attendance"]]

    y = df["Pass"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test