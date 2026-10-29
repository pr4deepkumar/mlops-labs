import joblib
from pathlib import Path

# Load the model once when the API starts, instead of on every request
model = joblib.load(Path(__file__).parent.parent / "model" / "wine_model.pkl")

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
        confidence (numpy.ndarray): Probability of the predicted class.
    """
    y_pred = model.predict(X)
    confidence = model.predict_proba(X).max(axis=1)
    return y_pred, confidence
