import os, json, random
import joblib
import argparse
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    # Access the timestamp
    timestamp = args.timestamp
    model_version = f'model_{timestamp}_dt_model'  # Use a timestamp as the version
    model_path = f'{model_version}.joblib'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'Model file not found: {os.path.abspath(model_path)}')

    try:
        # Generate evaluation data with the same settings as training
        X, y = make_classification(
                            n_samples=random.randint(50, 2000),  # changed to 50 to avoid an empty dataset
                            n_features=6,
                            n_informative=3,
                            n_redundant=0,
                            n_repeated=0,
                            n_classes=2,
                            random_state=0,
                            shuffle=True,
                        )
    except ValueError as e:
        raise ValueError(f'Failed to generate evaluation data: {e}')

    y_predict = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    metrics = {"F1_Score": f1_score(y, y_predict),
               "Accuracy": accuracy_score(y, y_predict),
               "Precision": precision_score(y, y_predict),
               "Recall": recall_score(y, y_predict),
               "ROC_AUC": roc_auc_score(y, y_proba)}

    # Save metrics to a JSON file in the working directory; the workflow moves it into metrics/
    with open(f'{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
