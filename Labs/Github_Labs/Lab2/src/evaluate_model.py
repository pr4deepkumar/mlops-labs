import os, json, pickle, sys
import joblib
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    parser.add_argument("--min-f1", type=float, default=0.7, help="Fail if the model's F1 score is below this")
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
        # Load the held-out test data saved by train_model.py
        with open('data/data.pickle', 'rb') as data:
            X = pickle.load(data)

        with open('data/target.pickle', 'rb') as data:
            y = pickle.load(data)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Test data not found, run train_model.py first: {e}')

    y_predict = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    metrics = {"F1_Score": f1_score(y, y_predict),
               "Accuracy": accuracy_score(y, y_predict),
               "Precision": precision_score(y, y_predict),
               "Recall": recall_score(y, y_predict),
               "ROC_AUC": roc_auc_score(y, y_proba),
               "Brier_Score": brier_score_loss(y, y_proba)}

    # Save metrics to a JSON file in the working directory; the workflow moves it into metrics/
    with open(f'{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    # Quality gate: a non-zero exit fails the workflow, so a weak model is never committed
    if metrics["F1_Score"] < args.min_f1:
        print(f'Quality gate failed: F1 {metrics["F1_Score"]:.3f} is below {args.min_f1}')
        sys.exit(1)
    print(f'Quality gate passed: F1 {metrics["F1_Score"]:.3f} (minimum {args.min_f1})')
