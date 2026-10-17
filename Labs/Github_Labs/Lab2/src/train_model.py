import mlflow, mlflow.sklearn, datetime, pickle, random
from pathlib import Path
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    
    # Use the timestamp in your script
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    seed = 0
    # Check if the file exists within the folder
    X, y = make_classification(
                            n_samples=random.randint(50, 2000), # changed to 50 to avoid an empty dataset
                            n_features=6,
                            n_informative=3,
                            n_redundant=0,
                            n_repeated=0,
                            n_classes=2,
                            random_state=seed,
                            shuffle=True,
                        )
    # Hold out 20% of the data so evaluate_model.py scores the model on data it never saw
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    Path('data').mkdir(parents=True, exist_ok=True)
    with open(Path('data') / 'data.pickle', 'wb') as data:
        pickle.dump(X_test, data)

    with open(Path('data') / 'target.pickle', 'wb') as data:
        pickle.dump(y_test, data)
            
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Reuters Corpus Volume"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"    
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id,
                        run_name= f"{dataset_name}"):
        
        params = {
                    "dataset_name": dataset_name,
                    "number of datapoint": X.shape[0],
                    "number of dimensions": X.shape[1],
                    "random_seed": seed,
                    "calibration_method": "sigmoid"}

        mlflow.log_params(params)


        forest = RandomForestClassifier(random_state=seed)
        mlflow.log_params(forest.get_params())
        # Calibrate the forest so its predicted probabilities match how often each class actually occurs
        model = CalibratedClassifierCV(forest, method='sigmoid', cv=5)
        model.fit(X, y)

        y_predict = model.predict(X)
        mlflow.log_metrics({'Accuracy': accuracy_score(y, y_predict),
                            'F1 Score': f1_score(y, y_predict),
                            'Brier Score': brier_score_loss(y, model.predict_proba(X)[:, 1])})
        mlflow.sklearn.log_model(model, "model")

        Path('models').mkdir(parents=True, exist_ok=True)

        # After retraining the model
        model_version = f'model_{timestamp}'  # Use a timestamp as the version
        model_filename = Path(f'{model_version}_dt_model.joblib')
        dump(model, model_filename)
        