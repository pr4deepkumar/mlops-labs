import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

# Yearly spending per product category in the Wholesale Customers dataset
SPENDING_COLUMNS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("We are here")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[SPENDING_COLUMNS]

    # Spending is very skewed (the biggest customers spend up to 50x a typical one), so take the log first
    clustering_data = np.log1p(clustering_data)

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # bytes -> base64 string for XCom (the scaler goes along so new data can be scaled the same way)
    clustering_serialized_data = pickle.dumps((clustering_data_minmax, min_max_scaler))
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data and saves it.
    Returns the SSE list (JSON-serializable).
    """
    # decode -> bytes -> numpy array and the fitted scaler
    data_bytes = base64.b64decode(data_b64)
    df, scaler = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    # Refit with the number of clusters picked by the elbow method,
    # instead of saving the last model from the loop (k=49)
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    kmeans = KMeans(n_clusters=kl.elbow, **kmeans_kwargs).fit(df)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump((kmeans, scaler), f)  # save the scaler with the model

    return sse  # list is JSON-safe


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns the first prediction (as a plain int) for test.csv.
    """
    # load the saved model (fitted with the elbow k) and its scaler
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model, scaler = pickle.load(open(output_path, "rb"))

    # elbow for information/logging
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")

    # predict on test data, transformed the same way as the training data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    pred = loaded_model.predict(scaler.transform(np.log1p(df[SPENDING_COLUMNS])))[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        # if not numeric, still return a JSON-friendly version
        return pred.item() if hasattr(pred, "item") else pred


def summarize_clusters(data_b64: str, filename: str):
    """
    Loads the saved model, assigns every customer to a cluster, and prints
    each cluster's size and average yearly spending per category.
    Returns the number of customers in each cluster (JSON-safe dict).
    """
    # decode -> bytes -> DataFrame (the raw data from load_data)
    df = pickle.loads(base64.b64decode(data_b64)).dropna()

    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model, scaler = pickle.load(open(output_path, "rb"))

    df["cluster"] = loaded_model.predict(scaler.transform(np.log1p(df[SPENDING_COLUMNS])))
    print("Average yearly spending per cluster:")
    print(df.groupby("cluster")[SPENDING_COLUMNS].mean().round(0).to_string())

    sizes = df["cluster"].value_counts().sort_index()
    return {int(cluster): int(count) for cluster, count in sizes.items()}
