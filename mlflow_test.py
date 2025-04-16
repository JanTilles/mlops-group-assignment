import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris_models")

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define an input example for MLflow logging
input_example = np.array([X_train[0]])

# Define the models and their hyperparameters
models = {
    "LogisticRegression": {
        "model": LogisticRegression(),
        "params": {
            "C": [0.1, 1, 10],
            "solver": ["liblinear"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [10, 50, 100],
            "max_depth": [None, 10, 20]
        }
    },
    "SVM": {
        "model": SVC(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    }
}

# Start an MLflow run
with mlflow.start_run():
    best_model = None
    best_score = 0
    best_params = None

    for model_name, model_info in models.items():
        clf = GridSearchCV(model_info["model"], model_info["params"], cv=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters, metrics, and the model with MLflow
        mlflow.log_param(f"{model_name}_params", clf.best_params_)
        mlflow.log_metric(f"{model_name}_accuracy", accuracy)
        mlflow.sklearn.log_model(sk_model=clf.best_estimator_, artifact_path=f"{model_name}_model", input_example=input_example)

        print(f"{model_name} accuracy: {accuracy}")

        if accuracy > best_score:
            best_model = model_name
            best_score = accuracy
            best_params = clf.best_params_

    print(f"Best model: {best_model} with accuracy: {best_score} and params: {best_params}")

# To view results in MLflow UI, run:
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
