# MLOps with MLflow and Scikit-learn

This repository demonstrates the use of **MLflow** for tracking machine learning experiments with models built using **Scikit-learn**. The project includes a Jupyter Notebook for training and evaluating models on the Iris dataset.

## Features
- Tracks experiments using MLflow.
- Logs model parameters, metrics, and artifacts.
- Supports multiple models: Logistic Regression, Random Forest, DecisionTree and Support Vector Machine (SVM).
- Uses `GridSearchCV` for hyperparameter tuning.
- Publishes a mlflow model from DecicisionTree experiment.
- Loads a model from MLflow and makes predictions with it.
- Provides an example of setting up an MLflow server.

## Repository Structure
```
MLOps/
├── train_model.ipynb       # Jupyter Notebook for MLflow experiments, training different models with GridSearchCV
├── use_model.ipynb         # Jupyter Notebook for loading model from MLflow and using it
├── .gitignore              # Gitignore file to exclude unnecessary files
├── README.md               # Documentation for the repository
├── requirements.txt        # Dependencies for the project
└── mlruns/                 # Folder for MLflow runs (ignored in .gitignore)
```

## Prerequisites
- Python 3.8 or higher
- Required Python libraries:

  - `numpy`
  - `scikit-learn`
  - `mlflow`

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Start the MLflow Server
Run the following command to start the MLflow server:
```bash
mlflow server --host 127.0.0.1 --port 5000
```

### 2. Run the Notebook
Open the `train_model.ipynb` file in Jupyter Notebook and execute the cells to:
- Train models on the Iris dataset.
- Log parameters, metrics, and models to MLflow.

### 3. View Results in MLflow UI
Access the MLflow UI at `http://127.0.0.1:5000` to view experiment results.

### 4. Run second Notebook
Open the `use_model.ipynb` file in Jupyter Notebook and execute the cells to:
- Load model from mlflow
- Use the model to make predictions on Iris dataset

## Example Output In MlFlow
- **Experiment Name**: `iris_models_experiment`
- **Model Name**: 'iris_model'

## Notes
- The `mlruns/` folder is excluded from version control using `.gitignore`.

