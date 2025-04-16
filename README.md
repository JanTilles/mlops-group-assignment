# MLOps with MLflow and Scikit-learn

This repository demonstrates the use of **MLflow** for tracking machine learning experiments with models built using **Scikit-learn**. The project includes a Jupyter Notebook for training and evaluating models on the Iris dataset.

## Features
- Tracks experiments using MLflow.
- Logs model parameters, metrics, and artifacts.
- Supports multiple models: Logistic Regression, Random Forest, and Support Vector Machine (SVM).
- Uses `GridSearchCV` for hyperparameter tuning.
- Provides an example of setting up an MLflow server.

## Repository Structure
```
MLOps/
├── mlflow_notebook.ipynb   # Jupyter Notebook for MLflow experiments
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
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

### 2. Run the Notebook
Open the `mlflow_notebook.ipynb` file in Jupyter Notebook and execute the cells to:
- Train models on the Iris dataset.
- Log parameters, metrics, and models to MLflow.

### 3. View Results in MLflow UI
Access the MLflow UI at `http://127.0.0.1:5000` to view experiment results.

## Example Output
- **Experiment Name**: `iris_models_experiment`
- **Best Model**: Displays the best model with its accuracy and hyperparameters.

## Notes
- The `mlruns/` folder is excluded from version control using `.gitignore`.
- To remove the `mlruns/` folder from the remote repository, follow these steps:
  ```bash
  git rm -r --cached mlruns
  git commit -m "Remove mlruns folder from the repository"
  git push origin <branch-name>
  ```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
