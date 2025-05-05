import os
import sys
import yaml
import joblib
import pandas as pd
import warnings
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config.model_params import PARAM_GRID, RANDOM_SEARCH_PARAMS, RANDOM_STATE
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from common.utils import load_data
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  
mlflow.set_experiment("Model Tracking 2")
# Import model parameters from config
# try:
#     from config.model_params import PARAM_GRID, RANDOM_SEARCH_PARAMS, RANDOM_STATE
# except ImportError:
#     raise ImportError("Could not import model parameters. Please ensure config/model_params.py exists and contains PARAM_GRID, RANDOM_SEARCH_PARAMS, and RANDOM_STATE.")

# Model training
def train_model(X_train, y_train):
    model = LGBMClassifier(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        model, PARAM_GRID,
        n_iter=RANDOM_SEARCH_PARAMS["n_iter"],
        cv=RANDOM_SEARCH_PARAMS["cv"],
        n_jobs=RANDOM_SEARCH_PARAMS["n_jobs"],
        verbose=RANDOM_SEARCH_PARAMS["verbose"],
        scoring=RANDOM_SEARCH_PARAMS["scoring"],
        random_state=RANDOM_SEARCH_PARAMS["random_state"]
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

# Evaluation
def eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Test Accuracy: {acc:.4f}")
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    return acc, y_pred

# Retraining final model
def retrain_model(X_train, y_train, best_params):
    model = LGBMClassifier(random_state=RANDOM_STATE, **best_params)
    model.fit(X_train, y_train)
    return model

# Main flow
def main_training_flow():

    try:
        config_path = os.path.join(project_root, "config", "config.yaml")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        target_column = config["target_column"]
        output_path = config["output_file_path"]
        train_path = os.path.join(output_path, "processed_train.csv")
        test_path = os.path.join(output_path, "processed_test.csv")
        model_output_path = os.path.join("data", "data_artifacts")
        os.makedirs(model_output_path, exist_ok=True)

        print(f"\n Loading data from:\nTrain: {train_path}\nTest: {test_path}")
        train_df = load_data(train_path)
        test_df = load_data(test_path)

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column].astype(int)
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column].astype(int)

        print("\n Training model with hyperparameter tuning...")
        best_model, best_params, best_score = train_model(X_train, y_train)
        print(f"\n Best CV Score: {best_score:.4f}\n Best Params: {best_params}")

        print("\n Evaluating best model...")
        test_acc, _ = eval_model(best_model, X_test, y_test)

        print("\n Retraining model with best parameters on full data...")
        final_model = retrain_model(X_train, y_train, best_params)
        model_path = os.path.join(model_output_path, "lightgbm_model.pkl")
        joblib.dump(final_model, model_path)
        print(f"\n Model saved at: {model_path}")

        # Log metrics to MLflow
        with mlflow.start_run():
            mlflow.log_param("random_state", RANDOM_STATE)
            for param, value in best_params.items():
                mlflow.log_param(param, value)
            mlflow.log_metric("cv_best_score", best_score)
            mlflow.log_metric("test_accuracy", test_acc)
            print(f"\nMetrics logged to MLflow.")

        print("\n" + "="*80)
        print("TRAINING PIPELINE COMPLETED")
        print("="*80)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_training_flow()
