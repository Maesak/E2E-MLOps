# LightGBM parameter grid for RandomizedSearchCV
PARAM_GRID = {
    "n_estimators": [50, 100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "num_leaves": [20, 31, 50, 70],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

# Configuration for RandomizedSearchCV
RANDOM_SEARCH_PARAMS = {
    "n_iter": 10,        # Number of parameter settings sampled
    "cv": 3,             # Cross-validation folds
    "n_jobs": -1,        # Use all available processors
    "verbose": 1,        # Output progress
    "random_state": 42,  # For reproducibility
    "scoring": "accuracy" # Metric to optimize
}

# Random state for reproducibility
RANDOM_STATE = 42