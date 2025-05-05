import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from common.utils import load_data

def split_data(df, output_path, test_ratio, random_state):
    try:
        train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_state)

        os.makedirs(output_path, exist_ok=True)

        train_path = os.path.join(output_path, "train.csv")
        test_path = os.path.join(output_path, "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Train and test data saved to '{output_path}'")
    except Exception as e:
        print(f"Error while splitting/saving the data: {e}")

if __name__ == "__main__":
    try:
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        input_file_path = config["input_file_path"]
        output_file_path = config["output_file_path"]
        test_ratio = config["test_ratio"]
        random_state = config["random_state"]
        target_column = config["target_column"]  # Added target column reference

        df = load_data(input_file_path)
        if df is not None:
            # Verify that target column exists
            if target_column in df.columns:
                print(f"Target column '{target_column}' found in the dataset")
                split_data(df, output_file_path, test_ratio, random_state)
            else:
                print(f"Error: Target column '{target_column}' not found in dataset")
                print(f"Available columns: {df.columns.tolist()}")

        print("Ingestion pipeline complete.")
    except Exception as e:
        print(f"Failed to run ingestion script: {e}")