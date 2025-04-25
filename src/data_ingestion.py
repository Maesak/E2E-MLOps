import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

def load_data(input_file_path):
    try:
        df = pd.read_csv(input_file_path)
        print(f"Data loaded successfully from {input_file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at path '{input_file_path}'")
    except pd.errors.ParserError:
        print("Error: Failed to parse CSV file.")
    except Exception as e:
        print(f"Unexpected error while loading data: {e}")
        return None

def split_data(df, output_path, test_ratio, random_state):
    try:
        # Split the data into train and test sets
        train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_state)

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Define paths for the train and test CSV files
        train_path = os.path.join(output_path, "train.csv")
        test_path = os.path.join(output_path, "test.csv")

        # Save the split data to the defined paths
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Train and test data saved to '{output_path}'")
    except Exception as e:
        print(f"Error while splitting/saving the data: {e}")

if __name__ == "__main__":
    try:
        # Loading config.yaml file (make sure the path is correct)
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Reading configurations from YAML
        input_file_path = config["input_file_path"]
        output_file_path = config["output_file_path"]
        test_ratio = config["test_ratio"]
        random_state = config["random_state"]

        # Loading the data
        df = load_data(input_file_path)
        if df is not None:
            # Splitting the data into train and test sets
            split_data(df, output_file_path, test_ratio, random_state)

        print("Ingestion pipeline complete.")
    except Exception as e:
        print(f"Failed to run ingestion script: {e}")
