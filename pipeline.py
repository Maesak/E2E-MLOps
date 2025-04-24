import os
import pandas as pd
import yaml

def load_data(input_file_path):
    try:
        df = pd.read_csv(input_file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at path '{input_file_path}'")
    except pd.errors.ParserError:
        print("Error: Failed to parse CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")

def split_num_cat_data(df):
    try:
        numerical_df = df.select_dtypes(include=['number'])
        categorical_df = df.select_dtypes(exclude=['number'])
        print("Data split into numerical and categorical successfully.")
        return numerical_df, categorical_df
    except Exception as e:
        print(f"An error occurred while splitting the data: {e}")

def generate_metadata(numerical_df, categorical_df):
    try:
        metadata = {
            "numerical_shape": numerical_df.shape,
            "categorical_shape": categorical_df.shape
        }
        print("Metadata generated successfully.")
        return metadata
    except Exception as e:
        print(f"An error occurred while generating metadata: {e}")

def save_data(numerical_df, categorical_df, metadata, output_file_path):
    try:
        os.makedirs(output_file_path, exist_ok=True)  # Create dir if it doesn't exist

        numerical_df.to_csv(os.path.join(output_file_path, "numerical_data.csv"), index=False)
        categorical_df.to_csv(os.path.join(output_file_path, "categorical_data.csv"), index=False)

        with open(os.path.join(output_file_path, "metadata.txt"), "w") as f:
            f.write(str(metadata))

        print("Data and metadata saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving data: {e}")

def run():
    try:
        with open("congif/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        input_file_path = config["input_file_path"]
        output_file_path = config["output_file_path"]
        print(f"File path loaded from YAML: {input_file_path}")
        print(f"Save path loaded from YAML: {output_file_path}")

        df = load_data(input_file_path)
        if df is not None:
            numerical_df, categorical_df = split_num_cat_data(df)
            if numerical_df is not None and categorical_df is not None:
                metadata = generate_metadata(numerical_df, categorical_df)
                if metadata is not None:
                    save_data(numerical_df, categorical_df, metadata, output_file_path)
    except Exception as e:
        print(f"An error occurred during the run: {e}")

run()
