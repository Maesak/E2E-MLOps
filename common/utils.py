import pandas as pd
import yaml  

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

if __name__ == "__main__":
    try:
        # Loading config.yaml file
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
            print("Config file loaded successfully.")
            
        # Get input file path from config and load the data
        data_path = config["input_file_path"]
        df = load_data(data_path)
        
        if df is not None:
            # Print basic info about the loaded dataframe
            print(f"Data shape: {df.shape}")
            print(f"Column names: {df.columns.tolist()}")
            print(f"First few rows:\n{df.head()}")
    except FileNotFoundError:
        print("Error: config.yaml file not found.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file: {e}")
    except KeyError:
        print("Error: 'input_file_path' key not found in config.")
    except Exception as e:
        print(f"Unexpected error in main execution: {e}")