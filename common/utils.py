import pandas as pd

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