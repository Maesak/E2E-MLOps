import yaml
from src.data_ingestion import load_data, split_data
from src.data_processing import process_train_data, process_test_data, save_processed_data

def run():
    try:
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        input_file_path = config["input_file_path"]
        output_file_path = config["output_file_path"]
        test_ratio = config["test_ratio"]
        random_state = config["random_state"]

        print(f"File paths loaded from YAML: {input_file_path}, {output_file_path}")

        
        df = load_data(input_file_path)
        if df is not None:
            split_data(df, output_file_path, test_ratio, random_state)

        
        train_df = load_data(f"{output_file_path}/train.csv")
        test_df = load_data(f"{output_file_path}/test.csv")

        if train_df is not None and test_df is not None:
            
            train_processed, num_imputer, scaler, encoder, cat_columns , cat_imputer= process_train_data(train_df)
            test_processed = process_test_data(test_df, num_imputer, scaler, encoder, cat_columns, cat_imputer)

            
            save_processed_data(train_processed, output_file_path, "processed_train.csv")
            save_processed_data(test_processed, output_file_path, "processed_test.csv")

        print("Pipeline completed successfully.")

    except Exception as e:
        print(f"An error occurred during the run: {e}")

if __name__ == "__main__":
    run()
