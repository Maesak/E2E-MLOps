import pandas as pd
import os
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from common.utils import load_data

def split_num_cat(df):
    numerical = df.select_dtypes(include='number')
    categorical = df.select_dtypes(exclude='number')
    return numerical, categorical

def process_train_data(train_df):
    num_df, cat_df = split_num_cat(train_df)

    # Numerical data processing
    num_imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    num_imputed = num_imputer.fit_transform(num_df)
    num_scaled = scaler.fit_transform(num_imputed)

    # Categorical data processing
    if not cat_df.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_imputed = cat_imputer.fit_transform(cat_df)

        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(cat_imputed)

        # Combine numerical and categorical data
        final = pd.DataFrame(data=pd.np.hstack((num_scaled, cat_encoded)))
    else:
        encoder = None
        cat_imputer = None
        final = pd.DataFrame(data=num_scaled)

    print("Processed training data.")
    return final, num_imputer, scaler, encoder, cat_df.columns.tolist(), cat_imputer

def process_test_data(test_df, num_imputer, scaler, encoder, cat_columns, cat_imputer):
    num_df, cat_df = split_num_cat(test_df)

    # Numerical data processing
    num_imputed = num_imputer.transform(num_df)
    num_scaled = scaler.transform(num_imputed)

    # Categorical data processing
    if encoder and not cat_df.empty:
        # cat_imputer = SimpleImputer(strategy='most_frequent')
        
        cat_imputed = cat_imputer.transform(cat_df[cat_columns])
        # cat_encoded = encoder.transform(cat_imputed)


        final = pd.DataFrame(data=pd.np.hstack((num_scaled, cat_imputed)))
    else:
        final = pd.DataFrame(data=num_scaled)

    print("Processed testing data.")
    return final

def save_processed_data(df, path, name):
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, name)
    df.to_csv(save_path, index=False)
    print(f"Saved {name} to {save_path}")

if __name__ == "__main__":
    try:
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        train_path = config["train_raw_path"]
        test_path = config["test_raw_path"]
        output_path = config["output_file_path"]

        train_df = load_data(train_path)
        test_df = load_data(test_path)

        if train_df is not None and test_df is not None:
            train_processed, num_imputer, scaler, encoder, cat_columns, cat_imputer = process_train_data(train_df)
            test_processed = process_test_data(test_df, num_imputer, scaler, encoder, cat_columns, cat_imputer)

            save_processed_data(train_processed, output_path, "processed_train.csv")
            save_processed_data(test_processed, output_path, "processed_test.csv")

    except Exception as e:
        print(f"An error occurred in the processing script: {e}")
