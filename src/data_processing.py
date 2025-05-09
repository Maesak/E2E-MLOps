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

def process_train_data(train_df, target_col):
    target = train_df[target_col].copy()
    features_df = train_df.drop(columns=[target_col])
    
    num_df, cat_df = split_num_cat(features_df)  
    
    # Store original column names
    num_cols = num_df.columns.tolist()
    
    # Numerical data processing
    num_imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    num_imputed = num_imputer.fit_transform(num_df)
    num_scaled = scaler.fit_transform(num_imputed)
    
    num_processed_df = pd.DataFrame(num_scaled, columns=num_cols)
    
    if not cat_df.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_imputed = cat_imputer.fit_transform(cat_df)
        
        cat_cols = cat_df.columns.tolist()
        
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(cat_imputed)

        encoded_feature_names = []
        for i, col in enumerate(cat_cols):
            categories = encoder.categories_[i]
            for cat in categories:
                encoded_feature_names.append(f"{col}_{cat}")
        
        cat_processed_df = pd.DataFrame(cat_encoded, columns=encoded_feature_names)
        
        processed_df = pd.concat([num_processed_df, cat_processed_df], axis=1)
    else:
        encoder = None
        cat_imputer = None
        cat_cols = []
        processed_df = num_processed_df
    
    final_df = pd.concat([processed_df, pd.DataFrame({target_col: target.values})], axis=1)
    
    print("Processed training data with preserved column names.")
    return final_df, num_imputer, scaler, encoder, cat_cols, cat_imputer

def process_test_data(test_df, num_imputer, scaler, encoder, cat_columns, cat_imputer, target_col):
    target = test_df[target_col].copy()
    features_df = test_df.drop(columns=[target_col])
    
    # Split into numerical and categorical features
    num_df, cat_df = split_num_cat(features_df)
    
    # Store original column names
    num_cols = num_df.columns.tolist()
    
    # Numerical data processing
    num_imputed = num_imputer.transform(num_df)
    num_scaled = scaler.transform(num_imputed)
    
    num_processed_df = pd.DataFrame(num_scaled, columns=num_cols)
    
    if encoder and not cat_df.empty:
        cat_df_selected = cat_df[cat_columns]
        cat_imputed = cat_imputer.transform(cat_df_selected)
        cat_encoded = encoder.transform(cat_imputed)
        
        encoded_feature_names = []
        for i, col in enumerate(cat_columns):
            categories = encoder.categories_[i]
            for cat in categories:
                encoded_feature_names.append(f"{col}_{cat}")
        
        cat_processed_df = pd.DataFrame(cat_encoded, columns=encoded_feature_names)
        processed_df = pd.concat([num_processed_df, cat_processed_df], axis=1)
    else:
        processed_df = num_processed_df
    
    final_df = pd.concat([processed_df, pd.DataFrame({target_col: target.values})], axis=1)
    
    print("Processed testing data with preserved column names.")
    return final_df

def save_processed_data(df, path, name):
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, name)
    df.to_csv(save_path, index=False)
    print(f"Saved {name} to {save_path} with {len(df.columns)} columns")

if __name__ == "__main__":
    try:
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        train_path = config["train_raw_path"]
        test_path = config["test_raw_path"]
        output_path = config["output_file_path"]
        target_column = config["target_column"]

        train_df = load_data(train_path)
        test_df = load_data(test_path)

        if train_df is not None and test_df is not None:
            train_processed, num_imputer, scaler, encoder, cat_columns, cat_imputer = process_train_data(train_df, target_column)
            test_processed = process_test_data(test_df, num_imputer, scaler, encoder, cat_columns, cat_imputer, target_column)

            save_processed_data(train_processed, output_path, "processed_train.csv")
            save_processed_data(test_processed, output_path, "processed_test.csv")

    except Exception as e:
        print(f"An error occurred in the processing script: {e}")
        import traceback
        traceback.print_exc()