import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def split_num_cat_data(df):
    return df.select_dtypes(include=['number']), df.select_dtypes(exclude=['number'])

def generate_metadata(numerical_df, categorical_df):
    return {
        "numerical_shape": numerical_df.shape,
        "categorical_shape": categorical_df.shape
    }

def save_data(numerical_df, categorical_df, metadata):
    numerical_df.to_csv("numerical_data.csv", index=False)
    categorical_df.to_csv("categorical_data.csv", index=False)
    with open("metadata.txt", "w") as f:
        f.write(str(metadata))

def run(file_path):
    df = load_data(file_path)
    numerical_df, categorical_df = split_num_cat_data(df)
    metadata = generate_metadata(numerical_df, categorical_df)
    save_data(numerical_df, categorical_df, metadata)

run("<-path->.csv")
