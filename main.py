import os
import pandas as pd
import features.prepare_data as prepare_data

file_path = 'data/transaction_data.parquet'
preprocessed_file_path = 'data/sub_df.csv'

if os.path.exists(preprocessed_file_path):
    df = pd.read_csv(preprocessed_file_path)
else:
    data = prepare_data.load_data(file_path)
    df = prepare_data.preprocess_data(data)
    data.to_csv(preprocessed_file_path, index=False)