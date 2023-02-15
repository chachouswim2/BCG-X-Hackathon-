import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")

from features.rule_based_churn import *
import features.prepare_data as prepare_data


## Set Paths
file_path = 'data/transaction_data.parquet'
preprocessed_file_path = 'data/sub_df.csv'

if __name__ == "__main__":
    #Load data
    print("Step 1- Load the data")

    if os.path.exists(preprocessed_file_path):
        print("## Data is being loaded")
        df = pd.read_csv(preprocessed_file_path)
        print("## Done!")
    else:
        print("## Data is being prepared and loaded")
        data = prepare_data.load_data(file_path)
        df = prepare_data.preprocess_data(data)
        data.to_csv(preprocessed_file_path, index=False)
        print("## Done!")
    
    #Rule Based model
    print("Step 2- Run the rule based model")
    df_grouped = count_orders_per_client(df)
    time_dif = compute_time_difference_between_orders(df_grouped)
    df_mean_time_diff = compute_mean_time_diff(time_dif)
    df_churned = compute_churn_flag(df_mean_time_diff)
    print("## Done!")

    #Plot results
    churn_counts = df_churned['churn_flag'].value_counts()

    ## create the bar plot
    fig, ax = plt.subplots()
    churn_counts.plot(kind='bar', ax=ax, color="#25AA6C")

    ## add the percentage labels
    for patch in ax.patches:
        height = patch.get_height()
        percent = 100 * height / churn_counts.values.sum()
        ax.annotate(f'{percent:.1f}%', (patch.get_x() + patch.get_width() / 2, height), ha='center', va='bottom')

    ## set labels and title
    ax.set_xlabel('Churn Flag')
    ax.set_ylabel('Count of clients')
    ax.set_title('Number of clients with each Churn Flag')
    plt.show()
