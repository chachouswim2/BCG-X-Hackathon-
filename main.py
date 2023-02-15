import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")

from features.rule_based_churn import *
from features.clustering_model import *
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
    plt.savefig('RuleBasedModel_output')
    plt.show()
    

    #Clustering model
    print("Step 3- Getting the data ready for the Clustering model")
    print("## Preprocessing the data")
    pre_df = preprocess_df(df)
    print("## Done!")
    print("## Adding new variables to the data")
    df_new_v = add_variables(pre_df)
    print("## Done!")
    print("## Creating the final dataset for the model")
    model_df = model_dataset(df_new_v)
    print("## Done!")

    print("Step 4- Clustering model")
    kmeans_model = create_kmeans_model(model_df)
    print("## Model successfully created!")
    print(" ## Let's look at the clusters")
    plot_kmeans_clusters(model_df, kmeans_model)
    print("## Let's look at the elbow method")
    plot_elbow_method(model_df)
