import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# for data preprocessing and clustering
from sklearn.cluster import KMeans

def preprocess_df(df):
    """
    Preprocesses the input data by dropping null values and duplicates, adding columns for negative and positive
    net sales, and converting date columns to datetime objects.

    Parameters:
    df (DataFrame): The input DataFrame containing the transaction data.

    Returns:
    DataFrame: A DataFrame containing preprocessed transaction data.
    """
    # Drop null values and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Add columns with only negative and positive net sales
    df['retours'] = df['sales_net'].apply(lambda x: (x <= 0) * x)
    df['sales'] = df['sales_net'].apply(lambda x: (x > 0) * x)

    # Convert date columns to datetime objects
    df['date_order'] = pd.to_datetime(df['date_order'])
    df['date_invoice'] = pd.to_datetime(df['date_invoice'])
    df["date_order"] = df["date_order"].apply(lambda x: x.date())
    df["date_invoice"] = df["date_invoice"].apply(lambda x: x.date())

    # Aggregate data by customer ID
    customers = df.groupby(['client_id']).agg(list)
    customers.reset_index(inplace=True)

    return customers


# Function to add the percentage of each channels per clients to the dataset (optionnal)
def calcul_percentage(col, value):
    col = list(col)
    new_col = []
    for i in range(len(col)):
        new_col.append(col[i].count(value)/len(col[i]))
    return pd.DataFrame(new_col, columns = [value])


def add_variables(df):
     """
    Add variables to the the input DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing the transaction data.

    Returns:
    DataFrame: A DataFrame containing the new engineered features.
    """
     # Create variables for clustering
     df["nb_order"] = df["date_order"].apply(len)  # Number of orders per client
     df["total"] = df["sales_net"].apply(sum)  # Total sales net per client
     df["retour"] = df["retours"].apply(lambda x: abs(sum(x)))  # Returns per client
     df["pos_sales"] = df["sales"].apply(sum)  # Total positive sales per client
     df['diff_prod'] = df["product_id"].apply(lambda x: len(set(x)))  # Variety of products per client
     df['mean_quantity'] = df["quantity"].apply(lambda x: np.mean(x))  # Mean quantity per order
     df = pd.concat([df, calcul_percentage(df['order_channel'], "by phone"),
                              calcul_percentage(df['order_channel'], "at the store"),
                              calcul_percentage(df['order_channel'], "online")], axis=1)  # Percentage of each channel

     # Select relevant features for clustering
     df = df[['nb_order', 'pos_sales', 'mean_quantity', 'retour', 'diff_prod']]

     return df

def apply_log1p_transformation(dataframe, column):
    """
    Applies numpy log1p transformation to a column in the input dataframe and returns the transformed column as a pandas series.

    Parameters:
    dataframe (DataFrame): The input dataframe containing the column to transform.
    column (str): The name of the column to transform.

    Returns:
    Series: The transformed column as a pandas series.
    """
    dataframe["log_" + column] = np.log1p(dataframe[column])
    return dataframe["log_" + column]

def model_dataset(df):
    """
    Transforms the input dataset by applying a log transformation to select columns, and returns a normalized version of the transformed dataset.

    Parameters:
    df (DataFrame): The input dataset to transform.

    Returns:
    DataFrame: The transformed and normalized dataset.

    Notes:
    - This function assumes that the input dataset contains the columns 'diff_prod', 'nb_order', 'pos_sales', 'mean_quantity', and 'retour'.
    - The returned dataset includes the original columns as well as the transformed columns.
    """
    apply_log1p_transformation(df, 'diff_prod')
    apply_log1p_transformation(df, 'nb_order')
    apply_log1p_transformation(df,'pos_sales')
    apply_log1p_transformation(df, 'mean_quantity')
    apply_log1p_transformation(df, 'retour')

    # create final dataset for clustering
    df_log = df[['log_diff_prod', 'log_nb_order', 'log_pos_sales', 'log_mean_quantity',
       'log_retour']]

    # normalize data
    scaler = StandardScaler()
    data= scaler.fit_transform(df_log)
    df_log = pd.DataFrame(df_log, index=df_log.index, columns=df_log.columns)

    return df_log

def create_kmeans_model(dataframe, n_clusters=4, init='k-means++', max_iter=500, random_state=42):
    """
    Creates a KMeans model with the given parameters, fits it to the input dataframe, and returns the trained model.

    Parameters:
    dataframe (DataFrame): The input dataframe to train the KMeans model on.
    n_clusters (int): The number of clusters to form.
    init (str): The method used to initialize the centroids.
    max_iter (int): The maximum number of iterations of the algorithm for a single run.
    random_state (int): Determines random number generation for centroid initialization.

    Returns:
    KMeans: The trained KMeans model.
    """
    kmeans_model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=random_state)
    kmeans_model.fit(dataframe)
    return kmeans_model


def plot_kmeans_clusters(dataframe, kmeans_model):
    """
    Plots a 3D scatter plot of the clusters formed by the trained KMeans model using the input dataframe.

    Parameters:
    dataframe (DataFrame): The input dataframe to plot the clusters for.
    kmeans_model (KMeans): The trained KMeans model.

    Returns:
    None
    """
    dataframe["label"] = kmeans_model.labels_

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(dataframe.loc[dataframe["label"] == 0,'log_mean_quantity'],
                dataframe.loc[dataframe["label"] == 0,'log_pos_sales'],
                dataframe.loc[dataframe["label"] == 0,'log_retour'], 
                s = 40 , color = 'orange', label = "x")
    ax.scatter(dataframe.loc[dataframe["label"] == 1,'log_mean_quantity'],
                dataframe.loc[dataframe["label"] == 1,'log_pos_sales'],
                dataframe.loc[dataframe["label"] == 1,'log_retour'], 
                s = 40 , color = 'red', label = "y")
    ax.scatter(dataframe.loc[dataframe["label"] == 2,'log_mean_quantity'],
                dataframe.loc[dataframe["label"] == 2,'log_pos_sales'],
                dataframe.loc[dataframe["label"] == 2,'log_retour'], 
                s = 40 , color = 'green', label = "z")
    ax.scatter(dataframe.loc[dataframe["label"] == 3,'log_mean_quantity'],
                dataframe.loc[dataframe["label"] == 3,'log_pos_sales'],
                dataframe.loc[dataframe["label"] == 3,'log_retour'],
                 s = 40 , color = 'yellow', label = "z")

    ax.set_xlabel('log_mean_quantity', fontsize=15, fontweight ='bold')
    ax.set_ylabel('pos_sales', fontsize=15, fontweight ='bold')
    ax.set_zlabel('retour', fontsize=15, fontweight ='bold')
    ax.legend()
    fig.suptitle('Customers Clustering', fontsize=25, fontweight='bold')
    plt.show()


def plot_elbow_method(df):
    """
    Plots the Elbow curve using the distortion values for each value of K.

    Parameters:
    distortions (list): The list of distortion values for each value of K.
    K (range): The range of K values used to generate the distortion values.

    Returns:
    None.
    """
    distortions = []
    K = range(1, 10)
      
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k,  
                            max_iter=500, 
                            random_state=42)
        kmeanModel.fit(df)

        distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / df.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()
