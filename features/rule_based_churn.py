import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm

def count_orders_per_client(df):
    """
    Count the number of orders for each client in the given dataframe.

    Args:
    df (pandas.DataFrame): The dataframe containing the client information.

    Returns:
    pandas.DataFrame: A dataframe containing the client_id and the order_number for each client.
    """
    df_grouped = df.sort_values(by=['client_id', 'date_order'])
    df_grouped['order_number'] = df_grouped.groupby('client_id')['date_order'].transform(lambda x: (x != x.shift()).cumsum())
    df_grouped = df_grouped[df_grouped.groupby('client_id')['order_number'].transform('size') > 1]
    return df_grouped


def compute_time_difference_between_orders(df):
    """
    Computes the time difference between orders for a given client and drop the clients with only one order.

    Parameters:
    df (DataFrame): The input DataFrame containing the transaction data.

    Returns:
    DataFrame: A DataFrame containing the time difference between orders for each client, sorted by client_id and date_order.

    """
    df['date_order'] = pd.to_datetime(df['date_order'])
    df = df.sort_values(by=['client_id', 'date_order'])
    df['time_diff'] = df.groupby('client_id')['date_order'].diff().dt.total_seconds()/(24*60*60)
    df.loc[df.groupby('client_id')['date_order'].head(1).index, 'time_diff'] = 0
    return df


def compute_mean_time_diff(df):
    """
    Computes the mean time difference between orders for each client in the input DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing the transaction data.

    Returns:
    DataFrame: The input DataFrame  with a new column containing the mean time difference between orders for each client.
    """
    mean_time_diff = df.groupby('client_id')['time_diff'].mean().reset_index(name='mean_time_diff')
    mean_time_diff['mean_time_diff'] = round(mean_time_diff['mean_time_diff'], 2)
    mean_time_diff['std_time_diff'] = df.groupby('client_id')['time_diff'].std().reset_index(name='std_time_diff')['std_time_diff']
    mean_time_diff['std_time_diff'] = round(mean_time_diff['std_time_diff'], 2)

    # Join the mean time difference and the standard deviation back to the original data
    df_mean_time_diff = df.merge(mean_time_diff, on='client_id', how='left')
    
    return df_mean_time_diff

def compute_churn_flag(df, confidence_level=0.99, warning_level=0.95):
    """
    Computes the churn flag for each client in the input DataFrame based on their time since last order.

    Parameters:
    df (DataFrame): The input DataFrame containing the transaction data.
    confidence_level (float): The confidence level used to compute the upper and lower thresholds.
    warning_level (float): The warning level used to compute the upper and lower warning levels.

    Returns:
    DataFrame: A DataFrame containing the churn flag for each client.
    """
    # Compute upper and lower thresholds based on confidence level
    df['upper_threshold'] = df['mean_time_diff'] + stats.norm.ppf((1 + confidence_level) / 2) * df['std_time_diff']
    df['lower_threshold'] = df['mean_time_diff'] - stats.norm.ppf((1 + confidence_level) / 2) * df['std_time_diff']

    # Compute upper and lower warning levels based on warning level
    df['upper_warning'] = df['mean_time_diff'] + stats.norm.ppf((1 + warning_level) / 2) * df['std_time_diff']
    df['lower_warning'] = df['mean_time_diff'] - stats.norm.ppf((1 + warning_level) / 2) * df['std_time_diff']

    # Group by client and keep only last order line
    df['date_order'] = pd.to_datetime(df['date_order'])
    df_last_order = df.groupby('client_id').tail(1)

    # Compute time difference between last order and max order date in df
    max_order_date = df['date_order'].max()
    df_last_order['time_since_last_order'] = (max_order_date - df_last_order['date_order']).dt.total_seconds()/(24*60*60)

    # Compute churn flag
    ## Define the conditions and the values for the churn_flag column
    conditions = [
        df_last_order['time_since_last_order'] > df_last_order['upper_threshold'],
        (df_last_order['time_since_last_order'] <= df_last_order['upper_threshold']) & (df_last_order['time_since_last_order'] > df_last_order['upper_warning']),
        ]
    values = ['churn', 'careful']

    # Add a new column named churn_flag based on the conditions and values
    df_last_order['churn_flag'] = np.select(conditions, values, default='good')

    return df_last_order


