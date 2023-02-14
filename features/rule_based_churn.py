import pandas as pd
import numpy as np

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
    return df


def compute_churned_clients(df):
    """
    This function computes the average time difference between orders for each client and labels clients as "churned" 
    
    Parameters:
    df (pandas DataFrame): DataFrame containing client_id, date_order, and time_diff columns.
    
    Returns:
    mean_time_diff (pandas DataFrame): DataFrame containing the average time difference between orders for each client, 
    rounded to 2 decimal places, and a column indicating whether the client is "churned" or not.
    """
    # Sort values by client_id and date_order
    df['date_order'] = pd.to_datetime(df['date_order'])
    df.sort_values(by=['client_id', 'date_order'], inplace=True)
    
    # Compute time difference between orders
    df['time_diff'] = df.groupby('client_id')['date_order'].diff().dt.total_seconds()/(24*60*60)
    
    # Compute the average time difference for each client
    mean_time_diff = df.groupby('client_id')['time_diff'].mean().reset_index(name='mean_time_diff')
    mean_time_diff['mean_time_diff'] = round(mean_time_diff['mean_time_diff'], 2)

    # Label clients as "churned" if the maximum time between two orders is 2.5 times greater than the average time difference
    mean_time_diff['churned'] = np.where(mean_time_diff['mean_time_diff'] * 2.5 < df.groupby('client_id')['time_diff'].max().reset_index(name='time_diff')['time_diff'], 1, 0)

    # Join the mean time difference back to the original data
    df_churned = df.merge(mean_time_diff, on='client_id', how='left')

    return df_churned



