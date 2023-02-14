import pandas as pd

def load_data(file_path):
    """Load data from a Parquet file.

    Args:
    file_path (str): The file path to the Parquet file.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the Parquet file.
    """
    import pyarrow.parquet as pq

    # Load the Parquet file
    table = pq.read_table(file_path)

    # Convert the table to a Pandas DataFrame
    df = table.to_pandas()
    
    return df


def preprocess_data(df, frac=0.03, random_state=42):
    """
    Preprocesses the given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be preprocessed.
    frac : float, optional
        The fraction of rows to be selected randomly from the DataFrame, by default 0.03.
    random_state : int, optional
        The seed to be used by the random number generator, by default 42.

    Returns
    -------
    pandas.DataFrame
        The preprocessed DataFrame.
    """
    # Randomly select half of the rows in the DataFrame
    sub_df = df.sample(frac=frac, random_state=random_state)
    # Drop duplicates
    sub_df = sub_df.drop_duplicates()
    # Transform as date
    sub_df['date_order'] = pd.to_datetime(sub_df['date_order'])

    return sub_df
