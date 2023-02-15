# BCG X Case Study

## Project Title

Developing a data-driven approach that supports ClientCo's overall growth and modernization strategy.

## Description

In this project, we are assisting the Chief Data Officer (CDO) of ClientCo, a private international distributor of construction materials, in developing value-added data programs to showcase the benefits of advanced data and analytics to the company's top leadership. We identify a comprehensive list of potential data use cases, evaluate their suitability and strategic fit with ClientCo's activities and internal context, and select the top three priority use cases to present a one-year implementation roadmap. We have access to several databases, meeting minutes with key stakeholders, and other materials to aid in our decision-making process. 

## Getting started with the repository
​
To ensure that all libraries are installed pip install the requirements file:
 
```pip install -r requirements.txt```

​
To run the model go to the console and run following command: 
 
```python main.py```

​
You should be at the source of the repository structure (ie. data-for-strategt) when running the command.
Note that when a graph pops-ip, you need to close windows in order to continue. The graph will be saved automatically so you can find it later.

Our repository is structured in the following way:
​
```
|data-for-strategy
   |--data
   |--features
   |--------clustering_model.py
   |--------prepare_data.py
   |--------rule_based_churn.py
   |--notebooks
   |--------Clustering_Model.ipynb
   |--------EDA.ipynb
   |--------Rule_Based_Churn.ipynb
   |--main.py
   |--README.md
   |--requirements.txt
   |--.gitignore
```

### data 

To properly get the data, one has to download it locally on his/her computer and put the folder in the repository. One should have a folder like data/ with 1 parquet file:
- transaction_data.parquet

Note that after running the main you will have an extra .csv file called 'sub_df.csv'

### notebooks

**1) EDA**

A notebook with some EDA and visualization of the dataset.

**2) Rule Based Churn**

A notebook testing the functions used to build the churn prediction model.

**3) Clustering Model**

A notebook testing the functions used to build the churn prediction model.

### features

**1) prepare_data.py**

This file contains function to load and preprocess the data, using Pandas and PyArrow to read and manipulate Parquet files.

The load_data() function takes a file path to a Parquet file, reads the data using PyArrow's read_table() function, and converts it into a Pandas DataFrame using to_pandas().

The preprocess_data() function takes a DataFrame, randomly selects a fraction of the rows specified by frac, drops any duplicates, and converts a column named "date_order" to a datetime format. The frac parameter controls how much of the data is being selected, so it can be adjusted based on the size of the original dataset and the user need.


**2) clustering_model.py**

The file contains functions that preprocess the input data to prepare it for clustering analysis. It includes data cleaning, feature engineering, and model training. Specifically, the functions preprocess the data, add variables to the dataset, apply logarithmic transformation to specific columns of the dataset, normalize the dataset, and finally, create a **KMeans model** with the specified parameters, fit it to the input dataset, and return the trained model. 

The functions include preprocess_df(), add_variables(), apply_log1p_transformation(), model_dataset(), and create_kmeans_model(). The file provides an end-to-end solution for analyzing the input data through clustering.

**3) rule_based_churn.py**

This file contains functions for computing customer churn in the input dataset. The functions take a pandas DataFrame containing customer data, compute the time between orders for each customer, and use statistical analysis to classify customers as "good", "careful", or "churned". The functions include count_orders_per_client(), compute_time_difference_between_orders(), compute_mean_time_diff(), and compute_churn_flag(). The compute_churn_flag() function generates a new DataFrame with a "churn_flag" column indicating the churn status for each customer. Note that the statistical thresholds used to determine churn status can be adjusted by the user.
