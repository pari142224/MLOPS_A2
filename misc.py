# Inside data.py or misc.py
import pandas as pd
import numpy as np

def load_data():
    """Loads the Boston Housing dataset manually as specified in the assignment."""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    # The separator is one or more spaces ('\s+')
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

    # Split into data and target arrays
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    # Create a DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # MEDV is the target variable
    return df
