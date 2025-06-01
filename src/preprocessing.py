import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Optional, Any

#TODO: Find a method to detect & deal with outliers

def normalize_data(df, columns=None):
    """
    Normalize specified columns using MinMaxScalerss.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of numeric column names to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized columns.
        scaler: Fitted StandardScaler instance (can be reused on test data).
    """
    scaler = MinMaxScaler()
    if isinstance(df, pd.DataFrame) and columns in df.columns:
        df[columns] = scaler.fit_transform(df[columns])
    df = scaler.fit_transform(df)
    return df, scaler


def one_hot_encode(df, columns):
    """
    Apply OneHotEncoding to specified categorical columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of categorical column names to encode.

    Returns:
    pd.DataFrame: One-hot encoded DataFrame.
    encoder: Fitted OneHotEncoder instance.
    """
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns), index=df.index)
    df = df.drop(columns=columns)
    return pd.concat([df, encoded_df], axis=1), encoder


def ordinal_encode(df, columns, categories):
    """
    Apply OrdinalEncoding to specified columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to encode.
    categories (list of lists): List of category orders for each column.

    Returns:
    pd.DataFrame: Ordinally encoded DataFrame.
    encoder: Fitted OrdinalEncoder instance.
    """
    encoder = OrdinalEncoder(categories=categories)
    df[columns] = encoder.fit_transform(df[columns])
    return df, encoder


def log_transform(data: pd.Series | np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """
    Apply log1p (log(1 + x)) transformation to numeric data.
    Supports pandas Series, NumPy arrays, or DataFrames.

    Parameters:
        data (Union[pd.Series, np.ndarray, pd.DataFrame]): Input numeric data.

    Returns:
        Union[pd.Series, np.ndarray, pd.DataFrame]: Log-transformed data.
    """
    if isinstance(data, pd.Series):
        return np.log1p(data)
    elif isinstance(data, pd.DataFrame):
        return data.apply(np.log1p)
    elif isinstance(data, np.ndarray):
        return np.log1p(data)
    else:
        raise TypeError("Input must be a pandas Series, DataFrame, or NumPy array.")


def split_data(
    X: pd.DataFrame | np.ndarray, y: np.ndarray, test_size: Optional[float]=0.2, random_state=42, shuffle: Optional[bool] = False
):
    """
    Split the dataset into training and test sets.

    Parameters:
        X (pd.DataFrame | np.array): Features.
        y (np.array): Target.
        test_size (float): Fraction of data to use as test set.
        random_state (int): Random seed.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)


def drop_feature_if_nan_exceeds_threshold(df: pd.DataFrame, feature: str, threshold: Optional[float] = 0.8) -> pd.DataFrame:
    """
    Drops the specified feature (column) from the DataFrame if the proportion of NaN values
    exceeds the given threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        feature (str): Name of the column to check and potentially drop.
        threshold (Optional[float]=0.8):  Proportion threshold of NaN values above which the feature is dropped.

    Returns:
        pd.DataFrame: DataFrame with the feature dropped if threshold exceeded, else unchanged.

    """
    if df[feature].isna().mean() > threshold:
        return df.drop(columns=[feature], inplace=False)
    return df


def feature_fill_nan_with_value(df: pd.DataFrame, feature: str, fill_value: Any, deep: Optional[bool] = False) -> pd.DataFrame:
    """
    Fills NaN values in the specified feature (column) with a given fill_value.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        feature (str): Name of the column in which to fill NaN values.
        fill_value (Any): Value to replace NaNs with (e.g., string, number).
        deep (Optional[bool], optional): Whether to perform a deep copy of the DataFrame before modification. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with NaN values in the feature replaced by fill_value.
    """
    df = df.copy(deep=deep)
    df[feature] = df[feature].fillna(fill_value)
    return df


def fill_nan(df: pd.DataFrame, fill_value: Any, deep: Optional[bool] = False) -> pd.DataFrame:
    """
    Fills all NaN values in the DataFrame with a specified fill value.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        fill_value (Any): Value to replace NaNs with (e.g., string, number).
        deep (Optional[bool], optional): Whether to perform a deep copy of the DataFrame before modification. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with all NaN values replaced by fill_value.
    """
    df = df.copy(deep=deep)
    return df.fillna(fill_value)
