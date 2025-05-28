import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Any
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def read_data_from_csv(root_dir: Path, filename: str) -> pd.DataFrame:
    """
    Reads a CSV file from the specified directory and returns it as a pandas DataFrame.

    Parameters:
        root_dir (Path): The root directory where the CSV file is located.
        filename (str): The name of the CSV file to read.

    Returns:
        pd.DataFrame: The contents of the CSV file as a DataFrame.
    """
    file_path = root_dir / filename
    return pd.read_csv(file_path)


def feature_names(data: pd.DataFrame) -> pd.Index:
    """
    Returns the column names (features) of a pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Index: An index object containing the column names of the DataFrame.

    Raises:
        ValueError: If the input is None or not a pandas DataFrame.
    """
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("Expected a non-null pandas DataFrame.")

    return data.columns


def columns_with_nans(data: pd.DataFrame) -> pd.Series:
    """
    Returns a pandas Series of column names and the number of NaN values
    for columns that contain at least one NaN.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: A Series where the index is column names and the values
                   are the count of NaN values, filtered to only include
                   columns with at least one NaN.

    Raises:
        ValueError: If the input is None or not a pandas DataFrame.
    """
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("Expected a non-null pandas DataFrame.")

    nan_counts = data.isna().sum()
    return nan_counts[nan_counts > 0]


def get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """
    Returns a list of column names in the DataFrame that have numeric data types.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        list[str]: A list of column names with numeric data types (e.g., int, float).

    Raises:
        ValueError: If the input is None or not a pandas DataFrame.
    """
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("Expected a non-null pandas DataFrame.")

    return data.select_dtypes(include='number').columns.tolist()


def get_categorical_columns(data: pd.DataFrame) -> list[str]:
    """
    Returns a list of column names in the DataFrame that have categorical or object data types.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        list[str]: A list of column names with object or category data types.

    Raises:
        ValueError: If the input is None or not a pandas DataFrame.
    """
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("Expected a non-null pandas DataFrame.")

    return data.select_dtypes(include=['object', 'category']).columns.tolist()


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


def fill_nan_with_value(df: pd.DataFrame, feature: str, fill_value: Any, deep: Optional[bool] = False) -> pd.DataFrame:
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


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each feature in the DataFrame.

    Parameters:
        X (pd.DataFrame): Feature matrix (only numeric features).

    Returns:
        pd.DataFrame: DataFrame with features and their corresponding VIF scores.
    """
    X = X.copy()
    X = add_constant(X)  # Add intercept term

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif_data.drop(index=0)  # Drop the constant
