from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import Literal


class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector that selects features based on correlation with the target.

    Parameters:
        method (str): Correlation method to use ('pearson', 'spearman', 'kendall').
        threshold (float): Minimum absolute correlation value to select features. Default is 0.1.

    Attributes:
        selected_features_ (list of str): List of selected feature names after fitting.
    """

    def __init__(self, method: Literal['pearson', 'spearman', 'kendall'], threshold: float = 0.1):
        self.method = method
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "CorrelationFeatureSelector":
        """
        Fit the selector by computing correlations with the target and selecting features.

        Parameters:
            X (pd.DataFrame): Feature data.
            y (pd.Series): Target variable.

        Returns:
            self (CorrelationFeatureSelector): Fitted selector.
        """
        self.correlations_ = X.corr(method=self.method)[y.name].abs()
        self.selected_features_ = self.correlations_[self.correlations_ > self.threshold].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data to keep only the selected features.

        Parameters
        X (pd.DataFrame): Feature data.

        Returns
        pd.DataFrame: DataFrame with selected features only.
        """
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        """
        Fit to data, then transform it to keep only selected features.

        Parameters:
            X (pd.DataFrame): Feature data.
            y (pd.Series | np.ndarray): Target variable.

        Returns:
            pd.DataFrame: Transformed DataFrame with selected features.
        """
        self.fit(X, y)
        return self.transform(X)
