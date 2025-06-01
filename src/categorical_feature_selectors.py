from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import get_categorical_columns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import  OrdinalEncoder
import numpy as np
import pandas as pd


class MutualInformationSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector based on mutual information scores.

    This transformer encodes categorical features using an OrdinalEncoder
    and selects features whose mutual information score with the target
    exceeds a specified threshold.

    Parameters
        mi_threshold (float, optional): Threshold for mutual information score above which a
            feature is selected. Default is 0.01.

    Attributes
        mi_support_ (numpy.ndarray): Boolean mask indicating which features passed the MI threshold.
        selected_features_ (pandas.Index): Names of the selected features after MI thresholding.
    """

    def __init__(self, mi_threshold=0.01):
        self.mi_threshold = mi_threshold
        self.encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-1

        )

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "MutualInformationSelector":
        """
        Fit the selector to the data.

        Parameters:
            X (pandas.DataFrame): Feature data.
            y (pandas.Series | np.ndarray): Target variable.

        Returns:
            self (MutualInformationSelector): Fitted selector.
        """

        X_encoded = pd.DataFrame(
            self.encoder.fit_transform(X),
            columns=self.encoder.get_feature_names_out(X.columns),
            index=X.index
        )

        mi_scores = mutual_info_regression(X_encoded, y, discrete_features=True)

        self.mi_support_ = mi_scores > self.mi_threshold
        self.selected_features_ = X_encoded.columns[self.mi_support_]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data to retain only selected features.

        Parameters:
            X (pandas.DataFrame): Feature data.

        Returns:
            pandas.DataFrame: DataFrame with selected features after encoding.
        """
        X_encoded = pd.DataFrame(
            self.encoder.transform(X),
            columns=self.encoder.get_feature_names_out(X.columns),
            index=X.index
        )
        return X_encoded[self.selected_features_]

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
