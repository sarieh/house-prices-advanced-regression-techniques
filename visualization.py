from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple
"""
TODO
6- Pairplot for highly correlated features
"""

def correlation_heat_map(df: pd.DataFrame, method: str, title: str, figsize: Optional[Tuple[int, int]] = (16, 12)):
    """
    Plots a correlation heatmap of numeric features in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing numeric features.
        title (str): Title for the heatmap.
        figsize (Optional[Tuple[int, int]]): Size of the figure (width, height). Default is (16, 12).

    Returns:
        None
    """
    corr = df.corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        square=True,
        annot_kws={"size": 7},  # Smaller annotation text
        cbar_kws={"shrink": 0.75}  # Shrink color bar
    )
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title, fontsize=14)
    plt.tight_layout()  # Fit everything nicely
    plt.show()


def scatter_plots_against_target(df: pd.DataFrame, target_column: pd.Series):
    """
    Creates scatter plots of each numeric feature against a target column with regression and mean lines.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing numeric features and the target column.
        target_column (str): The name of the target column.

    Returns:
        None
    """
    features = df.columns.to_list()
    n_cols = 4
    n_rows = (len(features) + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 5, n_rows * 4))

    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        # Scatter plot with regression line
        sns.regplot(
            x=df[feature],
            y=target_column,
            scatter_kws={"color":"blue",'alpha': 0.5},
            line_kws={'color': 'red'},
            ci=None,
        )
        # Add mean line
        mean_val = df[feature].mean()
        plt.axvline(mean_val, color='green', linestyle='--', label=f'Mean {feature}: {mean_val:.2f}')

        # Titles and labels
        plt.xlabel(feature)
        plt.ylabel(target_column.name)
        plt.title(f'{feature} vs {target_column.name}')

        # Add legend
        plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def categorical_vs_target_plot(cat_df: pd.DataFrame, target: pd.Series):
    """
    Creates boxplots for each categorical variable against a continuous target variable.

    Parameters:
        cat_df (pd.DataFrame): DataFrame containing categorical variables.
        target (pd.Series): Series representing the continuous target variable.

    Returns:
        None
    """
    n_cols = 4
    n_rows = (len(cat_df.columns) + n_cols - 1) // n_cols
    _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

    # Ensure axes is always 2D array for consistent indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, col in enumerate(cat_df.columns):
        row = i // n_cols
        col_idx = i % n_cols

        sns.boxplot(x=cat_df[col], y=target, ax=axes[row, col_idx], hue=cat_df[col], legend=False, palette="pastel")
        axes[row, col_idx].tick_params(axis='x', rotation=45)
        axes[row, col_idx].set_title(f'{col} vs Sale Price', fontsize=12) # Adjust font size as needed
        axes[row, col_idx].set_xlabel(col, fontsize=10)
        axes[row, col_idx].set_ylabel('Sale Price', fontsize=10) # More descriptive Y-label
        plt.setp(axes[row, col_idx].get_xticklabels(), ha="right", rotation_mode="anchor")

    # Hide any unused subplots
    for i in range(len(cat_df.columns), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_target_distribution(target: pd.Series, title: str, xlablel: str, ylabel: str):
    """
    Plots the distribution of the target variable using a histogram and KDE.

    Parameters:
        df (pd.DataFrame): A dataframe containing the target.
        title (str): Title of the plot.
        xlablel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.

    Returns:
        None
    """
    sns.histplot(target, kde=True)

    plt.title(title)
    plt.xlabel(xlablel)
    plt.ylabel(ylabel)
    plt.show()
