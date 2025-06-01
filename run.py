import argparse
import torch

import logging
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

from src.utils import set_seed

import neps
print("Neps version:", neps)

from functools import partial
from pathlib import Path
from src.automl import SvrAutoML

from src.utils import *
from src.preprocessing import *
from src.visualization import *

from src.categorical_feature_selectors import MutualInformationSelector
from src.correlation_feature_selector import CorrelationFeatureSelector


def run_pipeline_once(seed, train_dataset, test_dataset, **config) -> dict:
    output_path = Path(f"preds-{seed}.npy")
    
    # Model params    
    kernel_option = config["kernel_options"]
    C_option = config["C_options"]
    epsilon_option = config["epsilon_options"]
    max_iter_option = config["max_iter_options"]
    
    # Construct the model
    model = SvrAutoML(
        kernel_option, C_option, max_iter_option, epsilon_option
    )

    x_train = train_dataset["X_train"]
    y_train = train_dataset["y_train"]

    model.fit(X=x_train, y=y_train)

    prediction_output = model.predict(X=test_dataset["X_test"], y_test=test_dataset["y_test"])

    mse = prediction_output["mse"]
    rmse = prediction_output["rmse"]
    r2 = prediction_output["r2"]
    
    train_info = {}
    train_info["loss"] = rmse
    train_info["accuracy"] = r2
    train_info["objective_to_minimize"] = rmse
    train_info["cost"] = 0.0
    train_info["learning_curve"] = None
    train_info["exception"] = None
    train_info["extra"] = {
        "kernel_option": kernel_option,
        "C_option": C_option,
        "epsilon_option": epsilon_option,
        "max_iter_option": max_iter_option
    }
    
    return train_info

def get_dataset():
    root_dir = Path("data").resolve()
    filename = 'train.csv'

    X  = read_data_from_csv(root_dir, filename).drop(columns=["Id"])
    y = X["SalePrice"]
    
    n_df = X[get_numeric_columns(X)]
    for col in columns_with_nans(n_df).index:
        n_df = feature_fill_nan_with_value(n_df, col, n_df[col].mean())
    
    n_df = CorrelationFeatureSelector(method='spearman', threshold=0.2).fit_transform(n_df, y)
    n_df = n_df.drop(columns='SalePrice')
    c_df = X[get_categorical_columns(X)]
    c_df = fill_nan(c_df, "NA")
    c_df = MutualInformationSelector(mi_threshold=0.1).fit_transform(c_df, y)
    
    X_preprocessed = pd.concat([n_df, c_df], axis=1).to_numpy()
    
    X_train, X_test, y_train, y_test = split_data(X_preprocessed, log_transform(y), test_size=0.3, shuffle=True)
    X_train, scalar = normalize_data(X_train)
    X_test = scalar.transform(X_test)

    return X_train, X_test, y_train, y_test

def main(seed: int, root_directory: str, overwrite: bool):
    logger.info(f"Running main function with seed: {seed}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using Device: {device}")

    # set seed for reproducibility
    set_seed(seed, deterministic=True)

    # =================================================================
    ##################### Define the search space #####################
    # =================================================================
    
    pipeline_space = dict(
        kernel_options = neps.Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
        C_options = neps.Float(lower=1, upper=100, log=True),
        epsilon_options = neps.Float(lower=1, upper=10, log=True),
        max_iter_options = neps.Integer(lower=1000, upper=50000)
    )
        
    X_train, X_test, y_train, y_test = get_dataset()
    train_dataset = { "X_train": X_train, "y_train": y_train }
    test_dataset = { "X_test": X_test, "y_test": y_test }
    
    neps.run(
        evaluate_pipeline=partial(run_pipeline_once, seed, train_dataset, test_dataset),
        pipeline_space=pipeline_space,
        root_directory=root_directory,
        max_evaluations_total=3000,
        overwrite_working_directory=overwrite,
        post_run_summary=True,
        optimizer=("bayesian_optimization")
        # max_cost_total=24 * 60 * 60,  # x hours
        # ignore_errors=False
    )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Add possible arguments
    # parser.add_argument(
    #     "--name",
    #     type=str, # Path, str, int, float, bool
    #     required=True, 
    #     help="Description of the parameter",
    #     choices=["fashion", "flowers", "emotions", "skin_cancer"],
    #     help = "Dataset to use for training the model",
    #     default="some_default_value"
    # )

    parser.add_argument(
        "--seed",
        type=int,
        default=69
    )
    
    parser.add_argument(
        "--root",
        type=str,
        default="results",
        help="The root directory to save or retrieve the results from."
    )
    
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=True,
        help="Whether to overwrite the existing results in the root directory."
    )
    
    args = parser.parse_args()
    
    logger.info(f"Started the main script with arguments: {args}")
    
    seed = args.seed
    root_directory = f"{args.root}/seed_{seed}"
    overwrite = args.overwrite
    
    logger.info(f"Root directory: {root_directory}, Overwrite: {overwrite}, Seed: {seed}")

    main(seed=seed, root_directory=root_directory, overwrite=overwrite)
    