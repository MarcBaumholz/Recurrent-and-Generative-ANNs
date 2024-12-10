#! env/bin/python3

"""
This script contains python functions that are useful for model evaluation, such as data writing and plotting.
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from scipy.stats import gaussian_kde
from dask.diagnostics import ProgressBar


def write_iot_dataset(
    cfg: DictConfig,
    inputs: np.array,
    outputs: np.array,
    targets: np.array,
    file_path: str
) -> None:
    """
    Creates a netCDF dataset for inputs, outputs, and targets and writes it to file.
    
    :param cfg: The hydra configuration of the model
    :param inits: The inputs to the model
    :param outputs: The outputs of the model (predictions)
    :param targets: The ground truth and target for prediction
    :param file_path: The path to the directory where the datasets are written to
    """
    if cfg.verbose: print("Building and writing input/output/target dataset")

    # Set up netCDF dataset
    coords = {}
    coords["sample"] = range(inputs.shape[0])
    coords["input_vars"] = cfg.data.input_features
    coords["target_vars"] = cfg.data.target_features

    inputs_da = xr.DataArray(data=inputs, dims=["sample", "input_vars"])
    outputs_da = xr.DataArray(data=outputs, dims=["sample", "target_vars"])
    targets_da = xr.DataArray(data=targets, dims=["sample", "target_vars"])

    dataset = xr.Dataset(coords=coords, data_vars={"inputs": inputs_da, "outputs": outputs_da, "targets": targets_da})

    # Write dataset to file
    dst_path_name = os.path.join(file_path, "iot_dataset.nc")
    if os.path.exists(dst_path_name): os.remove(dst_path_name)  # Delete file if it exists
    
    if cfg.verbose:
        write_job = dataset.to_netcdf(os.path.join(dst_path_name), compute=False)
        with ProgressBar(): write_job.compute()
        print()
    else:
        dataset.to_netcdf(os.path.join(dst_path_name)) 


def write_fi_dataset(
    cfg: DictConfig,
    fi_scores: np.array,
    file_path: str
) -> None:
    """
    Creates and writes a netCDF dataset for the feature importance analysis to file.

    :param cfg: The hydra configuration of the model
    :param fi_scores: The feature importance array
    :param file_path: The path to the directory where the datasets are written to
    """
    if cfg.verbose: print("Building and writing feature importance dataset")

    # Set up netCDF dataset
    coords = {}
    coords["input_vars"] = cfg.data.input_features
    
    fi_da = xr.DataArray(data=fi_scores, dims=["input_vars"])
    dataset = xr.Dataset(coords=coords, data_vars={"feature_importance_score": fi_da})

    # Write dataset to file
    dst_path_name = os.path.join(file_path, "feature_importance.nc")
    if os.path.exists(dst_path_name): os.remove(dst_path_name)  # Delete file if it exists

    if cfg.verbose:
        write_job = dataset.to_netcdf(os.path.join(dst_path_name), compute=False)
        with ProgressBar(): write_job.compute()
        print()
    else:
        dataset.to_netcdf(os.path.join(dst_path_name))


def generate_scatter_plot(
    cfg: DictConfig,
    ds_name: str,
    file_path: str
) -> None:
    # Open dataset and compute some statistics
    ds = xr.open_dataset(os.path.join(file_path, ds_name + ".nc"))

    outputs = ds.outputs.values
    targets = ds.targets.values

    rmse = np.sqrt(np.mean(np.square(outputs-targets)))
    std = np.std(np.sqrt(np.square(outputs-targets)))
    corr = np.corrcoef(x=outputs[:, 0], y=targets[:, 0])[0, 1]
    if cfg.verbose: print(f"\tRMSE: {rmse} +- {std}")

    # Prepare the data for plotting
    xy = np.vstack([np.transpose(targets),
                    np.transpose(outputs)])
    z = gaussian_kde(xy)(xy)

    # Generate linear fit data
    slope, intercept = np.polyfit(targets[:, 0], outputs[:, 0], 1)
    x_vals = targets[:, 0]
    y_vals = intercept + slope*x_vals

    # Plot the data as scatterplot
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(x=targets[:, 0], y=outputs[:, 0], c=z, s=25, edgecolor="k", label=f"Data points, rmse={np.round(rmse, 2)}")
    ax.plot(x_vals, x_vals, "-", label="Main diagonal (target)")
    ax.plot(x_vals, y_vals, "--", label=rf"Fit $y={np.round(slope, 2)}x+{np.round(intercept, 2)}$, $r^2={np.round(corr, 2)}$")
    ax.set_title("Prediction on unseen data for " + cfg.data.target_features[0])
    ax.set_xlabel("Observations")
    ax.set_ylabel("Predictions")
    ax.legend(fontsize="small")
    fig.savefig(f"{os.path.join(file_path, 'scatterplot.pdf')}", bbox_inches="tight")
    if cfg.verbose:
        print(f"\tSaved plot to {os.path.join(file_path, 'scatterplot.pdf')}")
        plt.show()
    else:
        plt.close()


def generate_feature_importance_barplot(
    cfg: DictConfig,
    ds_name: str,
    file_paths: list,
    multi_model_plot_path: str = None
) -> None:
    
    # Retrieve feature importance scores and compute mean and standard deviation if multiple models are compared
    feat_imp_score_list = []
    for file_path in file_paths:
        ds = xr.open_dataset(os.path.join(file_path, ds_name + ".nc"))
        feat_imp_score_list.append(ds.feature_importance_score.values)
    if not all(len(l) == len(feat_imp_score_list[0]) for l in feat_imp_score_list): raise ValueError(
        "Multi-model feature importance not possible because provided models have different parameter counts.")
    feat_imp_score_ary = np.array(feat_imp_score_list)
    mean = np.mean(feat_imp_score_ary, axis=0)
    std = np.std(feat_imp_score_ary, axis=0)
    var_names = ds.input_vars.values

    # Sort feature importance starting with largest value
    sort_indices = np.argsort(mean)
    mean = mean[sort_indices]
    std = std[sort_indices]
    var_names = var_names[sort_indices]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.barh(var_names, mean, xerr=std, align="center", capsize=3, color="dodgerblue", ecolor="grey")
    ax.set_yticks(range(len(var_names)))
    ax.set_yticklabels(var_names, rotation=30)
    ax.set_title(f"Feature importance scores for {cfg.data.target_features[0]}")
    ax.set_xlim([0, 1])
    #ax.set_xscale("log")

    # Write plot to file
    if multi_model_plot_path is not None: file_path = multi_model_plot_path
    dst_path = os.path.join(file_path, f"feature_importance_{cfg.data.target_features[0]}.pdf")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    fig.savefig(dst_path, bbox_inches="tight")
    if cfg.verbose:
        print(f"\tSaved plot to {dst_path}")
        plt.show()
    else:
        plt.close()
