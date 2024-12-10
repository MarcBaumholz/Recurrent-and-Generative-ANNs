#! bin/env/python3

import argparse
import os
import sys

import hydra
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
import torch as th
from omegaconf import DictConfig
import os

sys.path.append("")
from models import *
from data.pytorch_datasets import *


def predict(
    cfg: DictConfig,
    dataset: th.utils.data.DataLoader,
    model: th.nn.Module,
    fp_idx: int = None
):
    """
    Generates predictions with the given model using the provided dataset. Optionally permutes inputs, if specified.

    :param cfg: The configuration of the model
    :param dataloader: The dataloader to prepare the data
    :param model: The ML model to generate predictions with
    :param fp_idx: The index of the feature to be permuted. None for no permutation.
    """

    device = th.device(cfg.device)
    dataloader = th.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.testing.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Evaluate (without gradients): iterate over all test samples
    #print()
    with th.no_grad():
        inputs = list(); outputs = list(); targets = list()
        # Load data and generate predictions
        #for inpt, trgt in tqdm(dataloader, desc=f"Generating predictions"):
        for inpt, trgt in dataloader:
            # Permute the inputs along a given index if desired
            if fp_idx is not None: inpt = permute_inputs(inpt=inpt, fp_idx=fp_idx)
            inpt, trgt = inpt.to(device=device), trgt.to(device=device)
            otpt = model(
                x=inpt,
                teacher_forcing_steps=cfg.testing.teacher_forcing_steps,
                inference_steps=inpt.shape[1] - cfg.data.context_size
            )
            inputs.append(inpt.cpu()); outputs.append(otpt.cpu()); targets.append(trgt.cpu())
        inputs, outputs, targets = th.cat(inputs).numpy(), th.cat(outputs).numpy(), th.cat(targets).numpy()

    return inputs, outputs, targets


def evaluate_model(cfg: DictConfig, file_path: str, fpem_reps: int = 0) -> None:
    """
    Evaluates a single model for a given configuration.

    :param cfg: The hydra configuration for the model
    :param file_path: The destination path for the resulting plots
    :param fpem_reps: The number of repetitions for the feature permutation per variable
    """

    if cfg.verbose: print("\n\nInitialize dataloader and model")
    device = th.device(cfg.device)

    # Initializing dataset for testing
    dataset = hydra.utils.instantiate(config=cfg.data, mode="test")

    # Set up model
    model = hydra.utils.instantiate(config=cfg.model).to(device=device)
    if cfg.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\tModel {cfg.model.name} has {trainable_params} trainable parameters")

    # Load checkpoint from file to continue training or initialize training scalars
    checkpoint_path = os.path.join("outputs", cfg.model.name, "checkpoints", f"{cfg.model.name}_best.ckpt")
    if cfg.verbose: print(f"\tRestoring model from {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Generate predictions
    if cfg.verbose: print("\nGenerate predictions")
    inputs, outputs, targets = predict(cfg=cfg, dataset=dataset, model=model)

    rmse = np.sqrt(np.mean((outputs-targets)**2))
    print("Evaluating the model")
    print("\trmse = ", rmse)

    otpt, trgt = outputs[0, :, 0], targets[0, :, 0]

    # Plot the wave activity at one position
    print(f"\tSaving wave animation and wave plot at one position to {file_path}")
    idx_plot_point = (12, 12)
    teacher_forcing_steps = cfg.training.teacher_forcing_steps

    fig, ax = plt.subplots(1, 1, figsize=[8, 2])
    ax.plot(range(len(otpt)), otpt[:, idx_plot_point[0], idx_plot_point[1]], label="Prediction")
    ax.plot(range(len(trgt)), trgt[:, idx_plot_point[0], idx_plot_point[1]], label="Ground truth")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wave amplitude")

    # Add a red vertical line at x = teacher_forcing_steps - context_size
    ax.axvline(x=teacher_forcing_steps, color='red', linestyle='--', label='closed loop starts here')

    ax.legend()
    # ax.set_xlim([0, time_steps - 1])
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, "wave_at_one_pos.png"), dpi=150, bbox_inches='tight')
    plt.show()

    def animate(_t):
        im1.set_array(otpt[_t, :, :])
        im2.set_array(trgt[_t, :, :])
        fig.suptitle(f"Frame {_t}")
        return im1, im2

    # Animate the overall wave
    fig, axs = plt.subplots(1, 2, figsize=[12, 6], sharex=True, sharey=True)

    # Set the origin to 'lower' to match coordinate systems
    im1 = axs[0].imshow(otpt[0, :, :], vmin=-1.5, vmax=1.5, cmap="Blues", origin='lower')  # <-- Modified line
    im2 = axs[1].imshow(trgt[0, :, :], vmin=-1.5, vmax=1.5, cmap="Blues", origin='lower')  # <-- Modified line

    # Add markers at the idx_plot_point position
    axs[0].scatter(idx_plot_point[1], idx_plot_point[0], s=50, c='red')  # <-- Added line
    axs[1].scatter(idx_plot_point[1], idx_plot_point[0], s=50, c='red')  # <-- Added line

    title = fig.suptitle("Frame 0")
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(otpt),
        interval=200,
        blit=False
    )

    axs[0].axis("off")
    axs[1].axis("off")
    axs[0].set_title("Prediction")
    axs[1].set_title("Ground truth")

    # Save the animation
    writer = PillowWriter(fps=5)
    anim.save(os.path.join(file_path, "wave_animation.gif"), writer=writer)

    #plt.tight_layout()
    plt.show()


def run_evaluations(
    configuration_dir_list: str,
    device: str,
    overide: bool = False,
    batch_size: int = None,
    fpem_reps: int = 10,
    silent: bool = False
):
    """
    Evaluates a model with the given configuration.

    :param configuration_dir_list: A list of hydra configuration directories to the models for evaluation
    :param device: The device where the evaluations are performed
    """

    for configuration_dir in configuration_dir_list:
        # Initialize the hydra configurations for this forecast
        with hydra.initialize(version_base=None, config_path=os.path.join("..", configuration_dir, ".hydra")):
            cfg = hydra.compose(config_name="config")
            # Dirty hack to use WaveDataset instead of WaveUnetDataset during inference also for UNet
            cfg.data._target_ = "data.pytorch_datasets.WaveDataset"
            cfg.device = device
            if batch_size: cfg.testing.batch_size = batch_size

        if cfg.seed:
            np.random.seed(cfg.seed)
            th.manual_seed(cfg.seed)

        # Generate predictions if they do not exist and load them
        file_path = os.path.join("outputs", str(cfg.model.name), "plots")
        os.makedirs(file_path, exist_ok=True)
        evaluate_model(cfg=cfg, file_path=file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with a given configuration. Particular properties of the configuration can be "
                    "overwritten, as listed by the -h flag.")
    parser.add_argument("-c", "--configuration-dir-list", nargs='*', default=["configs"],
                        help="List of directories where the configuration files of all models to be evaluated lie.")
    parser.add_argument("-d", "--device", type=str, default="cpu",
                        help="The device to run the evaluation. Any of ['cpu' (default), 'cuda', 'mpg'].")
    parser.add_argument("-o", "--overide", action="store_true",
                        help="Override model predictions and evaluation files if they exist already.")
    parser.add_argument("-b", "--batch-size", type=int, default=None,
                        help="Batch size used for evaluation. Defaults to None to take entire test set in one batch.")
    parser.add_argument("-s", "--silent", action="store_true",
                        help="Silent mode to prevent printing results to console and visualizing plots dynamically.")

    run_args = parser.parse_args()
    run_evaluations(configuration_dir_list=run_args.configuration_dir_list,
                    device=run_args.device,
                    overide=run_args.overide,
                    batch_size=run_args.batch_size,
                    silent=run_args.silent)
    
    print("Done.")
