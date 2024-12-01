#! /usr/bin/env python3

"""
Main file for training a model
"""

import numpy as np
import torch as th
import torch.nn as nn
import glob
import os
import sys
import time
import matplotlib.pyplot as plt
from threading import Thread
sys.path.append("../../utils")
from modules import Model
from configuration import Configuration
import helper_functions as helpers


def run_training():

    # Load the user configurations
    cfg = Configuration('config.json')

    # Print some information to console
    #print("Architecture name:", cfg.model.architecture)
    print("Model name:", cfg.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config
    # file
    if cfg.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(cfg.general.device)
    # Set device on GPU if specified in the configuration file, else CPU
    device = helpers.determine_device()
    
    # Initialize and set up the model
    model = Model(
        d_one_hot=cfg.model.d_one_hot,
        d_lstm=cfg.model.d_lstm,
        num_lstm_layers=cfg.model.num_lstm_layers
    ).to(device=device)

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    #
    # Set up an optimizer and the criterion (loss)
    optimizer = th.optim.Adam(model.parameters(),
                              lr=cfg.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    #
    # Set up a list to save and store the epoch errors
    epoch_errors = []
    best_error = np.inf

    #
    # Set up the training dataloader
    dataset, dataloader = helpers.build_dataloader(
        cfg=cfg, batch_size=1
    )

    """
    TRAINING
    """

    a = time.time()

    #
    # Start the training and iterate over all epochs
    for epoch in range(cfg.training.epochs):

        epoch_start_time = time.time()

        # List to store the errors for each sequence
        sequence_errors = []

        # Iterate over the training batches
        for batch_idx, (net_input, net_label) in enumerate(dataloader):

            # Move data to the desired device and convert from
            # [batch_size, time, dim] to [time, batch_size, dim]
            net_input = net_input.to(device=device).transpose(0, 1)
            net_label = net_label.to(device=device).transpose(0, 1)

            # Reset optimizer to clear the previous batch
            optimizer.zero_grad()

            # Generate prediction
            y_hat, state = model(x=net_input)

            # Convert target one hot to indices (required for CE-loss)
            target = net_label[:, 0].data.topk(1)[1][:, 0]

            loss = criterion(y_hat[:, 0], target)
            loss.backward()
            optimizer.step()
            sequence_errors.append(loss.item())

        epoch_errors.append(np.mean(sequence_errors))
        print(epoch_errors)
        # Save the model to file (if desired)
        if cfg.training.save_model and np.mean(sequence_errors) < best_error:
            # Start a separate thread to save the model
            thread = Thread(target=helpers.save_model_to_file(
                model_src_path=os.path.abspath(""),
                cfg=cfg,
                epoch=epoch,
                epoch_errors_train=epoch_errors,
                model=model))
            thread.start()

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors[-1] < best_error:
            train_sign = "(+)"
            best_error = epoch_errors[-1]

        #
        # Print progress to the console
        print(
            f"Epoch {str(epoch+1).zfill(int(np.log10(cfg.training.epochs))+1)}"
            f"/{str(cfg.training.epochs)} took "
            f"{str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} "
            f"seconds. \t\tAverage epoch training error: "
            f"{train_sign}"
            f"{str(np.round(epoch_errors[-1], 10)).ljust(12, ' ')}"
        )

    b = time.time()
    print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')


if __name__ == "__main__":
    th.set_num_threads(1)
    run_training()

    print("Done.")