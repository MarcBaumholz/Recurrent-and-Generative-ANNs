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
from modules import TemporalConvNet
from configuration import Configuration
import helper_functions as helpers


def run_training():

    # Load the user configurations
    cfg = Configuration("config.json")

    # Print some information to console
    print("Model name:", cfg.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config
    # file
    if cfg.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Set device on GPU if specified in the configuration file, else CPU
    device = helpers.determine_device()
    
    # Initialize and set up the model
    model = TemporalConvNet(config=cfg).to(device=device)

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    #
    # Set up an optimizer and the criterion (loss)
    optimizer = th.optim.Adam(model.parameters(),
                              lr=cfg.training.learning_rate)
    #criterion = nn.NLLLoss(reduction="mean")
    criterion = nn.CrossEntropyLoss()

    #
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    best_train = np.infty

    #
    # Set up the dataloader
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
            # [batch_size, time, dim] to [batch_size, dim, time]
            net_input = net_input.to(device=device).transpose(1, 2)
            net_label = net_label.to(device=device).transpose(1, 2)

            # Reset optimizer to clear the previous batch
            optimizer.zero_grad()

            # Generate prediction
            y_hat = model(x=net_input)

            # Convert target one hot to indices (required for CE-loss)
            target = net_label[:, 0].data.topk(1)[1][:, 0]

            loss = criterion(y_hat[:, 0], target)
            loss.backward()
            optimizer.step()
            sequence_errors.append(loss.item())

        epoch_errors_train.append(np.mean(sequence_errors))

        # Save the model to file (if desired)
        if cfg.training.save_model and np.mean(sequence_errors) < best_train:
            # Start a separate thread to save the model
            thread = Thread(target=helpers.save_model_to_file(
                model_src_path=os.path.abspath(""),
                cfg=cfg,
                epoch=epoch,
                epoch_errors_train=epoch_errors_train,
                model=model))
            thread.start()

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors_train[-1] < best_train:
            train_sign = "(+)"
            best_train = epoch_errors_train[-1]
        
        #
        # Print progress to the console
        print(
            f"Epoch {str(epoch+1).zfill(int(np.log10(cfg.training.epochs))+1)}"
            f"/{str(cfg.training.epochs)} took "
            f"{str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} "
            f"seconds. \t\tAverage epoch training error: "
            f"{train_sign}"
            f"{str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')}"
        )

    b = time.time()
    print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')


if __name__ == "__main__":
    th.set_num_threads(1)
    run_training()

    print("Done.")