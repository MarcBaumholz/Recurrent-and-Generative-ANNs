#!/usr/bin/env python3

import numpy as np
import torch as th
import os
import datasets


def determine_device():
    """
    This function evaluates whether a GPU is accessible at the system and
    returns it as device to calculate on, otherwise it returns the CPU.
    :return: The device where tensor calculations shall be made on
    """
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # Additional Info when using cuda
    if device.type == "cuda":
        print(th.cuda.get_device_name(0))
        print("Memory Usage:")
        print("\tAllocated:",
              round(th.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("\tCached:   ", round(th.cuda.memory_reserved(0) / 1024 ** 3, 1),
              "GB")
        print()
        
    return device


def build_dataloader(cfg, batch_size):
    """
    This function creates a dataset and return the appropriate dataloader to
    iterate over this dataset
    :param cfg: The general configurations of the model
    :param batch_size: The number of samples per batch
    :return: A PyTorch dataloader object
    """
    
    # Set up a dataset and dataloader
    dataset = datasets.TolkienDataset(
        dataset_name=cfg.dataset.name
    )
    dataloader = th.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
        prefetch_factor=2
    )

    return dataset, dataloader


def save_model_to_file(model_src_path, cfg, epoch, epoch_errors_train,
                       model):
    """
    This function writes the model weights along with the network configuration
    and current performance to file.
    :param model_src_path: The source path where the model will be saved to
    :param cfg: The configurations of the model
    :param epoch: The current epoch
    :param epoch_errors_train: The training epoch errors
    :param model: The actual model
    """

    model_save_path = os.path.join(
        model_src_path, "checkpoints", cfg.model.name
    )

    os.makedirs(model_save_path, exist_ok="True")

    # Save model weights to file
    th.save(model.state_dict(), 
            os.path.join(model_save_path, cfg.model.name + ".pt"))

    # Copy the configurations and add a results entry
    cfg["results"] = {
        "current_epoch": epoch + 1,
        "current_training_error": epoch_errors_train[-1],
        "lowest_train_error": min(epoch_errors_train),
    }

    # Save the configuration and current performance to file
    cfg.save(model_save_path)


def one_hot_to_char(one_hot_vector, alphabet):
    """
    Converts a one hot vector into a character given an alphabet.
    :param one_hot_vector: The one-hot vector
    :param alphabet: The alphabet
    :return: The character which the one_hot_vector represents
    """
    
    # TODO: Generate the charactor from the one-hot vector here
    # Get the index of the maximum value in the one-hot vector
    char_index = np.argmax(one_hot_vector)
    
    # Map the index to the corresponding character in the alphabet
    char = alphabet[char_index]

    return char


def char_to_one_hot(char, alphabet):
    """
    Converts a character into a one-hot vector given an alphabet.
    :param char: The character
    :param alphabet: The alphabet
    :return: The one-hot vector for the given character
    """

    # TODO: Generate the one-hot vector here
    # Initialize a zero vector of the same length as the alphabet
    one_hot_vector = np.zeros(len(alphabet), dtype=np.int8)
    
    # Find the index of the character in the alphabet
    char_index = np.where(alphabet == char)[0]
    
    # Set the corresponding index to 1 if the character exists in the alphabet
    if char_index.size > 0:
        one_hot_vector[char_index[0]] = 1

    return one_hot_vector


def softmax_to_one_hot(soft):
    """
    Converts a softmax output into a discretized one-hot vector, where the
    largest value of the softmax is encoded as one.
    :param soft: The softmax vector
    :return: The discretized one-hot vector
    """
    one_hot_vector = np.zeros(len(soft), dtype=np.int8)
    one_hot_vector[soft.data.topk(1)[1][0].detach().cpu().numpy()] = 1
    return one_hot_vector