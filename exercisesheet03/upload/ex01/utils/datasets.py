"""
This file contains the organization of the custom dataset such that it can be
read efficiently in combination with the DataLoader from PyTorch to prevent that
data reading and preparing becomes the bottleneck.

This script was inspired by
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import unittest
import torch as th
import numpy as np
import netCDF4 as nc
import xarray as xr
import os
import glob


class TolkienDataset(th.utils.data.Dataset):
    """
    The custom Tolkien dataset class which can be used with PyTorch's
    DataLoader.
    """

    def __init__(self, dataset_name):
        """
        Constructor class setting up the data loader
        :param dataset_name: The name of the dataset (e.g. "chapter1")
        :return: No return value
        """

        data_root_path = os.path.join(
            os.path.abspath("../.."), "data", dataset_name
        )
        self.data_paths = np.sort(
            glob.glob(os.path.join(data_root_path, "sample*.npy"))
        )
        self.alphabet = np.load(os.path.join(data_root_path, "alphabet.npy"))

    def __len__(self):
        """
        Denotes the total number of samples that exist
        :return: The number of samples
        """
        return len(self.data_paths)

    def __getitem__(self, index):
        """
        Generates a sample batch in the form [batch_size, time, dim],
        where x and y are the sizes of the data and dim is the number of
        features.
        :param index: The index of the sample in the path array
        :return: One batch of data as np.array
        """

        # Load a sample from file and divide it in input and label. The label
        # is the input shifted one timestep to train one step ahead prediction.
        sample = np.float32(np.load(self.data_paths[index]))
        net_input = np.copy(sample[:-1])
        net_label = np.copy(sample[1:])

        return net_input, net_label
