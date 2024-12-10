#! /usr/bin/env python3

import glob
import os

import numpy as np
import torch as th


class WaveDataset(th.utils.data.Dataset):
    """
    Dataset for the 2D wave equation
    """
    # Mapping of keys as short names of the heat flux variables to their corresponding column and full name

    def __init__(
            self,
            data_path: str,
            dataset_name: str,
            mode: str = "train",
            noise: float = 0.0,
            use_x_share_of_samples: float = 1.0,
            **kwargs
        ):
        """
        Constructor of a pytorch dataset module.

        :param data_path: The path to the data directory (without file name)
        :param dataset_name: The name of the specific dataset, e.g., '32x32_slow'
        :param mode: Any of 'train', 'val', or 'test'
        :param noise: Standard deviation of noise that is added to the input
        :param use_x_share_of_samples: Fraction in [0, 1] of the data to be used
        """
        self.noise = noise
        self.filenames = np.sort(glob.glob(os.path.join(data_path, dataset_name, mode, "*.npy")))
        if use_x_share_of_samples < 1.0:
            self.filenames = self.filenames[:int(len(self.filenames)*use_x_share_of_samples)]
                
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        """
        return: Input features and target features of the sample indicated by item.
        """
        sample = np.load(self.filenames[item])
        inpt, trgt = sample[:-1], sample[1:]

        # If specified, add noise to the input
        if self.noise > 0: inpt = inpt + np.random.randn(*inpt.shape)*self.noise
        
        return np.array(inpt, dtype=np.float32), np.array(trgt, dtype=np.float32)


if __name__ == "__main__":

    # Example of creating a WaveDataset for the PyTorch DataLoader
    dataset = WaveDataset(
        data_path=os.path.join("data", "numpy", "32x32_slow"),
        mode="train",
        noise=0.5
    )

    train_dataloader = th.utils.data.DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    for inpts, trgts in train_dataloader:
        print(inpts.shape)
        break

    print("Done.")
