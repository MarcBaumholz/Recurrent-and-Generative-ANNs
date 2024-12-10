# Import necessary libraries
import glob
import os

import numpy as np
import torch as th

class WaveUnetDataset(th.utils.data.Dataset):
    """
    Dataset for the 2D wave equation.
    In this UNet implementation, each sample has the following properties:
        inpt: length of context_size
        trgt: length of max_rollout_steps
    """

    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        context_size: int,
        max_rollout_steps: int,
        cut_off_first_n_frames: int,
        cut_off_last_n_frames: int,
        use_x_share_of_samples: float = 1.,
        mode: str = "train",
        **kwargs
        ):
        """
        Constructor of a PyTorch dataset module.

        :param data_path: The path to the data directory (without file name)
        :param dataset_name: The name of the specific dataset, e.g., '32x32_slow'
        :param context_size: Determines the length of 'inpt' in each sample
        :param max_rollout_steps: Determines the length of 'trgt' in each sample
        :param cut_off_first_n_frames: Cuts off the first n frames of each wave simulation/run
        :param cut_off_last_n_frames: Cuts off the last n frames of each wave simulation/run
        :param use_x_share_of_samples: Uses a certain share of the data (between 0 and 1)
        :param mode: One of 'train', 'val', or 'test'
        """

        # Set context size and maximum number of rollout steps
        self.context_size = context_size
        self.max_rollout_steps = max_rollout_steps

        # Collect all .npy files from the specified data path
        self.filenames = np.sort(glob.glob(os.path.join(data_path, dataset_name, mode, "*.npy")))
        # Use only a fraction of the available samples based on 'use_x_share_of_samples'
        self.filenames = self.filenames[:int(len(self.filenames) * use_x_share_of_samples)]
        # Load each .npy file into the 'runs' list
        self.runs = [np.load(filename) for filename in self.filenames]

        # Trim each run by removing the first and last specified number of frames
        self.runs = [run[cut_off_first_n_frames: len(run) - cut_off_last_n_frames] for run in self.runs]
        # Get the lengths of all runs after trimming
        run_lengths = [len(run) for run in self.runs]
        # Ensure all runs have the same length; raise an error if they differ
        if len(set(run_lengths)) > 1:
            raise ValueError(f"Arrays in runs have different lengths: {run_lengths}")
        # Calculate effective length of each run after accounting for context and rollout steps
        self.len_one_run_effective = run_lengths[0] - (context_size + max_rollout_steps)
        # Store the total number of runs
        self.number_runs = len(self.runs)
        # Calculate the total number of samples across all runs
        self.sample_num = self.len_one_run_effective * len(self.runs)
        # Raise an error if the effective length is non-positive
        if self.len_one_run_effective <= 0:
            raise ValueError("The sum of context_size and max_rollout_steps is too large for the data length.")

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.sample_num

    def __getitem__(self, item):
        """
        Return:
            inpt: array of length context_size
            trgt: array of length max_rollout_steps
        """
        # Determine which run (simulation) the item belongs to
        run_number = item // self.len_one_run_effective

        # Index of the first input frame within the run
        idx_first_input_frame = item % self.len_one_run_effective

        # Index of the last input frame (exclusive)
        idx_last_input_frame = idx_first_input_frame + self.context_size

        # Index of the first target frame
        idx_first_target_frame = idx_last_input_frame
        # Index of the last target frame (exclusive)
        idx_last_target_frame = idx_first_target_frame + self.max_rollout_steps

        # Get the current run data
        curr_run = self.runs[run_number]
        # Extract the input sequence from the run
        inpt = curr_run[idx_first_input_frame : idx_last_input_frame]
        # Extract the target sequence from the run
        trgt = curr_run[idx_first_target_frame : idx_last_target_frame]

        # Verify that the input sequence has the correct length
        if not len(inpt) == self.context_size:
            raise ValueError('Incorrect len of input')

        # Verify that the target sequence has the correct length
        if not len(trgt) == self.max_rollout_steps:
            raise ValueError('Incorrect len of target')

        # Return the input and target sequences as NumPy arrays of type float32
        return np.array(inpt, dtype=np.float32), np.array(trgt, dtype=np.float32)
