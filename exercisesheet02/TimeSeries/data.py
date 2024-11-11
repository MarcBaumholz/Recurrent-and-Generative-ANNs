import numpy as np
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.from_numpy(np.array(self.y[index]))


def generate_memory_task_data(num_samples=1000, sequence_length=30, allowed_positions=None, num_tokens=3):
    if allowed_positions is None:
        allowed_positions = np.arange(sequence_length)
    x = []
    y = []
    for _ in range(num_samples):
        position = np.random.choice(allowed_positions)
        token = np.random.randint(0, num_tokens)
        sequence = np.zeros([sequence_length, num_tokens + 2])  # additional tokens for "empty" and "end of sequence"
        sequence[position, token + 1] = 1
        sequence[-1, -1] = 1  # token that indicates that report is requested
        x.append(sequence)
        y.append(token)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=int)
    return x, y
