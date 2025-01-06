import torch as th
from torch import nn

# RGB to YCbCr
class RGB2YCbCr(nn.Module):
    def __init__(self):
        super(RGB2YCbCr, self).__init__()

        kr = 0.299
        kg = 0.587
        kb = 0.114

        # The transformation matrix from RGB to YCbCr (ITU-R BT.601 conversion)
        self.register_buffer("matrix", th.tensor([
            [                  kr,                  kg,                    kb],
            [-0.5 * kr / (1 - kb), -0.5 * kg / (1 - kb),                  0.5],
            [                 0.5, -0.5 * kg / (1 - kr), -0.5 * kb / (1 - kr)]
        ]).t(), persistent=False)

        # Adjustments for each channel
        self.register_buffer("shift", th.tensor([0., 0.5, 0.5]), persistent=False)

    def forward(self, img):
        if len(img.shape) != 4 or img.shape[1] != 3:
            raise ValueError('Input image must be 4D tensor with a size of 3 in the second dimension.')

        return th.tensordot(img.permute(0, 2, 3, 1), self.matrix, dims=1).permute(0, 3, 1, 2) + self.shift[None, :, None, None]

# YCbCr to RGB
class YCbCr2RGB(nn.Module):
    def __init__(self):
        super(YCbCr2RGB, self).__init__()

        kr = 0.299
        kg = 0.587
        kb = 0.114

        # The transformation matrix from YCbCr to RGB (ITU-R BT.601 conversion)
        self.register_buffer("matrix", th.tensor([
            [1,                       0,              2 - 2 * kr],
            [1, -kb / kg * (2 - 2 * kb), -kr / kg * (2 - 2 * kr)],
            [1,              2 - 2 * kb,                       0]
        ]).t(), persistent=False)

        # Adjustments for each channel
        self.register_buffer("shift", th.tensor([0., 0.5, 0.5]), persistent=False)

    def forward(self, img):
        if len(img.shape) != 4 or img.shape[1] != 3:
            raise ValueError('Input image must be 4D tensor with a size of 3 in the second dimension.')

        result = th.tensordot((img - self.shift[None, :, None, None]).permute(0, 2, 3, 1), self.matrix, dims=1).permute(0, 3, 1, 2)

        # Clamp the results to the valid range for RGB [0, 1]
        return th.clamp(result, 0, 1)

class MultiArgSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(MultiArgSequential, self).__init__(*args, **kwargs)

    def forward(self, *tensor):

        for n in range(len(self)):
            if isinstance(tensor, th.Tensor) or tensor == None:
                tensor = self[n](tensor)
            else:
                tensor = self[n](*tensor)

        return tensor
