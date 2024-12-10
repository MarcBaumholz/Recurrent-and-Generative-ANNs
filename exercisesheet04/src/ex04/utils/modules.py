#! /usr/env/bin python3

from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch as th
from torchvision.ops.misc import Permute
from torchvision.ops.stochastic_depth import StochasticDepth


class ConvNeXtBlock(th.nn.Module):
    """
    PyTorch's implementation of the ConvNeXt block from
    https://pytorch.org/vision/main/_modules/torchvision/models/convnext.html#convnext_tiny
    """
    def __init__(
        self,
        dim: int,
        layer_scale: Optional[float] = 1e-6,
        stochastic_depth_prob: Optional[float] = 0.0,
        norm_layer: Optional[Callable[..., th.nn.Module]] = None,
    ) -> None:
        super(ConvNeXtBlock, self).__init__()

        if norm_layer is None:
            norm_layer = partial(th.nn.LayerNorm, eps=1e-6)

        self.block = th.nn.Sequential(
            th.nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            th.nn.Linear(in_features=dim, out_features=4*dim, bias=True),
            th.nn.GELU(),
            th.nn.Linear(in_features=4*dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = th.nn.Parameter(th.ones(dim, 1, 1)*layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: th.Tensor) -> th.Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class ChannelResize(th.nn.Module):
    """
    A module to perform 2D conv channel growth or shrinkage, depending on the arguments provided.
    """
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        weight_init_fun: th.nn.Module = th.nn.init.dirac_
    ):
        super(ChannelResize, self).__init__()
        self.conv = th.nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1)
        if weight_init_fun is not None: weight_init_fun(self.conv.weight)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.conv(x)
