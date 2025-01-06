import torch as th
from torch import nn
import torch.nn.functional as F
from nn.convnext import FiLMConvNeXtBlock, PatchDownscale, PatchUpscale, ConvNeXtStem
from model.base import AbstractLatentDiffusionModel
from utils.utils import MultiArgSequential
import math
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

def get_timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    This is the same as used in Stable Diffusion, DDPM, etc.
    timesteps: a 1D tensor of shape [N], containing the time steps
    dim: the dimension of the output embedding
    """
    # Ensure float32
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(half, dtype=th.float32, device=timesteps.device) / half
    )
    # Shape: [N, half]
    args = timesteps.float()[:, None] * freqs[None]
    # Shape: [N, 2*half] = [N, dim] (assuming dim is even)
    embedding = th.cat([th.sin(args), th.cos(args)], dim=-1)
    if dim % 2 == 1:
        # zero pad if dim is odd
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class FfhqDiffusion(AbstractLatentDiffusionModel):
    def __init__(
        self, 
        cfg,
        vae,
        denoiser_layers=3,
        denoiser_channels=512,
        conditioning_channels=512,
        hyper = False
    ):
        super().__init__(vae, num_timesteps=cfg.num_timesteps)
        self.cfg = cfg
        self.embedding_dim = conditioning_channels
        self.hyper = hyper

        self.conv_in = nn.Conv2d(cfg.latent_size, denoiser_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(denoiser_channels, cfg.latent_size, kernel_size=3, padding=1)

        self.enc1 = MultiArgSequential(
            FiLMConvNeXtBlock(denoiser_channels, conditioning_channels),
            FiLMConvNeXtBlock(denoiser_channels, conditioning_channels),
        )

        self.enc2 = MultiArgSequential(
            PatchDownscale(denoiser_channels, denoiser_channels, kernel_size=2),
            FiLMConvNeXtBlock(denoiser_channels, conditioning_channels),
            FiLMConvNeXtBlock(denoiser_channels, conditioning_channels),
        )

        self.enc3 = MultiArgSequential(
            PatchDownscale(denoiser_channels, denoiser_channels, kernel_size=2),
            FiLMConvNeXtBlock(denoiser_channels, conditioning_channels),
            FiLMConvNeXtBlock(denoiser_channels, conditioning_channels),
            PatchUpscale(denoiser_channels, denoiser_channels, kernel_size=2),
        )

        self.merge2 = nn.Conv2d(denoiser_channels*2, denoiser_channels, kernel_size=3, padding=1)
        self.dec2 = MultiArgSequential(
            FiLMConvNeXtBlock(denoiser_channels, conditioning_channels),
            FiLMConvNeXtBlock(denoiser_channels, conditioning_channels),
            PatchUpscale(denoiser_channels, denoiser_channels, kernel_size=2),
        )

        self.merge1 = nn.Conv2d(denoiser_channels*2, denoiser_channels, kernel_size=3, padding=1)
        self.dec1 = MultiArgSequential(
            FiLMConvNeXtBlock(denoiser_channels, conditioning_channels),
            FiLMConvNeXtBlock(denoiser_channels, conditioning_channels)
        )

    def denoise(self, x: th.Tensor, timestep: th.Tensor):
        
        embedding = get_timestep_embedding(timestep, self.embedding_dim)

        x = self.conv_in(x)
        x1 = self.enc1(x, embedding)[0]
        x2 = self.enc2(x1, embedding)[0]
        x3 = self.enc3(x2, embedding)[0]
        x2 = self.dec2(self.merge2(th.cat([x3, x2], dim=1)), embedding)[0]
        x1 = self.dec1(self.merge1(th.cat([x2, x1], dim=1)), embedding)[0]
        return self.conv_out(x1)
#"""

