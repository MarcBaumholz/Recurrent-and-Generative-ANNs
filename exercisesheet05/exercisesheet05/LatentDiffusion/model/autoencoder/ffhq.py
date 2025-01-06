import torch as th
from torch import nn
import torch.nn.functional as F
from nn.convnext import PatchDownscale, PatchUpscale, ConvNeXtBlock, ConvNeXtStem
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
import numpy as np
from model.base import AbstractAutoencoder
from utils.loss import YCbCrL2SSIMLoss

class FfhqAutoencoder(AbstractAutoencoder):
    def __init__(
        self, 
        cfg,
        channels = 64, 
    ):
        super().__init__(cfg)

        self.loss = YCbCrL2SSIMLoss()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=5, stride=2, padding=2),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(cfg.latent_size, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(channels * 2, channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(channels, channels // 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 3, kernel_size=3, padding=1),
        )


        # Variational Layer
        self.fc_mu = nn.Conv2d(channels * 4, cfg.latent_size, kernel_size=3, padding=1)
        self.fc_logvar = nn.Conv2d(channels * 4, cfg.latent_size, kernel_size=3, padding=1)
        

    def encode(self, x: th.Tensor):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def decode(self, x: th.Tensor):
        return self.decoder(x)
            
    def calculate_loss(self, x: th.Tensor, y: th.Tensor, **kwargs):
        loss, cbcr_loss, y_loss = self.loss(x, y)
        return {
            'loss': loss,
            'cbcr_loss': cbcr_loss,
            'y_loss': y_loss,
        }

    def output_postprocessing(self, x: th.Tensor):
        mean = th.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(x.device)
        std = th.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(x.device)
        
        # Reverse input data normalization
        x = x * std + mean
        
        # Clamp to ensure values are within (0, 1) range
        return th.clamp(x, 0, 1)
