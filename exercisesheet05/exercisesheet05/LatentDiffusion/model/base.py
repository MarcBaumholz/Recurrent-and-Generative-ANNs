import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod

def kl_loss(mu, logvar):
    """
    Computes the KL divergence loss between a Gaussian distribution parameterized
    by (mu, logvar) and a standard normal distribution.
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()

class VariationalLayer(nn.Module):
    def __init__(self):
        """
        Variational layer using standard log-variance parameterization.
        
        Args:
            input_dim (int): Number of input channels.
            latent_dim (int): Number of latent dimensions.
        """
        super(VariationalLayer, self).__init__()
        
    def forward(self, mu, logvar):
        
        # Sample from the latent state
        std = th.exp(0.5 * logvar)  # Convert logvar to standard deviation
        noise = th.randn_like(std)
        z = mu + std * noise if self.training else mu
        
        return z, {
            'mu': mu,
            'logvar': logvar,
            'sigma': std,
            'kl_loss': kl_loss(mu, logvar)
        }

class AbstractAutoencoder(nn.Module, ABC):
    def __init__(self, cfg):
        super(AbstractAutoencoder, self).__init__()

        self.bottleneck = VariationalLayer()

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    @abstractmethod
    def calculate_loss(self, output, input, **kwargs):
        pass

    @abstractmethod
    def output_postprocessing(self, output):
        pass

    def forward(self, input: th.Tensor):
        mu, logvar = self.encode(input)
        latent, state = self.bottleneck(mu, logvar)
        decoded = self.decode(latent)
        loss = self.calculate_loss(decoded, input, **state)
        output = self.output_postprocessing(decoded)
        input  = self.output_postprocessing(input)
        return {'output': output, 'input': input,  **loss, **state}


def cosine_beta_schedule(num_timesteps, s=0.008):
    # Timesteps are from 0 to T-1
    # TODO implement cosine beta schedule

    return betas.clamp(0, 0.999)

class AbstractLatentDiffusionModel(nn.Module, ABC):
    def __init__(self, vae, num_timesteps: int = 4000, beta_start: float = 0.0001, beta_end: float = 0.02, use_cosine_schedule=True):
        super().__init__()
        self.vae = vae

        # disable gradienst for the vae
        for param in self.vae.parameters():
            param.requires_grad = False

        if use_cosine_schedule:
            betas = cosine_beta_schedule(num_timesteps)
        else:
            betas = torch.linspace(beta_start, beta_end, num_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.num_timesteps = num_timesteps
        self.v_parameterization = True  # We use v-parameterization

        self.register_buffer('z_offset', torch.zeros(1))
        self.register_buffer('z_scale', torch.ones(1))
        self.register_buffer('counter', torch.zeros(1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            z = mu.detach() #self.vae.bottleneck(mu, logvar)[0].detach()

            # update z_offset and z_scale
            self.z_offset = self.z_offset + z.mean()
            self.z_scale = self.z_scale + z.std()
            self.counter += 1

            z_offset = self.z_offset / self.counter
            z_scale = self.z_scale / self.counter

            return (z - z_offset) / z_scale

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():

            z_offset = self.z_offset / self.counter
            z_scale = self.z_scale / self.counter

            z = z * z_scale + z_offset

            return self.vae.decode(z).detach()

    @abstractmethod
    def denoise(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Now this method returns v_pred if v_parameterization=True.
        """
        pass

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        # Encode image into latent
        z_0 = self.encode(x)
        N       = x.size(0)
        t       = torch.randint(0, self.num_timesteps, (N,), device=x.device)
        noise   = torch.randn_like(z_0)

        alpha_cumprod_t = self.alphas_cumprod[t]
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        while sqrt_alpha_cumprod_t.dim() < z_0.dim():
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)

        # TODO compute z_t and v (target)

        v_pred = self.denoise(z_t, t)  # model predicts v

        # TODO compute loss

        return loss.mean()

    def ddim_step(self, z_t, t, t_next, clip_limit=None):
        # Given z_t and predicted v, compute z_{t_next} using DDIM
        # t and t_next are scalars or the same for all in batch
        # We assume t_next = t-1 in a normal DDIM schedule
        # For simplicity, we'll do a uniform skip with DDIM (no ETA)

        # Compute alpha terms
        alpha_cum_t = self.alphas_cumprod[t]
        alpha_cum_next = self.alphas_cumprod[t_next]
        sqrt_alpha_cum_t = alpha_cum_t.sqrt()
        sqrt_one_minus_alpha_cum_t = (1 - alpha_cum_t).sqrt()
        sqrt_alpha_cum_next = alpha_cum_next.sqrt()
        sqrt_one_minus_alpha_cum_next = (1 - alpha_cum_next).sqrt()

        print(f"Z[{t:03d}|{t_next:03d}]: {z_t.mean().item():.3f} +- {z_t.std().item():.3f}, Alpha: {sqrt_alpha_cum_t.item():.3f}, {sqrt_one_minus_alpha_cum_t.item():.3f}")
        # From v, retrieve x_0 and epsilon:
        v_pred  = self.denoise(z_t, torch.full((z_t.size(0),), t, device=z_t.device, dtype=torch.long))

        # TODO compute x_0 and epsilon

        # DDIM update:
        if t_next == 0:
            z_t_next = x_0
        else:
            z_t_next = sqrt_alpha_cum_next * x_0 + sqrt_one_minus_alpha_cum_next * epsilon
            if clip_limit is not None:
                z_t_next = z_t_next.clamp(-clip_limit, clip_limit)

        return z_t_next

    @torch.no_grad()
    def sample_ddim(self, z_shape: torch.Size, steps: int = None, clip_limit = None) -> torch.Tensor:
        """
        DDIM sampling. If steps < self.num_timesteps, we skip steps evenly.
        """
        device = self.betas.device
        if steps is None:
            steps = self.num_timesteps

        # Create a time grid
        # For simplicity, assume steps divides num_timesteps. Otherwise, use interpolation.
        step_indices = torch.linspace(self.num_timesteps-1, 0, steps, dtype=torch.long)
        z = torch.randn(z_shape, device=device)
        for i in range(steps-1):
            t = step_indices[i].item()
            t_next = step_indices[i+1].item()
            z = self.ddim_step(z, int(t), int(t_next), clip_limit=clip_limit)

        # At the end, decode
        return self.decode(z)

    def ddpm_step(self, z_t: torch.Tensor, t: int, clip_limit=None) -> torch.Tensor:
        """
        One reverse diffusion step in the stochastic DDPM sampler.
        z_t: latent at time t
        t: integer time step in [0, self.num_timesteps-1]
        """
        # Grab parameters from buffers
        beta_t = self.betas[t]  # beta_t
        alpha_t = self.alphas[t]  # alpha_t = 1 - beta_t
        alpha_cum_t = self.alphas_cumprod[t]  # alpha_bar_t
        alpha_cum_t_prev = self.alphas_cumprod_prev[t]  # alpha_bar_{t-1}
        
        sqrt_alpha_cum_t = torch.sqrt(alpha_cum_t)
        sqrt_one_minus_alpha_cum_t = torch.sqrt(1. - alpha_cum_t)

        # Predict v at time t
        v_pred = self.denoise(z_t, torch.full((z_t.size(0),), t, device=z_t.device, dtype=torch.long))
        
        # TODO Retrieve predicted noise eps_pred from v_pred
        
        # TODO Mean of the posterior q(z_{t-1} | z_t)

        print(f"Z[{t:03d}]: {z_t.mean().item():.3f} +- {z_t.std().item():.3f}, Alpha: {sqrt_alpha_cum_t.item():.3f}, {sqrt_one_minus_alpha_cum_t.item():.3f}")
        
        # TODO Sample z_{t-1} ~ q(z_{t-1} | z_t)
        
        if clip_limit is not None:
            z_t_prev = z_t_prev.clamp(-clip_limit, clip_limit)
        
        return z_t_prev


    @torch.no_grad()
    def sample_ddpm(self, z_shape: torch.Size, steps: int = None, clip_limit=None) -> torch.Tensor:
        """
        Stochastic DDPM sampling. If steps < num_timesteps, we skip steps evenly 
        (though sub-sampling in DDPM is less trivial than in DDIM).
        """
        device = self.betas.device
        if steps is None or steps == self.num_timesteps:
            # We'll do the full chain from T-1 down to 0
            step_indices = torch.arange(self.num_timesteps-1, -1, -1, device=device, dtype=torch.long)
        else:
            # If you want fewer steps in DDPM, you'll have to define your skip schedule.
            # For simplicity, uniform skipping:
            step_indices = torch.linspace(self.num_timesteps-1, 0, steps, device=device).long()
        
        # Start from standard Normal noise in latent space
        z = torch.randn(z_shape, device=device)
        
        for t in step_indices:
            z = self.ddpm_step(z, t.item(), clip_limit=clip_limit)

        # Decode the final latent
        x_sampled = self.decode(z)
        return x_sampled

    @torch.no_grad()
    def sample(self, input, z_shape: torch.Size, steps: int = None, clip_limit = None, use_ddim = False) -> torch.Tensor:
        if use_ddim:
            x_sampled = self.sample_ddim(z_shape, steps=steps, clip_limit=clip_limit)
        else:
            x_sampled = self.sample_ddpm(z_shape, steps=steps, clip_limit=clip_limit)
        return self.vae.output_postprocessing(x_sampled)
