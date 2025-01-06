import os
import torch as th
import numpy as np
from PIL import Image

import pytorch_lightning as pl

from model.autoencoder.ffhq import FfhqAutoencoder
from model.diffusion.ffhq import FfhqDiffusion
from model.lightning.trainer import TrainerModule
from utils.configuration import Configuration
from data.lightning_ffhq import FFHQDataModule
from tqdm import tqdm

def sample(cfg: Configuration, checkpoint_path, output_path='samples', num_samples=1, steps=50, device=0, use_ddim=False, clip_limit=1):
    os.makedirs(output_path, exist_ok=True)

    # Set device
    device = th.device(f"cuda:{device}" if th.cuda.is_available() else "cpu")

    # Initialize the model
    if cfg.model.data == 'ffhq':

        data_module = FFHQDataModule(cfg)
        data_module.setup(stage='test')
        test_loader = data_module.test_dataloader()

        model_state  = th.load(checkpoint_path)['state_dict']
        model_state = {k.replace('net.', ''): v for k, v in model_state.items()}

        vae = FfhqAutoencoder(cfg = cfg.model)
        vae.eval()

        model = FfhqDiffusion(cfg = cfg.model, vae=vae)
        model.load_state_dict(model_state, strict=False)
        model = model.to(device)
        model.eval()
    else:
        raise NotImplementedError(f"Data {cfg.model.data} not implemented")


    # get first batch
    batch = next(iter(test_loader))[0].to(device)

    latent_shape = (num_samples,cfg.model.latent_size, cfg.model.image_height // 8, cfg.model.image_width // 8)

    # Sample from the diffusion model
    with th.no_grad():
        output = model.sample(batch, latent_shape, steps=steps, clip_limit=clip_limit, use_ddim=use_ddim)

    # Process and save the output images using PIL
    for i in tqdm(range(num_samples)):
        output_image = output[i].cpu()

        np_image = output_image.numpy().transpose(1, 2, 0) * 255  # Convert to [0, 255]
        np_image = np_image.astype(np.uint8)

        if np_image.shape[2] == 1:
            np_image = np_image[:, :, 0]

        pil_image = Image.fromarray(np_image)

        pil_image.save(os.path.join(output_path, f'sample_{i + 1}.png'))

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("-load", "--load", required=True, type=str, help="Path to the model checkpoint")
    parser.add_argument("-output", "--output", default="diffusion-samples", type=str, help="Output directory for samples")
    parser.add_argument("-num_samples", "--num_samples", default=1, type=int, help="Number of samples to generate")
    parser.add_argument("-steps", "--steps", default=100, type=int, help="Number of diffusion steps")
    parser.add_argument("-device", "--device", default=0, type=int, help="CUDA device ID")
    parser.add_argument("-seed", "--seed", default=42, type=int, help="Random seed")
    parser.add_argument("-ddim", "--ddim", default=False, action="store_true", help="Use DDIM for sampling")
    parser.add_argument("-clip_limit", "--clip_limit", default=100, type=float, help="Clip limit for the diffusion model")

    args = parser.parse_args(sys.argv[1:])
    cfg = Configuration(args.cfg)

    cfg.seed = args.seed
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)

    # if we are using ddpm set sample steps to cfg.model.num_timesteps
    if not args.ddim:
        args.steps = cfg.model.num_timesteps
        print(f"Setting steps to {args.steps} for DDPM sampling")

    sample(
        cfg=cfg,
        checkpoint_path=args.load,
        output_path=args.output,
        num_samples=args.num_samples,
        steps=args.steps,
        device=args.device,
        use_ddim=args.ddim,
        clip_limit=args.clip_limit
    )
