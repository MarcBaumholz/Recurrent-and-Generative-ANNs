import os
import torch as th

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.lightning_ffhq import FFHQDataModule
from model.lightning.vae_trainer import VAETrainerModule
from model.lightning.trainer import TrainerModule
from model.autoencoder.ffhq import FfhqAutoencoder
from model.diffusion.ffhq import FfhqDiffusion
from utils.configuration import Configuration
from utils.io import PeriodicCheckpoint

def train_vae(cfg: Configuration, checkpoint_path):
    
    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    model = None
    data_module = None
    
    if cfg.model.data == 'ffhq':
        model = FfhqAutoencoder(cfg = cfg.model)
        data_module = FFHQDataModule(cfg)
    else:
        raise NotImplementedError(f"Data {cfg.model.data} not implemented")

    trainer_module = VAETrainerModule(model, cfg)

    # Load the model from the checkpoint if provided, otherwise create a new model
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        trainer_module = VAETrainerModule.load_from_checkpoint(checkpoint_path, cfg=cfg, model=model)

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=f"out/{cfg.model_path}",
        filename="Model-{epoch:02d}-{loss:.2f}",
        save_top_k=3,
        mode="min",
        verbose=True,
    )

    periodic_checkpoint_callback = PeriodicCheckpoint(
        save_path=f"out/{cfg.model_path}",
        save_every_n_steps=1000,  # Save checkpoint every 1000 global steps
    )

    trainer = pl.Trainer(
        devices=-1, 
        accumulate_grad_batches=cfg.model.gradient_accumulation_steps,
        accelerator='cuda',
        max_epochs=cfg.epochs,
        #callbacks=[checkpoint_callback, periodic_checkpoint_callback],
        callbacks=[periodic_checkpoint_callback],
        enable_checkpointing=False,
        precision=16 if cfg.model.mixed_precision else 32,
        enable_progress_bar=False,
        logger=False
    )

    if cfg.validate:
        trainer.validate(trainer_module, data_module)
    else:
        trainer.fit(trainer_module, data_module)


def train_denoiser(cfg: Configuration, checkpoint_path_vae, checkpoint_path):
    
    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    model = None
    data_module = None
    
    if cfg.model.data == 'ffhq':
        # load and remove net. prefix from keys
        vae_state  = th.load(checkpoint_path_vae)['state_dict']
        vae_state = {k.replace('net.', ''): v for k, v in vae_state.items()}

        vae = FfhqAutoencoder(cfg = cfg.model)
        vae.load_state_dict(vae_state, strict=False)

        model = FfhqDiffusion(cfg = cfg.model, vae=vae)
        data_module = FFHQDataModule(cfg)
    else:
        raise NotImplementedError(f"Data {cfg.model.data} not implemented")

    trainer_module = TrainerModule(model, cfg)

    # Load the model from the checkpoint if provided, otherwise create a new model
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        trainer_module = TrainerModule.load_from_checkpoint(checkpoint_path, cfg=cfg, model=model)

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=f"out/{cfg.model_path}",
        filename="Model-{epoch:02d}-{loss:.2f}",
        save_top_k=3,
        mode="min",
        verbose=True,
    )

    periodic_checkpoint_callback = PeriodicCheckpoint(
        save_path=f"out/{cfg.model_path}",
        save_every_n_steps=1000,  # Save checkpoint every 1000 global steps
    )

    trainer = pl.Trainer(
        devices=-1, 
        accumulate_grad_batches=cfg.model.gradient_accumulation_steps,
        accelerator='cuda',
        max_epochs=cfg.epochs,
        #callbacks=[checkpoint_callback, periodic_checkpoint_callback],
        callbacks=[periodic_checkpoint_callback],
        enable_checkpointing=False,
        precision=16 if cfg.model.mixed_precision else 32,
        enable_progress_bar=False,
        logger=False
    )

    if cfg.validate:
        trainer.validate(trainer_module, data_module)
    else:
        trainer.fit(trainer_module, data_module)

