import pytorch_lightning as pl
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from utils.optimizers import Ranger
from utils.configuration import Configuration
from utils.io import UEMA, Timer
import torch.distributed as dist

def beta_schedule(period, t, min_value=0, alpha=0.1):
    return max(min_value, min(1, alpha**(t / max(1, period)))*10)

class VAETrainerModule(pl.LightningModule):
    def __init__(self, model, cfg: Configuration, state_dict={}):
        super().__init__()
        self.cfg = cfg

        print(f"RANDOM SEED: {cfg.seed}")
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)

        self.net = model

        print(f"Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

        self.lr = self.cfg.learning_rate
        self.own_loggers = {}
        self.timer = Timer()

        self.register_buffer("num_updates", th.tensor(-1, dtype=th.long))
        self.register_buffer("num_iterations", th.tensor(0, dtype=th.long))

    def forward(self, input):
        return self.net(input)

    def log(self, name, value, on_step=True, on_epoch=True, prog_bar=False, logger=True):
        super().log(name, value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger, sync_dist=True)

        if name not in self.own_loggers:
            self.own_loggers[name] = UEMA(1000)

        self.own_loggers[name].update(value.item() if isinstance(value, th.Tensor) else value)

    def training_step(self, batch, batch_idx):
        self.num_iterations = (self.num_iterations + 1).detach()
        results = self(batch[0])

        kl_beta = beta_schedule(self.cfg.model.beta_decay_period, self.num_updates, self.cfg.model.kl_factor)

        loss = results["loss"] + results['kl_loss'] * kl_beta

        self.log("loss", results['loss'])
        self.log("kl_loss", results['kl_loss'] * kl_beta)
        self.log("cbcr_loss", results['cbcr_loss'])
        self.log("y_loss", results['y_loss'])

        mu, sigma = results['mu'], results['sigma']
        mean_mu         = th.mean(mu[:,0])
        mean_sigma      = th.mean(sigma[:,0])
        std_mu          = th.std(mu[:,0])
        std_sigma       = th.std(sigma[:,0])

        self.log("mean_mu",    mean_mu)
        self.log("mean_sigma", mean_sigma)
        self.log("std_mu",     std_mu)
        self.log("std_sigma",  std_sigma)

        if self.num_iterations % self.cfg.model.gradient_accumulation_steps == 0:
            self.num_updates = (self.num_updates + 1).detach()
            print("Epoch[{}|{}|{}|{:.2f}%]: {}, Loss: {:.2e}, Color-Loss: {:.2e}, Structure-Loss: {:.2e}, KL-Loss: {:.2e}|{:.3f},  Latent: mu: {:.3f} +- {:.3f}, sigma: {:.3f} +- {:.3f}".format(
                self.trainer.local_rank,
                self.num_updates,
                self.trainer.current_epoch,
                (batch_idx + 1) / len(self.trainer.train_dataloader) * 100,
                str(self.timer),
                float(self.own_loggers['loss']),
                float(self.own_loggers['cbcr_loss']),
                float(self.own_loggers['y_loss']),
                float(self.own_loggers['kl_loss']),
                kl_beta,
                float(self.own_loggers['mean_mu']),
                float(self.own_loggers['std_mu']),
                float(self.own_loggers['mean_sigma']),
                float(self.own_loggers['std_sigma']),
            ), flush=True)

        return loss

    def configure_optimizers(self):
        return Ranger([
            {
                'params': self.net.parameters(), 
                'lr': self.cfg.learning_rate, 
                'weight_decay': self.cfg.weight_decay,
                #'betas': (0.9, 0.95),
            },
        ])

