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

class TrainerModule(pl.LightningModule):
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
        return self.net.compute_loss(input)

    def log(self, name, value, on_step=True, on_epoch=True, prog_bar=False, logger=True):
        super().log(name, value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger, sync_dist=True)

        if name not in self.own_loggers:
            self.own_loggers[name] = UEMA(1000)

        self.own_loggers[name].update(value.item() if isinstance(value, th.Tensor) else value)

    def training_step(self, batch, batch_idx):
        self.num_iterations = (self.num_iterations + 1).detach()
        loss = self(batch[0])

        self.log("loss", loss)

        if self.num_iterations % self.cfg.model.gradient_accumulation_steps == 0:

            z_offset = self.net.z_offset / self.net.counter
            z_scale = self.net.z_scale / self.net.counter

            self.num_updates = (self.num_updates + 1).detach()
            print("Epoch[{}|{}|{}|{:.2f}%]: {}, L: {:.2e}, z_offset: {:.3f}, z_scale: {:.3f}".format(
                self.trainer.local_rank,
                self.num_updates.item(),
                self.trainer.current_epoch,
                (batch_idx + 1) / len(self.trainer.train_dataloader) * 100,
                str(self.timer),
                float(self.own_loggers['loss']),
                z_offset.item(),
                z_scale.item()
            ), flush=True)

        return loss

    def configure_optimizers(self):
        return Ranger([
            {'params': self.net.parameters(), 'lr': self.cfg.learning_rate, 'weight_decay': self.cfg.weight_decay},
        ])

