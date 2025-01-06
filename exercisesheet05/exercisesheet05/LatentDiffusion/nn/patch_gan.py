import torch as th
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as sp_norm
import torch.nn.functional as F

class PatchGANLoss(nn.Module):
    def __init__(
        self, 
        discriminator_start = 2500,
        discriminator_weight = 0.1,
        base_channels = 24,
        rec_loss_downscale_factor = 4,
        rec_loss = nn.MSELoss(),
    ):
        super(PatchGANLoss, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels*2, 5, 2, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 5, 2, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 8, 5, 2, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 8, 1, 5, 1, 0),
        )


        self.discriminator_start  = discriminator_start
        self.discriminator_weight = discriminator_weight

        self.criterion = nn.BCEWithLogitsLoss()
        self.rec_loss = rec_loss
        self.rec_loss_downscale_factor = rec_loss_downscale_factor

        self.register_buffer("num_updates", th.tensor(0))

    def calculate_adaptive_weight(self, rec_loss, g_loss, reconstructions):
        rec_grads = th.autograd.grad(rec_loss, reconstructions, retain_graph=True)[0]
        g_grads = th.autograd.grad(g_loss, reconstructions, retain_graph=True)[0]

        d_weight = th.linalg.norm(rec_grads) / (th.linalg.norm(g_grads) + 1e-8)
        d_weight = th.clamp(d_weight, 0.0, 1e8).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def r1_gradient_penalty(self, real_data):
        real_data.requires_grad_(True)
        disc_real = self.discriminator(real_data)

        gradients = th.autograd.grad(
            outputs=disc_real.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        r1_penalty = (gradients ** 2).mean()
        return r1_penalty

    def forward(self, reconstructions, inputs):
        self.num_updates = (self.num_updates + 1).detach()

        if self.rec_loss_downscale_factor > 1 and self.num_updates.item() >= self.discriminator_start and self.training:
            _reconstructions = F.interpolate(reconstructions, scale_factor=1/self.rec_loss_downscale_factor, mode='bilinear')
            _inputs = F.interpolate(inputs, scale_factor=1/self.rec_loss_downscale_factor, mode='bilinear')
            rec_loss = self.rec_loss(_reconstructions, _inputs).mean()
        else:
            rec_loss = self.rec_loss(reconstructions, inputs).mean()

        if self.num_updates.item() >= self.discriminator_start and self.training:

            # Ensure that reconstructions requires gradients
            reconstructions.requires_grad_(True)

            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())

            # Compute the discriminator's predictions for real and fake images
            preds_real = th.sigmoid(logits_real).detach()
            preds_fake = th.sigmoid(logits_fake).detach()

            # Calculate accuracy
            acc_real = (preds_real > 0.5).float().mean()  # Accuracy for real images
            acc_fake = (preds_fake <= 0.5).float().mean()

            # Update the loss function to use Binary Cross Entropy
            real_labels = th.ones_like(logits_real)
            fake_labels = th.zeros_like(logits_fake)
            d_loss_real = self.criterion(logits_real, real_labels)
            d_loss_fake = self.criterion(logits_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            d_loss = d_loss + 10 * self.r1_gradient_penalty(inputs.detach())
        else:
            d_loss = th.zeros_like(rec_loss)

        if self.num_updates.item() < self.discriminator_start or not self.training:
            return { 
                'loss': rec_loss, 
                'rec_loss': th.zeros_like(rec_loss),
                'g_loss': th.zeros_like(rec_loss),
                'd_loss': th.zeros_like(rec_loss), 
                'acc_real': th.zeros_like(rec_loss),
                'acc_fake': th.zeros_like(rec_loss),
            }

        # Freeze discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False

        # generator update
        logits_fake = self.discriminator(reconstructions)
        
        real_labels = th.ones_like(logits_fake)
        g_loss = self.criterion(logits_fake, real_labels)

        d_weight = 0.01 # self.calculate_adaptive_weight(rec_loss, g_loss, reconstructions)

        # unfreeze discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = True

        return {
            'loss': rec_loss + d_weight * g_loss + d_loss, 
            'rec_loss': rec_loss,
            'g_loss': g_loss * d_weight,
            'd_loss': d_loss,
            'acc_real': acc_real,
            'acc_fake': acc_fake,
        }


