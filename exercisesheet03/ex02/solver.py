"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from utils import cuda, grid2gif
from model import BetaVAE
from dataset import return_data


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.nc = 3
        self.decoder_dist = 'gaussian'
                
        self.net = cuda(BetaVAE(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None

        self.ckpt_dir = args.ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, self.ckpt_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        self.gather = DataGather()

    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for x in self.data_loader:
                self.global_iter += 1
                pbar.update(1)

                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                loss = reconstruction_loss(x, x_recon, self.decoder_dist)

                if self.beta > 0:
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                    loss = loss + self.beta*total_kld

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint(self.ckpt_name)
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

    def generate_from_latent(self):
        self.net_mode(train=False)
        self.net.to(device="cpu")

        # TODO: Generate images from randomly sampled latent codes.
        #       You may use the torchvision "save_image" function to save images to file.
        num_samples = 16
        z = torch.randn(num_samples, self.z_dim)
        with torch.no_grad():
            output = self.net._decode(z)
        output = torch.sigmoid(output)

        save_path = os.path.join(self.output_dir, f"generated_{self.ckpt_name}.png")
        save_image(make_grid(output, nrow=num_samples //2), save_path)


    def latent_analysis(self):
        self.net_mode(train=False)
        self.net.to(device="cpu")
        
        # TODO: Investigate which latent code corresponds to what feature in the image space.

        num_samples=6
        num_steps=8
        step_size=2.0

        # Permute each dimension
        base_z = torch.randn(num_samples, self.z_dim)
        for dim in range(self.z_dim):
            images = []
            for step in range(-num_steps // 2, num_steps // 2 + 1):
                z = base_z.clone()
                z[:, dim] += step * step_size
                with torch.no_grad():
                    output = self.net._decode(z)
                    output = torch.sigmoid(output)
                images.append(output)
            images = torch.cat(images, dim=0)

            # Reorder to group images based on sample
            group_size = images.size(0) // num_samples
            new_order = torch.arange(images.size(0)).reshape(group_size, num_samples).T.reshape(-1)
            images = images[new_order]

            save_path = os.path.join(self.output_dir, f"analysis_dim_{dim}_{self.ckpt_name}.png")
            save_image(make_grid(images, nrow=num_steps + 1), save_path)

    def rotate(self):
        self.net_mode(train=False)
        self.net.to(device="cpu")
        
        # TODO: Create different views of a given input image
        from PIL import Image
        import os
        from torchvision import transforms
        # Load image -- you may also try loading other images, even one of yourself
        img = np.expand_dims(np.swapaxes(np.swapaxes(np.array(Image.open(
            os.path.join("data", "CelebA", "img_align_celeba", "img_align_celeba", "000007.jpg")
        ).convert("RGB"), dtype=np.float32), 1, 2), 0, 1), axis=0)
        img[0] = img[0] / 255

        # TODO: Rotate the face displayed on the image

        transform = transforms.Compose([
            transforms.Resize((64, 64)) 
        ])
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = transform(img_tensor)

        with torch.no_grad():
            _, mu, _ = self.net(img_tensor)

        rotation_dim = 6 
        num_steps = 16
        step_size = 2.0
        images = []

        for step in range(-num_steps // 2, num_steps // 2 + 1):
            z =mu.clone()
            z[:, rotation_dim] += step * step_size
            with torch.no_grad():
                output = self.net._decode(z)
                output = torch.sigmoid(output)
            images.append(output)

        images = torch.cat(images, dim=0)
        save_path = os.path.join(self.output_dir, f"rotated_{self.ckpt_name}.png")
        save_image(make_grid(images, nrow=num_steps + 1), save_path)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
