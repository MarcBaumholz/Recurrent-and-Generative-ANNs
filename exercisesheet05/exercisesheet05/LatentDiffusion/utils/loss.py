import torch as th
from torch import nn
from utils.utils import RGB2YCbCr

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return th.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)  

class L1SSIMLoss(nn.Module):
    def __init__(self, ssim_factor = 0.85):
        super(L1SSIMLoss, self).__init__()

        self.ssim = SSIM()
        self.ssim_factor = ssim_factor

    def forward(self, output, target):
        
        l1    = th.abs(output - target)
        ssim  = self.ssim(output, target)

        f = self.ssim_factor

        return th.mean(l1 * (1 - f) + ssim * f), th.mean(l1), th.mean(ssim)

class YCbCrL2SSIMLoss(nn.Module):
    def __init__(self):
        super(YCbCrL2SSIMLoss, self).__init__()

        self.to_YCbCr = RGB2YCbCr()
        self.ssim     = SSIM()
    
    def forward(self, x, y):

        x = self.to_YCbCr(x)

        with th.no_grad():
            y = self.to_YCbCr(y.detach()).detach()

        y_loss  = th.mean(self.ssim(x[:,0:1], y[:,0:1]))
        cb_loss = th.mean((x[:,1] - y[:,1])**2) * 10
        cr_loss = th.mean((x[:,2] - y[:,2])**2) * 10

        loss = y_loss + cb_loss + cr_loss
        return loss, cb_loss + cr_loss, y_loss

class YCbCrL1SSIMLoss(nn.Module):
    def __init__(self, factor_Y = 0.95, factor_Cb = 0.025, factor_Cr = 0.025):
        super(YCbCrL1SSIMLoss, self).__init__()
        self.factor_Y  = factor_Y
        self.factor_Cb = factor_Cb
        self.factor_Cr = factor_Cr

        self.to_YCbCr = RGB2YCbCr()
        self.l1ssim   = L1SSIMLoss()
    
    def forward(self, x, y):
        x = self.to_YCbCr(x)

        with th.no_grad():
            y = self.to_YCbCr(y.detach()).detach()

        y_loss, l1, ssim = self.l1ssim(x[:,0:1], y[:,0:1])

        cb_loss = th.mean((x[:,1] - y[:,1])**2)
        cr_loss = th.mean((x[:,2] - y[:,2])**2)

        cr_factor = (y_loss / cr_loss).detach() * self.factor_Cr
        cb_factor = (y_loss / cb_loss).detach() * self.factor_Cb
        
        sum_factors = cr_factor + cb_factor + self.factor_Y

        y_factor  = self.factor_Y  / sum_factors
        cb_factor = cb_factor / sum_factors
        cr_factor = cr_factor / sum_factors

        loss = y_loss * y_factor + cb_loss * cb_factor + cr_loss * cr_factor

        return loss, l1, ssim

