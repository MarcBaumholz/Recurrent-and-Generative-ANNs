#! /usr/env/bin python3

import torch as th
import pytorch_msssim


class MyMSELoss(th.nn.Module):
	def __init__(self, **kwargs):
		super(MyMSELoss, self).__init__()
		self.loss = th.nn.MSELoss(**kwargs)

	def forward(self, x: th.Tensor, y: th.Tensor, **kwargs) -> th.Tensor:
		return self.loss(x, y)


class MySSIMLoss(th.nn.Module):
	def __init__(self, **kwargs):
		super(MySSIMLoss, self).__init__()
		self.loss = pytorch_msssim.SSIM(**kwargs)

	def forward(self, x: th.Tensor, y: th.Tensor, **kwargs) -> th.Tensor:
		return 1-self.loss(x, y)


class MSESSIMLoss(th.nn.Module):
	def __init__(self, max_epochs: int, channel: int = 1):
		super(MSESSIMLoss, self).__init__()
		self.max_epochs = max_epochs
		self.mse = th.nn.MSELoss()
		self.ssim = pytorch_msssim.SSIM(channel=channel)

	def forward(self, x: th.Tensor, y: th.Tensor, epoch: int) -> th.Tensor:
		w = epoch/self.max_epochs
		mse = self.mse(x, y)
		ssim = 1-self.ssim(x, y)
		return (1-w)*mse + w*ssim
