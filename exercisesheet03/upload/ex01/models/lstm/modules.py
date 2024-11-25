#! /usr/bin/env python3

import numpy as np
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F


###########
# CLASSES #
###########

class Model(nn.Module):
	"""
	The actual model consisting of some feed forward and recurrent layers.
	"""

	def __init__(self, d_one_hot, d_lstm, num_lstm_layers, dropout=0.1,
				 bias=True):
		"""
		Constructor method of the Model module.
		:param d_one_hot: The size of the input and output vector
		:param d_lstm: The hidden size of the lstm layers
		:param num_lstm_layers: The number of sequential lstm layers
		:param dropout: Probability of dropping out certain neurons
		:param bias: Whether to use bias neurons
		"""
		super().__init__()

		self.linear_layer_in = nn.Linear(
			in_features=d_one_hot,
			out_features=d_lstm,
			bias=bias
		)
		self.lstm_layers = nn.LSTM(
			input_size=d_lstm,
			hidden_size=d_lstm,
			num_layers=num_lstm_layers,
			bias=bias,
			dropout=dropout
		)
		self.linear_layer_out = nn.Linear(
			in_features=d_lstm,
			out_features=d_one_hot,
			bias=bias
		)

	def forward(self, x, state=None):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:return: The module's output
		"""

		x = self.linear_layer_in(x)
		if state is not None:
			x, (h, c) = self.lstm_layers(x, (state[0], state[1]))
		else:
			x, (h, c) = self.lstm_layers(x)
		x = self.linear_layer_out(x)

		return x, (h, c)
