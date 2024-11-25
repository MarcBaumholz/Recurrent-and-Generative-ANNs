#!/usr/bin/env python3

import numpy as np
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
	"""
	A specific feed forward module that concists of a relu layer followed by a
	dropout and a linear layer.
	"""

	def __init__(self, d_model, linear_layer_size, dropout=0.1, bias=True):
		"""
		Constructor method of the feed forward module.
		:param linear_layer_size: The internal size of the feed forward module
		:param dropout: Probability of dropping out certain neurons
		:param bias: Whether to use bias neurons
		"""
		super().__init__()

		self.relu_layer = nn.Linear(
			in_features=d_model,
			out_features=linear_layer_size,
			bias=bias
		)
		self.dropout_layer = nn.Dropout(
			p=dropout
		)
		self.linear_layer = nn.Linear(
			in_features=linear_layer_size,
			out_features=d_model,
			bias=bias
		)

	def forward(self, x):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:return: The module's output
		"""
		x = F.relu(self.relu_layer(x))
		x = self.dropout_layer(x)
		x = self.linear_layer(x)
		return x


class MultiHeadAttention(nn.Module):
	"""
	The core component of the transformer realizing the attention mechanism.
	"""

	def __init__(self, n_heads, d_model, dropout=0.1, bias=True):
		"""
		Constructor method of the attention module.
		:param n_heads: The number of attention heads
		:param d_model: The size of the K, V, Q and output vectors
		:param dropout: Probability of dropping out certain neurons
		:param bias: Whether to use bias neurons
		"""
		super().__init__()

		self.d = d_model
		self.h = n_heads
		self.d_per_h = d_model // n_heads

		self.q_layer = nn.Linear(
			in_features=d_model,
			out_features=d_model,
			bias=bias
		)
		self.k_layer = nn.Linear(
			in_features=d_model,
			out_features=d_model,
			bias=bias
		)
		self.v_layer = nn.Linear(
			in_features=d_model,
			out_features=d_model,
			bias=bias
		)
		self.dropout_layer = nn.Dropout(
			p=dropout
		)
		self.linear_layer = nn.Linear(
			in_features=d_model,
			out_features=d_model,
			bias=bias
		)

	def forward(self, x, mask=None):
		"""
		Forward pass of the multi head attention module.
		:param k: Key vector
		:param v: Value vector
		:param q: Query vector
		:param mask: Mask to hide future entries
		:return: The attention weighted output as linear combination of v
		"""

		seq_len, batch_size, _ = x.shape

		q = self.q_layer(x)
		k = self.k_layer(x)
		v = self.v_layer(x)

		q = q.view(seq_len, batch_size, self.h, self.d_per_h)
		k = k.view(seq_len, batch_size, self.h, self.d_per_h)
		v = v.view(seq_len, batch_size, self.h, self.d_per_h)

		q = q.transpose(0, 2)
		k = k.transpose(0, 2)
		v = v.transpose(0, 2)

		v_weighted = self.attention(q=q, k=k, v=v, mask=mask)

		conc = v_weighted.transpose(0, 2).contiguous()
		conc = conc.view(seq_len, batch_size, self.d)
		
		linear_out = self.linear_layer(conc)
		drop_out = self.dropout_layer(linear_out)

		return drop_out

	def attention(self, q, k, v, mask=None):
		"""
		The attention mechanism computing a weighted linear combination of v
		based on the similarity of the according k and v entries.
		:param k: Key vector
		:param v: Value vector
		:param q: Query vector
		:param mask: Mask to hide future entries
		:return: Weighted linear combination v_hat and the attention weights
		"""

		scores = th.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_per_h)

		if mask is not None:
			scores = scores.masked_fill(mask==1, -1e6)

		a = F.softmax(scores, dim=-1)
		a = self.dropout_layer(a)

		v_hat = th.matmul(a, v)

		return v_hat


class DecoderLayer(nn.Module):
	"""
	A decoder layer part (of a Transformer) which predicts next observations
	based on the previous inputs.
	"""
	
	def __init__(self, n_heads, d_model, linear_layer_size, dropout=0.1,
				 bias=True):
		"""
		Constructor method of the attention module.
		:param n_heads: The number of attention heads
		:param d_model: The size of the K, V, Q and output vectors
		:param linear_layer_size: The internal size of the feed forward module
		:param dropout: Probability of dropping out certain neurons
		:param bias: Whether to use bias neurons
		"""
		super().__init__()

		self.mh_attention_layer1 = MultiHeadAttention(
			n_heads=n_heads,
			d_model=d_model
		)
		self.mh_attention_layer2 = MultiHeadAttention(
			n_heads=n_heads,
			d_model=d_model
		)

		self.feed_forward_layer = FeedForward(
			d_model=d_model,
			linear_layer_size=linear_layer_size
		)
		
		self.dropout_layer1 = nn.Dropout(p=dropout)
		self.dropout_layer2 = nn.Dropout(p=dropout)
		self.dropout_layer3 = nn.Dropout(p=dropout)
		
		self.norm_layer1 = nn.LayerNorm(normalized_shape=d_model)
		self.norm_layer2 = nn.LayerNorm(normalized_shape=d_model)
		self.norm_layer3 = nn.LayerNorm(normalized_shape=d_model)

	def forward(self, x, mask):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:param mask: Mask to hide future entries
		:return: The module's output
		"""
		
		x2 = self.norm_layer1(x)
		x = x + self.dropout_layer1(self.mh_attention_layer1(x=x2, mask=mask))
		
		x2 = self.norm_layer2(x)
		x = x + self.dropout_layer2(self.mh_attention_layer2(x=x2, mask=mask))

		x2 = self.norm_layer3(x)
		x = x + self.dropout_layer3(self.feed_forward_layer(x=x2))

		return x

class Model(nn.Module):
	"""
	The actual model consisting of a selected number of sequential decoder
	layers.
	"""

	def __init__(self, n_heads, d_model, linear_layer_size, d_one_hot,
				 dropout=0.1, bias=True):
		"""
		Constructor method of the Model module.
		:param n_heads: The number of attention heads
		:param d_model: The size of the K, V, Q and output vectors
		:param linear_layer_size: The internal size of the feed forward module
		:param d_one_hot: The size of the input and output vector
		:param dropout: Probability of dropping out certain neurons
		:param bias: Whether to use bias neurons
		"""
		super().__init__()

		self.linear_layer_in = nn.Linear(
			in_features=d_one_hot,
			out_features=d_model,
			bias=bias
		)
		self.decoder_layer1 = DecoderLayer(
			n_heads=n_heads,
			linear_layer_size=linear_layer_size,
			d_model=d_model
		)
		self.decoder_layer2 = DecoderLayer(
			n_heads=n_heads,
			linear_layer_size=linear_layer_size,
			d_model=d_model
		)
		self.decoder_layer3 = DecoderLayer(
			n_heads=n_heads,
			linear_layer_size=linear_layer_size,
			d_model=d_model
		)
		self.decoder_layer4 = DecoderLayer(
			n_heads=n_heads,
			linear_layer_size=linear_layer_size,
			d_model=d_model
		)
		self.decoder_layer5 = DecoderLayer(
			n_heads=n_heads,
			linear_layer_size=linear_layer_size,
			d_model=d_model
		)
		self.decoder_layer6 = DecoderLayer(
			n_heads=n_heads,
			linear_layer_size=linear_layer_size,
			d_model=d_model
		)
		self.linear_layer_out = nn.Linear(
			in_features=d_model,
			out_features=d_one_hot,
			bias=bias
		)

	def forward(self, x, mask):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:param mask: Mask to hide future entries
		:return: The module's output
		"""

		x = self.linear_layer_in(x)
		x = self.decoder_layer1(x=x, mask=mask)
		x = self.decoder_layer2(x=x, mask=mask)
		x = self.decoder_layer3(x=x, mask=mask)
		x = self.decoder_layer4(x=x, mask=mask)
		x = self.decoder_layer5(x=x, mask=mask)
		x = self.decoder_layer6(x=x, mask=mask)
		x = self.linear_layer_out(x)

		return x
