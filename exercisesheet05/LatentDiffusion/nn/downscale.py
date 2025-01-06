import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, repeat, reduce
from typing import Tuple, Union, List

class PositionalEmbedding(nn.Module):
    def __init__(self, num_frequencies, max_std = 0.5, position_limit = 1.0):
        super(PositionalEmbedding, self).__init__()

        self.register_buffer('grid_x', th.zeros(0), persistent=False)
        self.register_buffer('grid_y', th.zeros(0), persistent=False)

        self.num_frequencies = num_frequencies
        self.max_std = max_std
        self.position_limit = position_limit

    def update_grid(self, size: Tuple[int, int]):
        if self.grid_x is None or self.grid_x.shape[2:] != size:

            self.size = size
            H, W = size

            self.min_std = 1/min(size)

            grid_x = th.arange(W, device = self.grid_x.device)
            grid_y = th.arange(H, device = self.grid_y.device)

            self.grid_y, self.grid_x = th.meshgrid(grid_y, grid_x, indexing='ij')

            self.grid_x = (self.grid_x / (W-1)) * 2 - 1
            self.grid_y = (self.grid_y / (H-1)) * 2 - 1

            self.grid_x = self.grid_x.reshape(1, 1, H, W)
            self.grid_y = self.grid_y.reshape(1, 1, H, W)

    def forward(self, input: th.Tensor):
        assert input.shape[1] >= 2 and input.shape[1] <= 4

        x   = rearrange(input[:,0:1], 'b c -> b c 1 1')
        y   = rearrange(input[:,1:2], 'b c -> b c 1 1')
        std = th.zeros_like(x)

        if input.shape[1] == 3:
            std = rearrange(input[:,2:3], 'b c -> b c 1 1')

        if input.shape[1] == 4:
            std = rearrange(input[:,3:4], 'b c -> b c 1 1')

        x   = th.clip(x, -self.position_limit, self.position_limit)
        y   = th.clip(y, -self.position_limit, self.position_limit)
        std = 0.1 / th.clip(std, self.min_std, self.max_std)

        H, W = self.size
        std_y = std.clone()
        std_x = std * (W/H)

        return x, y, std_x, std_y, self.grid_x, self.grid_y, self.num_frequencies


class DepthDecoderStemFunction(Function):
    @staticmethod
    def forward(
        ctx, 
        x, y, std_x, std_y, grid_x, grid_y, num_frequencies, 
        gestalt, mask, weight1, bias1, weight2, bias2, scale_factor
    ):
        # Compute embedding
        norm_grid_x = (grid_x - x) * std_x * np.pi / 2
        norm_grid_y = (grid_y - y) * std_y * np.pi / 2
        
        embedding = []
        for i in range(num_frequencies):
            embedding.append(th.cos(norm_grid_x * 2**i))
            embedding.append(th.cos(norm_grid_y * 2**i))

        embedding = th.cat(embedding, dim=1)

        # Concatenate gestalt and embedding
        input = th.cat((gestalt * mask, embedding * mask), dim=1)
        
        # Reshape input tensor to 2D
        B, C, H, W = input.shape
        input = input.view(B, C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
        input = input.permute(0, 2, 4, 1, 3, 5).reshape(B * H // scale_factor * W // scale_factor, -1)
        
        # First linear layer
        output1 = th.matmul(input, weight1.t()) + bias1
        
        # SiLU activation function
        output2 = output1 * th.sigmoid(output1)
        
        # Second linear layer
        output3 = th.matmul(output2, weight2.t()) + bias2
        output3 = output3.view(B, H // scale_factor, W // scale_factor, -1).permute(0, 3, 1, 2)
        
        # Save tensors for backward pass
        ctx.save_for_backward(
            x, y, std_x, std_y, grid_x, grid_y, 
            gestalt, mask, weight1, bias1, weight2, bias2
        )
        ctx.scale_factor    = scale_factor
        ctx.num_frequencies = num_frequencies
        
        return output3
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, std_x, std_y, grid_x, grid_y, gestalt, mask, weight1, bias1, weight2, bias2 = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        num_frequencies = ctx.num_frequencies

        # Recompute embedding
        norm_grid_x = (grid_x - x) * std_x * np.pi / 2
        norm_grid_y = (grid_y - y) * std_y * np.pi / 2
        
        embedding = []
        for i in range(num_frequencies):
            embedding.append(th.cos(norm_grid_x * 2**i))
            embedding.append(th.cos(norm_grid_y * 2**i))

        embedding = th.cat(embedding, dim=1)
        
        # Recompute input
        input = th.cat((gestalt * mask, embedding * mask), dim=1)
        B, C, H, W = input.shape
        input = input.view(B, C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
        input = input.permute(0, 2, 4, 1, 3, 5).reshape(B * H // scale_factor * W // scale_factor, -1)
        
        # Recompute necessary outputs for backward pass
        output1 = th.matmul(input, weight1.t()) + bias1
        output1_sigmoid = th.sigmoid(output1)
        output2 = output1 * output1_sigmoid

        # Gradients for second linear layer
        grad_output2 = grad_output.permute(0, 2, 3, 1).reshape(B * H // scale_factor * W // scale_factor, -1)
        grad_weight2 = th.matmul(grad_output2.t(), output2)
        grad_bias2   = grad_output2.sum(dim=0)
        grad_output1 = th.matmul(grad_output2, weight2)
        
        # Gradients for SiLU activation function
        grad_silu = grad_output1 * output1_sigmoid + output1 * grad_output1 * output1_sigmoid * (1 - output1_sigmoid)
        
        # Gradients for first linear layer
        grad_weight1 = th.matmul(grad_silu.t(), input)
        grad_bias1   = grad_silu.sum(dim=0)
        
        # Gradients for gestalt and embedding
        #grad_input = th.matmul(grad_silu, weight1).reshape(B, H // scale_factor, W // scale_factor, -1).permute(0, 3, 1, 2)
        grad_input = th.matmul(grad_silu, weight1)
        grad_input = grad_input.reshape(B, H // scale_factor, W // scale_factor, C, scale_factor, scale_factor)
        grad_input = grad_input.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        grad_gestalt_mask, grad_embedding_mask = grad_input[:,:gestalt.shape[1]], grad_input[:,gestalt.shape[1]:]
        
        # Gradient for mask, gestalt and embedding
        grad_mask      = th.sum(grad_gestalt_mask * gestalt, dim=1, keepdim=True) + th.sum(grad_embedding_mask * embedding, dim=1, keepdim=True)
        grad_gestalt   = th.sum(grad_gestalt_mask * mask, dim=(2,3), keepdim=True)
        grad_embedding = grad_embedding_mask * mask

        # Initial grid gradients
        grad_grid_x = grad_grid_y = 0

        # Loop through frequencies and compute gradients for cosine functions
        for i in range(num_frequencies):
            grad_cos_x = -th.sin(norm_grid_x * 2**i) * 2**i
            grad_cos_y = -th.sin(norm_grid_y * 2**i) * 2**i
            
            grad_grid_x += grad_cos_x * grad_embedding[:, 2 * i:2 * i + 1]
            grad_grid_y += grad_cos_y * grad_embedding[:, 2 * i + 1: 2 * i + 2]

        grad_x = th.sum(grad_grid_x * -std_x * np.pi / 2, dim=(2, 3), keepdim=True)
        grad_y = th.sum(grad_grid_y * -std_y * np.pi / 2, dim=(2, 3), keepdim=True)
        grad_std_x = th.sum(grad_grid_x * (grid_x - x) * np.pi / 2, dim=(2, 3), keepdim=True)
        grad_std_y = th.sum(grad_grid_y * (grid_y - y) * np.pi / 2, dim=(2, 3), keepdim=True)

        return (
            grad_x, grad_y, grad_std_x, grad_std_y, None, None, None,
            grad_gestalt, grad_mask, grad_weight1, grad_bias1, grad_weight2, grad_bias2, None
        )


class MemoryEfficientDepthDecoderStem(nn.Module):
    def __init__(self, gestalt_size, base_channels, num_frequencies=8, expand_ratio=4, scale_factor=2):
        super(MemoryEfficientDepthDecoderStem, self).__init__()
        
        self.scale_factor = scale_factor
        self.embedding = PositionalEmbedding(num_frequencies)
        
        self.weight1 = nn.Parameter(th.randn(base_channels * expand_ratio, (gestalt_size + num_frequencies * 2) * scale_factor * scale_factor))
        self.bias1   = nn.Parameter(th.zeros(base_channels * expand_ratio))
        
        self.weight2 = nn.Parameter(th.randn(base_channels, base_channels * expand_ratio))
        self.bias2   = nn.Parameter(th.zeros(base_channels))

        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, position, gestalt, mask):
        self.embedding.update_grid(mask.shape[2:])
        x, y, std_x, std_y, grid_x, grid_y, num_frequencies = self.embedding(position)
        gestalt = rearrange(gestalt, 'b c -> b c 1 1')
        
        return DepthDecoderStemFunction.apply(
            x, y, std_x, std_y, grid_x, grid_y, num_frequencies,
            gestalt, mask, self.weight1, self.bias1, self.weight2, self.bias2, self.scale_factor
        )

class RGBDecoderStemFunction(Function):
    @staticmethod
    def forward(
        ctx, 
        x, y, std_x, std_y, grid_x, grid_y, num_frequencies, 
        gestalt, mask, depth, weight1, bias1, weight2, bias2, scale_factor
    ):
        # Compute embedding
        norm_grid_x = (grid_x - x) * std_x * np.pi / 2
        norm_grid_y = (grid_y - y) * std_y * np.pi / 2
        
        embedding = []
        for i in range(num_frequencies):
            embedding.append(th.cos(norm_grid_x * 2**i))
            embedding.append(th.cos(norm_grid_y * 2**i))

        embedding = th.cat(embedding, dim=1)

        # Concatenate gestalt and embedding
        input = th.cat((gestalt * mask, embedding * mask, depth * mask), dim=1)
        
        # Reshape input tensor to 2D
        B, C, H, W = input.shape
        input = input.view(B, C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
        input = input.permute(0, 2, 4, 1, 3, 5).reshape(B * H // scale_factor * W // scale_factor, -1)
        
        # First linear layer
        output1 = th.matmul(input, weight1.t()) + bias1
        
        # SiLU activation function
        output2 = output1 * th.sigmoid(output1)
        
        # Second linear layer
        output3 = th.matmul(output2, weight2.t()) + bias2
        output3 = output3.view(B, H // scale_factor, W // scale_factor, -1).permute(0, 3, 1, 2)
        
        # Save tensors for backward pass
        ctx.save_for_backward(
            x, y, std_x, std_y, grid_x, grid_y, 
            gestalt, mask, depth, weight1, bias1, weight2, bias2
        )
        ctx.scale_factor    = scale_factor
        ctx.num_frequencies = num_frequencies
        
        return output3
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, std_x, std_y, grid_x, grid_y, gestalt, mask, depth, weight1, bias1, weight2, bias2 = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        num_frequencies = ctx.num_frequencies

        # Recompute embedding
        norm_grid_x = (grid_x - x) * std_x * np.pi / 2
        norm_grid_y = (grid_y - y) * std_y * np.pi / 2
        
        embedding = []
        for i in range(num_frequencies):
            embedding.append(th.cos(norm_grid_x * 2**i))
            embedding.append(th.cos(norm_grid_y * 2**i))

        embedding = th.cat(embedding, dim=1)
        
        # Recompute input
        input = th.cat((gestalt * mask, embedding * mask, depth * mask), dim=1)
        B, C, H, W = input.shape
        input = input.view(B, C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
        input = input.permute(0, 2, 4, 1, 3, 5).reshape(B * H // scale_factor * W // scale_factor, -1)
        
        # Recompute necessary outputs for backward pass
        output1 = th.matmul(input, weight1.t()) + bias1
        output1_sigmoid = th.sigmoid(output1)
        output2 = output1 * output1_sigmoid

        # Gradients for second linear layer
        grad_output2 = grad_output.permute(0, 2, 3, 1).reshape(B * H // scale_factor * W // scale_factor, -1)
        grad_weight2 = th.matmul(grad_output2.t(), output2)
        grad_bias2   = grad_output2.sum(dim=0)
        grad_output1 = th.matmul(grad_output2, weight2)
        
        # Gradients for SiLU activation function
        grad_silu = grad_output1 * output1_sigmoid + output1 * grad_output1 * output1_sigmoid * (1 - output1_sigmoid)
        
        # Gradients for first linear layer
        grad_weight1 = th.matmul(grad_silu.t(), input)
        grad_bias1   = grad_silu.sum(dim=0)
        
        # Gradients for gestalt and embedding
        #grad_input = th.matmul(grad_silu, weight1).reshape(B, H // scale_factor, W // scale_factor, -1).permute(0, 3, 1, 2)
        grad_input = th.matmul(grad_silu, weight1)
        grad_input = grad_input.reshape(B, H // scale_factor, W // scale_factor, C, scale_factor, scale_factor)
        grad_input = grad_input.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        grad_gestalt_mask   = grad_input[:,:gestalt.shape[1]]
        grad_embedding_mask = grad_input[:,gestalt.shape[1]:-1]
        grad_depth_mask     = grad_input[:,-1:]
        
        # Gradient for mask, gestalt and embedding
        grad_mask      = (
            th.sum(grad_gestalt_mask * gestalt, dim=1, keepdim=True) + 
            th.sum(grad_embedding_mask * embedding, dim=1, keepdim=True) +
            th.sum(grad_depth_mask * depth, dim=1, keepdim=True)
        )
        grad_gestalt   = th.sum(grad_gestalt_mask * mask, dim=(2,3), keepdim=True)
        grad_embedding = grad_embedding_mask * mask
        grad_depth     = grad_depth_mask * mask

        # Initial grid gradients
        grad_grid_x = grad_grid_y = 0

        # Loop through frequencies and compute gradients for cosine functions
        for i in range(num_frequencies):
            grad_cos_x = -th.sin(norm_grid_x * 2**i) * 2**i
            grad_cos_y = -th.sin(norm_grid_y * 2**i) * 2**i
            
            grad_grid_x += grad_cos_x * grad_embedding[:, 2 * i:2 * i + 1]
            grad_grid_y += grad_cos_y * grad_embedding[:, 2 * i + 1: 2 * i + 2]

        grad_x = th.sum(grad_grid_x * -std_x * np.pi / 2, dim=(2, 3), keepdim=True)
        grad_y = th.sum(grad_grid_y * -std_y * np.pi / 2, dim=(2, 3), keepdim=True)
        grad_std_x = th.sum(grad_grid_x * (grid_x - x) * np.pi / 2, dim=(2, 3), keepdim=True)
        grad_std_y = th.sum(grad_grid_y * (grid_y - y) * np.pi / 2, dim=(2, 3), keepdim=True)

        return (
            grad_x, grad_y, grad_std_x, grad_std_y, None, None, None,
            grad_gestalt, grad_mask, grad_depth, grad_weight1, grad_bias1, grad_weight2, grad_bias2, None
        )


class MemoryEfficientRGBDecoderStem(nn.Module):
    def __init__(self, gestalt_size, base_channels, num_frequencies=16, expand_ratio=4, scale_factor=2):
        super(MemoryEfficientRGBDecoderStem, self).__init__()
        
        self.scale_factor = scale_factor
        self.embedding = PositionalEmbedding(num_frequencies)
        
        self.weight1 = nn.Parameter(th.randn(base_channels * expand_ratio, (gestalt_size + num_frequencies * 2 + 1) * scale_factor * scale_factor))
        self.bias1   = nn.Parameter(th.zeros(base_channels * expand_ratio))
        
        self.weight2 = nn.Parameter(th.randn(base_channels, base_channels * expand_ratio))
        self.bias2   = nn.Parameter(th.zeros(base_channels))

        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, position, gestalt, mask, depth):
        self.embedding.update_grid(mask.shape[2:])
        x, y, std_x, std_y, grid_x, grid_y, num_frequencies = self.embedding(position)
        gestalt = rearrange(gestalt, 'b c -> b c 1 1')
        
        return RGBDecoderStemFunction.apply(
            x, y, std_x, std_y, grid_x, grid_y, num_frequencies,
            gestalt, mask, depth, self.weight1, self.bias1, self.weight2, self.bias2, self.scale_factor
        )

class PatchDownScaleFunction(Function):
    @staticmethod
    def forward(ctx, input, weight1, bias1, weight2, bias2, scale_factor, residual):
        
        # Reshape input tensor to 2D
        B, C, H, W = input.shape
        permuted_input = input.view(B, C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
        permuted_input = permuted_input.permute(0, 2, 4, 1, 3, 5).reshape(B * H // scale_factor * W // scale_factor, -1)
        
        # First linear layer
        output1 = th.matmul(permuted_input, weight1.t()) + bias1
        
        # SiLU activation function
        output2 = output1 * th.sigmoid(output1)
        
        # Second linear layer
        output3 = th.matmul(output2, weight2.t()) + bias2
        output3 = output3.view(B, H // scale_factor, W // scale_factor, -1).permute(0, 3, 1, 2)
        
        # Save tensors for backward pass
        ctx.save_for_backward(input, weight1, bias1, weight2, bias2)
        ctx.scale_factor = scale_factor
        ctx.residual = residual

        if residual:
            input = reduce(input, 'b c (h s1) (w s2) -> b c h w', 'mean', s1=scale_factor, s2=scale_factor)
            input = repeat(input, 'b c h w -> b (c n) h w', n=output3.shape[1] // C)
            output3 = output3 + input
        
        return output3
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight1, bias1, weight2, bias2 = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        
        # Recompute input
        B, C, H, W = input.shape
        permuted_input = input.view(B, C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
        permuted_input = permuted_input.permute(0, 2, 4, 1, 3, 5).reshape(B * H // scale_factor * W // scale_factor, -1)
        
        # Recompute necessary outputs for backward pass
        output1 = th.matmul(permuted_input, weight1.t()) + bias1
        output1_sigmoid = th.sigmoid(output1)
        output2 = output1 * output1_sigmoid

        # Gradients for second linear layer
        grad_output2 = grad_output.permute(0, 2, 3, 1).reshape(B * H // scale_factor * W // scale_factor, -1)
        grad_weight2 = th.matmul(grad_output2.t(), output2)
        grad_bias2   = grad_output2.sum(dim=0)
        grad_output1 = th.matmul(grad_output2, weight2)
        
        # Gradients for SiLU activation function
        grad_silu = grad_output1 * output1_sigmoid + output1 * grad_output1 * output1_sigmoid * (1 - output1_sigmoid)
        
        # Gradients for first linear layer
        grad_weight1 = th.matmul(grad_silu.t(), permuted_input)
        grad_bias1   = grad_silu.sum(dim=0)
        
        # Gradients for gestalt and embedding
        #grad_input = th.matmul(grad_silu, weight1).reshape(B, H // scale_factor, W // scale_factor, -1).permute(0, 3, 1, 2)
        grad_input = th.matmul(grad_silu, weight1)
        grad_input = grad_input.reshape(B, H // scale_factor, W // scale_factor, C, scale_factor, scale_factor)
        grad_input = grad_input.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if ctx.residual:
            grad_output = reduce(grad_output, 'b (c n) h w -> b c h w', 'sum', n=grad_output.shape[1] // C)
            grad_output = repeat(grad_output, 'b c h w -> b c (h s1) (w s2)', s1=scale_factor, s2=scale_factor) / (scale_factor ** 2)
            grad_input  = grad_input + grad_output
        
        return grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2, None, None


class MemoryEfficientPatchDownScale(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4, scale_factor=2, residual=False):
        super(MemoryEfficientPatchDownScale, self).__init__()
        
        self.scale_factor = scale_factor
        self.residual = residual

        hidden_channels = max(in_channels, out_channels) * expand_ratio
        
        self.weight1 = nn.Parameter(th.randn(hidden_channels, in_channels * scale_factor * scale_factor))
        self.bias1   = nn.Parameter(th.zeros(hidden_channels))
        
        self.weight2 = nn.Parameter(th.randn(out_channels, hidden_channels))
        self.bias2   = nn.Parameter(th.zeros(out_channels))

        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, input):
        return PatchDownScaleFunction.apply(
            input, self.weight1, self.bias1, self.weight2, self.bias2, self.scale_factor, self.residual
        )

class RGBAutoencoderStemFunction(Function):
    @staticmethod
    def forward(
        ctx, 
        x, y, std_x, std_y, grid_x, grid_y, num_frequencies, 
        mask, depth, weight1, bias1, weight2, bias2, scale_factor
    ):
        # Compute embedding
        norm_grid_x = (grid_x - x) * std_x * np.pi / 2
        norm_grid_y = (grid_y - y) * std_y * np.pi / 2
        
        embedding = []
        for i in range(num_frequencies):
            embedding.append(th.cos(norm_grid_x * 2**i))
            embedding.append(th.cos(norm_grid_y * 2**i))

        embedding = th.cat(embedding, dim=1)

        # Concatenate embedding and depth
        input = th.cat((embedding * mask, depth * mask), dim=1)
        
        # Reshape input tensor to 2D
        B, C, H, W = input.shape
        input = input.view(B, C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
        input = input.permute(0, 2, 4, 1, 3, 5).reshape(B * H // scale_factor * W // scale_factor, -1)
        
        # First linear layer
        output1 = th.matmul(input, weight1.t()) + bias1
        
        # SiLU activation function
        output2 = output1 * th.sigmoid(output1)
        
        # Second linear layer
        output3 = th.matmul(output2, weight2.t()) + bias2
        output3 = output3.view(B, H // scale_factor, W // scale_factor, -1).permute(0, 3, 1, 2)
        
        # Save tensors for backward pass
        ctx.save_for_backward(
            x, y, std_x, std_y, grid_x, grid_y, 
            mask, depth, weight1, bias1, weight2, bias2
        )
        ctx.scale_factor    = scale_factor
        ctx.num_frequencies = num_frequencies
        
        return output3
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, std_x, std_y, grid_x, grid_y, mask, depth, weight1, bias1, weight2, bias2 = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        num_frequencies = ctx.num_frequencies

        # Recompute embedding
        norm_grid_x = (grid_x - x) * std_x * np.pi / 2
        norm_grid_y = (grid_y - y) * std_y * np.pi / 2
        
        embedding = []
        for i in range(num_frequencies):
            embedding.append(th.cos(norm_grid_x * 2**i))
            embedding.append(th.cos(norm_grid_y * 2**i))

        embedding = th.cat(embedding, dim=1)
        
        # Recompute input
        input = th.cat((embedding * mask, depth * mask), dim=1)
        B, C, H, W = input.shape
        input = input.view(B, C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
        input = input.permute(0, 2, 4, 1, 3, 5).reshape(B * H // scale_factor * W // scale_factor, -1)
        
        # Recompute necessary outputs for backward pass
        output1 = th.matmul(input, weight1.t()) + bias1
        output1_sigmoid = th.sigmoid(output1)
        output2 = output1 * output1_sigmoid

        # Gradients for second linear layer
        grad_output2 = grad_output.permute(0, 2, 3, 1).reshape(B * H // scale_factor * W // scale_factor, -1)
        grad_weight2 = th.matmul(grad_output2.t(), output2)
        grad_bias2   = grad_output2.sum(dim=0)
        grad_output1 = th.matmul(grad_output2, weight2)
        
        # Gradients for SiLU activation function
        grad_silu = grad_output1 * output1_sigmoid + output1 * grad_output1 * output1_sigmoid * (1 - output1_sigmoid)
        
        # Gradients for first linear layer
        grad_weight1 = th.matmul(grad_silu.t(), input)
        grad_bias1   = grad_silu.sum(dim=0)
        
        # Gradients for embedding, depth and mask
        #grad_input = th.matmul(grad_silu, weight1).reshape(B, H // scale_factor, W // scale_factor, -1).permute(0, 3, 1, 2)
        grad_input = th.matmul(grad_silu, weight1)
        grad_input = grad_input.reshape(B, H // scale_factor, W // scale_factor, C, scale_factor, scale_factor)
        grad_input = grad_input.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        grad_embedding_mask = grad_input[:,:-1]
        grad_depth_mask     = grad_input[:,-1:]
        
        grad_mask = (
            th.sum(grad_embedding_mask * embedding, dim=1, keepdim=True) +
            th.sum(grad_depth_mask * depth, dim=1, keepdim=True)
        )
        grad_embedding = grad_embedding_mask * mask
        grad_depth     = grad_depth_mask * mask

        # Initial grid gradients
        grad_grid_x = grad_grid_y = 0

        # Loop through frequencies and compute gradients for cosine functions
        for i in range(num_frequencies):
            grad_cos_x = -th.sin(norm_grid_x * 2**i) * 2**i
            grad_cos_y = -th.sin(norm_grid_y * 2**i) * 2**i
            
            grad_grid_x += grad_cos_x * grad_embedding[:, 2 * i:2 * i + 1]
            grad_grid_y += grad_cos_y * grad_embedding[:, 2 * i + 1: 2 * i + 2]

        grad_x = th.sum(grad_grid_x * -std_x * np.pi / 2, dim=(2, 3), keepdim=True)
        grad_y = th.sum(grad_grid_y * -std_y * np.pi / 2, dim=(2, 3), keepdim=True)
        grad_std_x = th.sum(grad_grid_x * (grid_x - x) * np.pi / 2, dim=(2, 3), keepdim=True)
        grad_std_y = th.sum(grad_grid_y * (grid_y - y) * np.pi / 2, dim=(2, 3), keepdim=True)

        return (
            grad_x, grad_y, grad_std_x, grad_std_y, None, None, None,
            grad_mask, grad_depth, grad_weight1, grad_bias1, grad_weight2, grad_bias2, None
        )


class MemoryEfficientRGBAutoencoderStem(nn.Module):
    def __init__(self, base_channels, num_frequencies=16, expand_ratio=4, scale_factor=2):
        super(MemoryEfficientRGBAutoencoderStem, self).__init__()
        
        self.scale_factor = scale_factor
        self.embedding = PositionalEmbedding(num_frequencies)
        
        self.weight1 = nn.Parameter(th.randn(base_channels * expand_ratio, (num_frequencies * 2 + 1) * scale_factor * scale_factor))
        self.bias1   = nn.Parameter(th.zeros(base_channels * expand_ratio))
        
        self.weight2 = nn.Parameter(th.randn(base_channels, base_channels * expand_ratio))
        self.bias2   = nn.Parameter(th.zeros(base_channels))

        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, position, mask, depth):
        self.embedding.update_grid(mask.shape[2:])
        x, y, std_x, std_y, grid_x, grid_y, num_frequencies = self.embedding(position)
        
        return RGBAutoencoderStemFunction.apply(
            x, y, std_x, std_y, grid_x, grid_y, num_frequencies,
            mask, depth, self.weight1, self.bias1, self.weight2, self.bias2, self.scale_factor
        )
