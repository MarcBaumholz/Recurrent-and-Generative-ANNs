import einops
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.transforms as T


class LayerNormChannelOnly(nn.Module):
    """
    Layer norm for ConvNeXt
    This LayerNorm only normalizes across the channel vector for each spatial pixel.
    Assumes input tensor to have shape b x c x h x w
    Implemented as in original ConvNeXt code:
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    # TODO: Don't touch this
    """
    def __init__(self, c: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.layer_norm(x)
        x = einops.rearrange(x, 'b h w c -> b c h w')
        return x


class SkipConnectionConvNext(nn.Module):
    """
    Skip Connection for the ConvNeXt Blocks. - No need to use this for the UNet skip connections
    # TODO: Don't touch this
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            spatial_factor: float = 1.0
    ):
        super(SkipConnectionConvNext, self).__init__()
        assert spatial_factor == 1 or int(spatial_factor) > 1 or int(
            1 / spatial_factor) > 1, f'invalid spatial scale factor in SpikeFunction: {spatial_factor}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_factor = spatial_factor

    def channel_skip(self, input: torch.Tensor):
        in_channels = self.in_channels
        out_channels = self.out_channels

        if in_channels == out_channels:
            return input

        if in_channels % out_channels == 0 or out_channels % in_channels == 0:

            if in_channels > out_channels:
                return einops.reduce(input, 'b (c n) h w -> b c h w', 'mean', n=in_channels // out_channels)

            if out_channels > in_channels:
                return einops.repeat(input, 'b c h w -> b (c n) h w', n=out_channels // in_channels)
        else:
            raise ValueError('in_channels % out_channels is not 0')


        mean_channels = np.gcd(in_channels, out_channels)
        input = einops.reduce(input, 'b (c n) h w -> b c h w', 'mean', n=in_channels // mean_channels)
        return einops.repeat(input, 'b c h w -> b (c n) h w', n=out_channels // mean_channels)

    def scale_skip(self, input: torch.Tensor):
        spatial_factor = self.spatial_factor

        if spatial_factor == 1:
            return input

        if spatial_factor > 1:
            return einops.repeat(
                input,
                'b c h w -> b c (h h2) (w w2)',
                h2=int(spatial_factor),
                w2=int(spatial_factor)
            )

        height = input.shape[2]
        width = input.shape[3]

        # scale factor < 1
        spatial_factor = int(1 / spatial_factor)

        if width % spatial_factor == 0 and height % spatial_factor == 0:
            return einops.reduce(
                input,
                'b c (h h2) (w w2) -> b c h w',
                'mean',
                h2=spatial_factor,
                w2=spatial_factor
            )

        if width >= spatial_factor and height >= spatial_factor:
            return nn.functional.avg_pool2d(
                input,
                kernel_size=spatial_factor,
                stride=spatial_factor
            )

        assert width > 1 or height > 1
        return einops.reduce(input, 'b c h w -> b c 1 1', 'mean')

    def forward(self, input: torch.Tensor):

        if self.spatial_factor > 1:
            return self.scale_skip(self.channel_skip(input))

        return self.channel_skip(self.scale_skip(input))


class ConvNext(nn.Module):
    """
    ConvNext Block
    c: num of channels
    num of input and output channels are equal
    # TODO: No need to touch this
    """
    def __init__(self, c: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(c, c*4, kernel_size=7, groups=c, stride=1, padding='same')  # groups=c makes this a depth wise convolution
        self.layer_norm = LayerNormChannelOnly(c*4)
        self.conv_2 = nn.Conv2d(c*4, c, kernel_size=1)
        self.conv_3 = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x: torch.Tensor):
        skip = x
        x = self.conv_1(x)
        x = self.layer_norm(x)
        x = self.conv_2(x)
        x = F.leaky_relu(x)
        x = self.conv_3(x)
        x += skip
        return x


class DownScale(nn.Module):
    """
    Downscaling block
    """
    def __init__(self, c_in: int, c_out: int, spatial_factor: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=spatial_factor, padding=1)
        self.norm = LayerNormChannelOnly(c_out)  # Use LayerNorm for stability
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class UpScale(nn.Module):
    """
    Upscaling block
    """
    def __init__(self, c_in: int, c_out: int, spatial_factor: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(c_in, c_out, kernel_size=spatial_factor, stride=spatial_factor, padding=0)
        self.norm = LayerNormChannelOnly(c_out)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.upsample(x)
        x = self.norm(x)
        x = self.relu(x)
        return x



class EncoderModule(nn.Module):
    """
    Encoder Module
    """
    def __init__(self, c_in: int, c_out: int, spatial_factor: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.Sequential(*[ConvNext(c_out) for _ in range(num_blocks)])
        self.downscale = DownScale(c_in, c_out, spatial_factor)

    def forward(self, x: torch.Tensor):
        x = self.downscale(x)
        x = self.blocks(x)
        return x


class DecoderModule(nn.Module):
    """
    One block of the Decoder that includes two ConvNext Block and an upsampling block.
    num_blocks: number of ConvNext blocks in the module (0 means only upsampling).
    First up-scaling block is applied, then the given number of convnext blocks
    """
    def __init__(self, c_in: int, c_out: int, spatial_factor: int, num_blocks: int):
        super().__init__()
        self.upsample = UpScale(c_in, c_out, spatial_factor)
        self.blocks = nn.Sequential(*[ConvNext(c_out) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor):
        x = self.upsample(x)
        x = x + skip_connection  # Add skip connection
        x = self.blocks(x)
        return x


class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, c_list: list[int], spatial_factor_list: list[int], num_blocks_list: list[int]):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderModule(c_list[i], c_list[i + 1], spatial_factor_list[i], num_blocks_list[i])
            for i in range(len(spatial_factor_list))
        ])

    def forward(self, x: torch.Tensor):
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)  # Save skip connections
        return skip_connections


class Decoder(nn.Module):
    """
    Decoder
    This returns the whole list for the skip connections
    """
    def __init__(self, c_list: list[int], spatial_factor_list: list[int], num_blocks_list: list[int]):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderModule(c_list[i + 1], c_list[i], spatial_factor_list[i], num_blocks_list[i])
            for i in range(len(spatial_factor_list))
        ])

    def forward(self, skip_connections: list):
        x = skip_connections.pop()  # Start with the last encoder output
        for layer in self.layers:
            skip_connection = skip_connections.pop()
            x = layer(x, skip_connection)
        return x


class ConvNeXtUNet(nn.Module):
    """
    ConvNeXt Unet

    Args:
        c_in:   Number of input channels (is upscaled to c_list[0]).
        c_out:  Number of output channels (is downscaled from c_list[0]).
        c_list: List that includes the number of channels that each scaling module should input/output.
                The length of the list corresponds to the (number of downscalings = number of upscalings) + 1.
        spatial_factor_list: List that includes the spatial downscaling/upscaling factors for each down/upscaling
                             module. The length of the list corresponds to the number of downscalings/upscalings.
        num_block_list: List of the number of ConvNeXt blocks for each downscaling/upscaling. The downsampling/upsampling
                         block itself is not included in the number. The length of the list corresponds to the number of
                         downscalings/upscalings.

    Example usage:
        model = ConvNeXtUNet(
            c_in=2,
            c_out=1,
            c_list=[4, 32, 64, 128, 256],
            spatial_factor_list=[2, 2, 2, 2],
            num_blocks_list=[2, 2, 2, 2],
        )
    """

    def __init__(
            self,
            c_in: int,
            c_out: int,
            c_list: list[int],
            spatial_factor_list: list[int],
            num_block_list: list[int],
            **kwargs,
    ):
        super().__init__()
        self.c_in = c_in
        if not len(c_list) - 1 == len(spatial_factor_list) == len(num_block_list):
            raise ValueError(
                'The length of (c_list-1) and length of spatial_factor_list and num_block_list must be equal as they '
                'correspond to the number of downscalings in the network.'
            )
        self.upscale_input_channels = nn.Conv2d(c_in, c_list[0], kernel_size=1)
        self.encoder = Encoder(c_list, spatial_factor_list, num_block_list)
        self.decoder = Decoder(c_list, spatial_factor_list, num_block_list)
        self.downscale = nn.Conv2d(c_list[0], c_out, kernel_size=1)


    def forward_one_step(self, x: torch.Tensor):
        if x.shape[1] != self.c_in:
            raise ValueError(
                f'Size of channel dim in network input is {x.shape[1]} but expected {self.c_in}'
            )
        x = self.upscale_input_channels(x)
        skip_list = self.encoder(x)
        x = self.decoder(skip_list)
        x = self.downscale(x)
        return x


    def forward(
            self,
            x: torch.Tensor,
            context_size: int = 4,
            epoch: int = 200,
            target_len: int = 3,
            **kwargs
    ):
        '''
        args:
            x: Input tensor
            rollout_steps: Number of roll-outs in closed loop during training. 1 is no roll out
        '''
        
        # TODO: possibly change rollout steps here

        if "inference_steps" in list(kwargs.keys()):
            rollout_steps = kwargs["inference_steps"]
            x = x[:, :context_size]
            outs = [x[:, t] for t in range(context_size)]
        else:
            outs = []

        # Replace the empty channel dim with the frame dim (we put the frames into the channels)
        x = x.squeeze()
        # Closed loop: auto-regressive roll-out using previous model outputs as input
        for t in range(rollout_steps):
            prediction = self.forward_one_step(x)
            x = torch.cat([x[:, 1:], prediction], dim=1)
            outs.append(prediction)

        if "inference_steps" in list(kwargs.keys()):
            return torch.stack(outs, dim=1)
        else:
            # Training and validation: fill up the output with zeros to align with target shape
            for t in range(target_len-len(outs)): outs.append(torch.zeros_like(outs[0]))
            return torch.stack(outs, dim=1)
