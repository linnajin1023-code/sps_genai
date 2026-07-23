import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Convert diffusion timestep t into a sinusoidal embedding.
    """

    def __init__(self, embedding_dim: int = 64, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2

        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            / half_dim
        )

        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)

        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=1,
        )

        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResidualBlock(nn.Module):
    """
    Convolutional block conditioned on the diffusion timestep.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.time_projection = nn.Linear(
            time_dim,
            out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.skip(x)

        h = self.conv1(x)
        h = self.norm1(h)

        time_features = self.time_projection(time_embedding)
        h = h + time_features[:, :, None, None]

        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + residual


class SimpleUNet(nn.Module):
    """
    Small time-conditioned UNet for CIFAR-10 images.

    Input:
        image: [batch, 3, 32, 32]
        timestep: [batch]

    Output:
        predicted noise: [batch, 3, 32, 32]
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        time_embedding_dim: int = 64,
        time_dim: int = 128,
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(
            image_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # 32 x 32
        self.encoder_block1 = ResidualBlock(
            base_channels,
            base_channels,
            time_dim,
        )

        # 32 x 32 -> 16 x 16
        self.downsample1 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.encoder_block2 = ResidualBlock(
            base_channels * 2,
            base_channels * 2,
            time_dim,
        )

        # 16 x 16 -> 8 x 8
        self.downsample2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.bottleneck = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            time_dim,
        )

        # 8 x 8 -> 16 x 16
        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.decoder_block1 = ResidualBlock(
            base_channels * 4,
            base_channels * 2,
            time_dim,
        )

        # 16 x 16 -> 32 x 32
        self.upsample2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.decoder_block2 = ResidualBlock(
            base_channels * 2,
            base_channels,
            time_dim,
        )

        self.output_layer = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(
                base_channels,
                image_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)

        x = self.input_conv(x)

        skip1 = self.encoder_block1(
            x,
            time_embedding,
        )

        x = self.downsample1(skip1)

        skip2 = self.encoder_block2(
            x,
            time_embedding,
        )

        x = self.downsample2(skip2)

        x = self.bottleneck(
            x,
            time_embedding,
        )

        x = self.upsample1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder_block1(
            x,
            time_embedding,
        )

        x = self.upsample2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder_block2(
            x,
            time_embedding,
        )

        return self.output_layer(x)
