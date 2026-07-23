import torch
import torch.nn as nn


class EnergyModel(nn.Module):
    """
    Convolutional Energy-Based Model for CIFAR-10.

    Input:
        images: [batch_size, 3, 32, 32]

    Output:
        one scalar energy value for each image
        shape: [batch_size]
    """

    def __init__(self, image_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                image_channels,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
        )

        self.energy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        energy = self.energy_head(features)

        return energy.squeeze(-1)


def langevin_sample(
    model: EnergyModel,
    images: torch.Tensor,
    steps: int = 20,
    step_size: float = 0.1,
    noise_scale: float = 0.01,
) -> torch.Tensor:
    """
    Modify the input images so that the model assigns them lower energy.

    The model parameters remain fixed.
    Gradients are calculated with respect to the images.
    """

    model.eval()

    samples = images.detach().clone()

    for _ in range(steps):
        samples.requires_grad_(True)

        energy = model(samples).sum()

        image_gradient = torch.autograd.grad(
            outputs=energy,
            inputs=samples,
            create_graph=False,
        )[0]

        with torch.no_grad():
            noise = noise_scale * torch.randn_like(samples)

            samples = (
                samples
                - step_size * image_gradient
                + noise
            )

            samples.clamp_(-1.0, 1.0)

        samples = samples.detach()

    return samples
