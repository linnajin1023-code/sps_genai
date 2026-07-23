import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.energy_model import EnergyModel, langevin_sample


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_model_gradients(
    model: torch.nn.Module,
    enabled: bool,
) -> None:
    """
    Enable or disable gradients for all model parameters.

    During Langevin sampling, model parameters stay fixed and
    gradients are calculated only with respect to the images.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def main():
    device = get_device()
    print("Using device:", device)

    batch_size = 64
    epochs = 1
    learning_rate = 1e-4

    # Number of image-gradient updates used to create negative samples
    langevin_steps = 5
    langevin_step_size = 0.5
    langevin_noise_scale = 0.01

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = EnergyModel().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    print("Training samples:", len(train_dataset))
    print("Number of batches:", len(train_loader))
    print("Langevin steps per batch:", langevin_steps)

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_positive_energy = 0.0
        total_negative_energy = 0.0

        for batch_index, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)

            # Start negative samples from random noise
            negative_images = torch.empty_like(real_images).uniform_(
                -1.0,
                1.0,
            )

            # Freeze the Energy Model while modifying the images.
            # Only negative_images contribute gradients here.
            set_model_gradients(model, False)

            negative_images = langevin_sample(
                model=model,
                images=negative_images,
                steps=langevin_steps,
                step_size=langevin_step_size,
                noise_scale=langevin_noise_scale,
            )

            # Re-enable model parameter gradients for training.
            set_model_gradients(model, True)
            model.train()

            positive_energy = model(real_images)
            negative_energy = model(negative_images.detach())

            # Lower the energy of real CIFAR-10 images and
            # raise the energy of generated negative images.
            contrastive_loss = (
                positive_energy.mean()
                - negative_energy.mean()
            )

            # Energy regularization prevents energy values
            # from growing without limit.
            energy_regularization = 1e-3 * (
                positive_energy.square().mean()
                + negative_energy.square().mean()
            )

            loss = contrastive_loss + energy_regularization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_positive_energy += positive_energy.mean().item()
            total_negative_energy += negative_energy.mean().item()

            if batch_index % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_index}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Positive Energy: "
                    f"{positive_energy.mean().item():.4f} "
                    f"Negative Energy: "
                    f"{negative_energy.mean().item():.4f}"
                )

        number_of_batches = len(train_loader)

        average_loss = total_loss / number_of_batches
        average_positive_energy = (
            total_positive_energy / number_of_batches
        )
        average_negative_energy = (
            total_negative_energy / number_of_batches
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Average Loss: {average_loss:.4f}"
        )
        print(
            f"Average Positive Energy: "
            f"{average_positive_energy:.4f}"
        )
        print(
            f"Average Negative Energy: "
            f"{average_negative_energy:.4f}"
        )

        # Save after every epoch so progress is not lost.
        torch.save(
            model.state_dict(),
            "app/energy_weights.pth",
        )

        print(
            "Checkpoint saved to app/energy_weights.pth"
        )

    print("Energy Model training complete")


if __name__ == "__main__":
    main()
