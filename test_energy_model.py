from pathlib import Path

import torch
from torchvision.utils import save_image

from app.energy_model import EnergyModel, langevin_sample


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():
    device = get_device()
    print("Using device:", device)

    model = EnergyModel().to(device)

    try:
        state_dict = torch.load(
            "app/energy_weights.pth",
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            "app/energy_weights.pth",
            map_location=device,
        )

    model.load_state_dict(state_dict)
    model.eval()

    # We do not update model parameters during sampling.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    number_of_images = 4

    initial_images = torch.empty(
        number_of_images,
        3,
        32,
        32,
        device=device,
    ).uniform_(-1.0, 1.0)

    with torch.no_grad():
        initial_energy = model(initial_images).mean().item()

    print(f"Initial average energy: {initial_energy:.4f}")
    print("Running Langevin sampling...")

    generated_images = langevin_sample(
        model=model,
        images=initial_images,
        steps=100,
        step_size=0.1,
        noise_scale=0.005,
    )

    with torch.no_grad():
        final_energy = model(generated_images).mean().item()

    print(f"Final average energy: {final_energy:.4f}")
    print(
        "Energy decreased:",
        final_energy < initial_energy,
    )

    # Convert [-1, 1] to [0, 1] for saving.
    images_to_save = (
        generated_images.detach().cpu().clamp(-1, 1) + 1
    ) / 2

    output_directory = Path("generated_samples")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory / "energy_model_samples.png"
    )

    save_image(
        images_to_save,
        output_path,
        nrow=2,
    )

    print("Image saved to:", output_path)


if __name__ == "__main__":
    main()
