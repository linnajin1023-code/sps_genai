from pathlib import Path

import torch
from torchvision.utils import save_image

from app.diffusion_model import SimpleUNet


NUM_TIMESTEPS = 1000


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def extract(values: torch.Tensor, timesteps: torch.Tensor, image_shape):
    """
    Select the diffusion value for every image in the batch
    and reshape it for broadcasting.
    """
    batch_size = timesteps.shape[0]
    selected = values.gather(0, timesteps)

    return selected.reshape(
        batch_size,
        *((1,) * (len(image_shape) - 1)),
    )


@torch.inference_mode()
def sample_images(
    model: SimpleUNet,
    number_of_images: int,
    device: torch.device,
) -> torch.Tensor:
    # Same noise schedule used during training
    betas = torch.linspace(
        1e-4,
        0.02,
        NUM_TIMESTEPS,
        device=device,
    )

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    alphas_cumprod_previous = torch.cat([
        torch.ones(1, device=device),
        alphas_cumprod[:-1],
    ])

    sqrt_reciprocal_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(
        1.0 - alphas_cumprod
    )

    posterior_variance = (
        betas
        * (1.0 - alphas_cumprod_previous)
        / (1.0 - alphas_cumprod)
    )

    # Start with random Gaussian noise
    images = torch.randn(
        number_of_images,
        3,
        32,
        32,
        device=device,
    )

    for step in reversed(range(NUM_TIMESTEPS)):
        timesteps = torch.full(
            (number_of_images,),
            step,
            device=device,
            dtype=torch.long,
        )

        beta_t = extract(
            betas,
            timesteps,
            images.shape,
        )

        sqrt_reciprocal_alpha_t = extract(
            sqrt_reciprocal_alphas,
            timesteps,
            images.shape,
        )

        sqrt_one_minus_alpha_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod,
            timesteps,
            images.shape,
        )

        predicted_noise = model(
            images,
            timesteps,
        )

        # Estimated mean of the previous image x_(t-1)
        model_mean = sqrt_reciprocal_alpha_t * (
            images
            - beta_t
            * predicted_noise
            / sqrt_one_minus_alpha_cumprod_t
        )

        if step > 0:
            posterior_variance_t = extract(
                posterior_variance,
                timesteps,
                images.shape,
            )

            random_noise = torch.randn_like(images)

            images = (
                model_mean
                + torch.sqrt(posterior_variance_t)
                * random_noise
            )
        else:
            images = model_mean

        if step % 100 == 0:
            print(f"Sampling step: {step}")

    # Convert the training range [-1, 1] to image range [0, 1]
    images = images.clamp(-1, 1)
    images = (images + 1.0) / 2.0

    return images


def main():
    device = get_device()
    print("Using device:", device)

    model = SimpleUNet().to(device)

    try:
        state_dict = torch.load(
            "app/diffusion_weights.pth",
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            "app/diffusion_weights.pth",
            map_location=device,
        )

    model.load_state_dict(state_dict)
    model.eval()

    output_directory = Path("generated_samples")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Generating images...")

    images = sample_images(
        model=model,
        number_of_images=4,
        device=device,
    )

    output_path = output_directory / "diffusion_samples.png"

    save_image(
        images.cpu(),
        output_path,
        nrow=2,
    )

    print("Generation complete")
    print("Image saved to:", output_path)


if __name__ == "__main__":
    main()
