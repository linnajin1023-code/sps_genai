import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.diffusion_model import SimpleUNet


def get_index_from_list(values, t, x_shape):
    batch_size = t.shape[0]
    out = values.gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def forward_diffusion_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x0)

    sqrt_alphas_cumprod_t = get_index_from_list(
        sqrt_alphas_cumprod,
        t,
        x0.shape,
    )
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod,
        t,
        x0.shape,
    )

    xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    return xt, noise


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    batch_size = 64
    epochs = 5
    learning_rate = 1e-3
    num_timesteps = 1000

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
        num_workers=2,
    )

    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    print("Training samples:", len(train_dataset))
    print("Number of batches:", len(train_loader))

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)

            t = torch.randint(
                0,
                num_timesteps,
                (images.shape[0],),
                device=device,
            ).long()

            xt, noise = forward_diffusion_sample(
                images,
                t,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
            )

            predicted_noise = model(xt, t)
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "app/diffusion_weights.pth")
    print("Training complete. Weights saved to app/diffusion_weights.pth")


if __name__ == "__main__":
    main()
