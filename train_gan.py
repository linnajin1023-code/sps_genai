import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan_model import Generator, Discriminator


# --------------------------------------------------
# Training settings
# --------------------------------------------------

BATCH_SIZE = 64
LATENT_DIM = 100
EPOCHS = 20
LEARNING_RATE = 0.0002


# --------------------------------------------------
# Select device
# --------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# --------------------------------------------------
# Create folders
# --------------------------------------------------

os.makedirs("generated_samples", exist_ok=True)
os.makedirs("app", exist_ok=True)


# --------------------------------------------------
# Prepare MNIST dataset
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


print(f"Number of training images: {len(train_dataset)}")


# --------------------------------------------------
# Create Generator and Discriminator
# --------------------------------------------------

generator = Generator().to(device)
discriminator = Discriminator().to(device)


# --------------------------------------------------
# Loss function
# --------------------------------------------------

criterion = nn.BCELoss()


# --------------------------------------------------
# Optimizers
# --------------------------------------------------

generator_optimizer = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)


discriminator_optimizer = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)


# --------------------------------------------------
# Fixed noise for monitoring training progress
# --------------------------------------------------

fixed_noise = torch.randn(
    64,
    LATENT_DIM,
    device=device
)


# --------------------------------------------------
# Training loop
# --------------------------------------------------

print("\nStarting GAN training...\n")


for epoch in range(EPOCHS):

    for batch_index, (real_images, _) in enumerate(train_loader):

        real_images = real_images.to(device)

        current_batch_size = real_images.size(0)


        # Create labels
        real_labels = torch.ones(
            current_batch_size,
            1,
            device=device
        )

        fake_labels = torch.zeros(
            current_batch_size,
            1,
            device=device
        )


        # ==================================================
        # Train the Discriminator
        # ==================================================

        discriminator_optimizer.zero_grad()


        # Train with real images
        real_predictions = discriminator(real_images)

        real_loss = criterion(
            real_predictions,
            real_labels
        )


        # Generate fake images
        noise = torch.randn(
            current_batch_size,
            LATENT_DIM,
            device=device
        )

        fake_images = generator(noise)


        # Train with fake images
        fake_predictions = discriminator(
            fake_images.detach()
        )

        fake_loss = criterion(
            fake_predictions,
            fake_labels
        )


        # Total Discriminator loss
        discriminator_loss = real_loss + fake_loss

        discriminator_loss.backward()

        discriminator_optimizer.step()


        # ==================================================
        # Train the Generator
        # ==================================================

        generator_optimizer.zero_grad()


        # The Generator wants the Discriminator
        # to classify fake images as real
        generator_predictions = discriminator(fake_images)

        generator_loss = criterion(
            generator_predictions,
            real_labels
        )


        generator_loss.backward()

        generator_optimizer.step()


        # ==================================================
        # Print training progress
        # ==================================================

        if batch_index % 100 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_index}/{len(train_loader)}] "
                f"D Loss: {discriminator_loss.item():.4f} "
                f"G Loss: {generator_loss.item():.4f}"
            )


    # --------------------------------------------------
    # Save generated images after each epoch
    # --------------------------------------------------

    generator.eval()

    with torch.no_grad():

        sample_images = generator(fixed_noise)

        save_image(
            sample_images,
            f"generated_samples/epoch_{epoch + 1:02d}.png",
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )

    generator.train()


    print(
        f"Saved sample image: "
        f"generated_samples/epoch_{epoch + 1:02d}.png\n"
    )


# --------------------------------------------------
# Save trained model weights
# --------------------------------------------------

torch.save(
    generator.state_dict(),
    "app/generator_weights.pth"
)


torch.save(
    discriminator.state_dict(),
    "app/discriminator_weights.pth"
)


print("Training complete!")

print(
    "Generator weights saved to: "
    "app/generator_weights.pth"
)

print(
    "Discriminator weights saved to: "
    "app/discriminator_weights.pth"
)
