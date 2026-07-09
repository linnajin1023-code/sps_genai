import torch

from app.gan_model import Generator, Discriminator


# Create the models
generator = Generator()
discriminator = Discriminator()


# Create a batch of 4 random noise vectors
noise = torch.randn(4, 100)

print("Noise shape:")
print(noise.shape)


# Test the Generator
fake_images = generator(noise)

print("\nGenerator output shape:")
print(fake_images.shape)


# Test the Discriminator
predictions = discriminator(fake_images)

print("\nDiscriminator output shape:")
print(predictions.shape)


# Print discriminator predictions
print("\nDiscriminator predictions:")
print(predictions)
