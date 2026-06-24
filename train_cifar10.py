import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from app.cnn_model import CNNClassifier


def main():
    # Use CPU for training
    device = torch.device("cpu")

    # CIFAR10 original images are 32x32.
    # Our assignment CNN expects 64x64 input, so resize to 64x64.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    model = CNNClassifier(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Step [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {running_loss / 100:.4f}"
                )
                running_loss = 0.0

    torch.save(model.state_dict(), "app/model_weights.pth")
    print("Model saved to app/model_weights.pth")


if __name__ == "__main__":
    main()
