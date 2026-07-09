from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from pathlib import Path

from app.bigram_model import BigramModel
from app.embedding_model import get_word_embedding
from app.cnn_model import CNNClassifier
from app.gan_model import Generator


app = FastAPI()


# ==================================================
# Bigram Text Generation Model
# ==================================================

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


# ==================================================
# Device
# ==================================================

device = torch.device("cpu")


# ==================================================
# CNN Image Classification Model
# ==================================================

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


cnn_model = CNNClassifier(num_classes=10)

cnn_model_path = (
    Path(__file__).parent
    / "app"
    / "model_weights.pth"
)

cnn_model.load_state_dict(
    torch.load(
        cnn_model_path,
        map_location=device
    )
)

cnn_model.eval()


image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# ==================================================
# GAN Generator Model
# ==================================================

gan_generator = Generator()

gan_model_path = (
    Path(__file__).parent
    / "app"
    / "generator_weights.pth"
)

gan_generator.load_state_dict(
    torch.load(
        gan_model_path,
        map_location=device
    )
)

gan_generator.eval()


# ==================================================
# API Endpoints
# ==================================================

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):

    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )

    return {
        "generated_text": generated_text
    }


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):

    word = request.word.strip()

    if not word:
        raise HTTPException(
            status_code=400,
            detail="Word cannot be empty"
        )

    embedding = get_word_embedding(word)

    if embedding is None:
        raise HTTPException(
            status_code=404,
            detail="Embedding not found"
        )

    return {
        "word": word,
        "embedding": embedding,
        "dimension": len(embedding)
    }


@app.post("/classify-image")
async def classify_image(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    image_tensor = (
        image_transform(image)
        .unsqueeze(0)
    )

    with torch.no_grad():

        outputs = cnn_model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    return {
        "predicted_class": class_names[
            predicted.item()
        ],
        "confidence": float(
            confidence.item()
        )
    }


@app.get(
    "/generate-digit",
    response_class=Response
)
def generate_digit():

    # Create one random 100-dimensional noise vector
    noise = torch.randn(
        1,
        100,
        device=device
    )

    # Generate one handwritten digit image
    with torch.no_grad():

        generated_image = gan_generator(
            noise
        )

    # Remove the batch dimension
    generated_image = (
        generated_image
        .squeeze(0)
        .cpu()
    )

    # Convert pixel values from [-1, 1] to [0, 1]
    generated_image = (
        generated_image + 1
    ) / 2

    # Convert the tensor to a PIL image
    image = transforms.ToPILImage()(
        generated_image
    )

    # Save the image to memory as PNG
    image_buffer = io.BytesIO()

    image.save(
        image_buffer,
        format="PNG"
    )

    return Response(
        content=image_buffer.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition":
            'inline; filename="generated_digit.png"'
        }
    )