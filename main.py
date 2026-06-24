from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from pathlib import Path

from app.bigram_model import BigramModel
from app.embedding_model import get_word_embedding
from app.cnn_model import CNNClassifier


app = FastAPI()

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


# CIFAR10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load the trained CNN model
device = torch.device("cpu")

cnn_model = CNNClassifier(num_classes=10)

model_path = Path(__file__).parent / "app" / "model_weights.pth"
cnn_model.load_state_dict(torch.load(model_path, map_location=device))

cnn_model.eval()

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    word = request.word.strip()

    if not word:
        raise HTTPException(status_code=400, detail="Word cannot be empty")

    embedding = get_word_embedding(word)

    if embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")

    return {
        "word": word,
        "embedding": embedding,
        "dimension": len(embedding)
    }


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    image_tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "predicted_class": class_names[predicted.item()],
        "confidence": float(confidence.item())
    }
