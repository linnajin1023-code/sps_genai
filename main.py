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
from app.diffusion_model import SimpleUNet
from app.energy_model import EnergyModel, langevin_sample


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

generation_device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Image generation device:", generation_device)


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
# Diffusion Image Generator
# ==================================================

diffusion_model = SimpleUNet().to(
    generation_device
)

diffusion_model_path = (
    Path(__file__).parent
    / "app"
    / "diffusion_weights.pth"
)

diffusion_model.load_state_dict(
    torch.load(
        diffusion_model_path,
        map_location=generation_device,
        weights_only=True
    )
)

diffusion_model.eval()


DIFFUSION_TIMESTEPS = 1000

diffusion_betas = torch.linspace(
    1e-4,
    0.02,
    DIFFUSION_TIMESTEPS,
    device=generation_device
)

diffusion_alphas = 1.0 - diffusion_betas

diffusion_alphas_cumprod = torch.cumprod(
    diffusion_alphas,
    dim=0
)

diffusion_alphas_cumprod_previous = torch.cat([
    torch.ones(
        1,
        device=generation_device
    ),
    diffusion_alphas_cumprod[:-1]
])

diffusion_sqrt_reciprocal_alphas = torch.sqrt(
    1.0 / diffusion_alphas
)

diffusion_sqrt_one_minus_alphas_cumprod = torch.sqrt(
    1.0 - diffusion_alphas_cumprod
)

diffusion_posterior_variance = (
    diffusion_betas
    * (
        1.0
        - diffusion_alphas_cumprod_previous
    )
    / (
        1.0
        - diffusion_alphas_cumprod
    )
)


def extract_diffusion_value(
    values,
    timesteps,
    image_shape
):
    batch_size = timesteps.shape[0]

    selected_values = values.gather(
        0,
        timesteps
    )

    return selected_values.reshape(
        batch_size,
        *((1,) * (len(image_shape) - 1))
    )


@torch.inference_mode()
def sample_diffusion_image():

    image = torch.randn(
        1,
        3,
        32,
        32,
        device=generation_device
    )

    for step in reversed(
        range(DIFFUSION_TIMESTEPS)
    ):

        timestep = torch.full(
            (1,),
            step,
            device=generation_device,
            dtype=torch.long
        )

        beta_t = extract_diffusion_value(
            diffusion_betas,
            timestep,
            image.shape
        )

        sqrt_reciprocal_alpha_t = (
            extract_diffusion_value(
                diffusion_sqrt_reciprocal_alphas,
                timestep,
                image.shape
            )
        )

        sqrt_one_minus_cumprod_t = (
            extract_diffusion_value(
                diffusion_sqrt_one_minus_alphas_cumprod,
                timestep,
                image.shape
            )
        )

        predicted_noise = diffusion_model(
            image,
            timestep
        )

        model_mean = (
            sqrt_reciprocal_alpha_t
            * (
                image
                - beta_t
                * predicted_noise
                / sqrt_one_minus_cumprod_t
            )
        )

        if step > 0:

            posterior_variance_t = (
                extract_diffusion_value(
                    diffusion_posterior_variance,
                    timestep,
                    image.shape
                )
            )

            random_noise = torch.randn_like(
                image
            )

            image = (
                model_mean
                + torch.sqrt(
                    posterior_variance_t
                )
                * random_noise
            )

        else:
            image = model_mean

    image = image.clamp(-1.0, 1.0)

    return (image + 1.0) / 2.0


# ==================================================
# Energy-Based Image Generator
# ==================================================

energy_model = EnergyModel().to(
    generation_device
)

energy_model_path = (
    Path(__file__).parent
    / "app"
    / "energy_weights.pth"
)

energy_model.load_state_dict(
    torch.load(
        energy_model_path,
        map_location=generation_device,
        weights_only=True
    )
)

energy_model.eval()

# During Langevin sampling, gradients are calculated
# with respect to the image, not model parameters.
for parameter in energy_model.parameters():
    parameter.requires_grad_(False)


def sample_energy_image():

    initial_image = torch.empty(
        1,
        3,
        32,
        32,
        device=generation_device
    ).uniform_(-1.0, 1.0)

    generated_image = langevin_sample(
        model=energy_model,
        images=initial_image,
        steps=100,
        step_size=0.1,
        noise_scale=0.005
    )

    generated_image = (
        generated_image
        .detach()
        .clamp(-1.0, 1.0)
    )

    return (generated_image + 1.0) / 2.0


def tensor_to_png_response(
    image_tensor,
    filename
):

    image_tensor = (
        image_tensor
        .squeeze(0)
        .detach()
        .cpu()
    )

    image = transforms.ToPILImage()(
        image_tensor
    )

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
            f'inline; filename="{filename}"'
        }
    )


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


@app.get(
    "/generate-diffusion-image",
    response_class=Response
)
def generate_diffusion_image():

    generated_image = sample_diffusion_image()

    return tensor_to_png_response(
        generated_image,
        "diffusion_image.png"
    )


@app.get(
    "/generate-energy-image",
    response_class=Response
)
def generate_energy_image():

    generated_image = sample_energy_image()

    return tensor_to_png_response(
        generated_image,
        "energy_image.png"
    )

