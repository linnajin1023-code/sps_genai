from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embedding_model import get_word_embedding

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