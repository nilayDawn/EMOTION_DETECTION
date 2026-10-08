import os
import threading
import gc
from pathlib import Path
from contextlib import asynccontextmanager
import pickle
import re
import numpy as np

# Suppress TensorFlow verbose CPU/oneDNN logs and force CPU mode
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import (
    MODEL_DOWNLOAD_URL,
    MODEL_PATH,
    TOKENIZER_DOWNLOAD_URL,
    TOKENIZER_PATH,
)
from predictor import download_artifact, patch_keras_deserialization

# Limit TensorFlow threading to save memory on container runtimes
try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
STATIC_DIR = BASE_DIR / "static"

model_path = ARTIFACTS_DIR / "BiGRU_Model.keras"
tokenizer_path = ARTIFACTS_DIR / "tokenizer.pkl"
max_sequence_length = 66

emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"}
    )

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    all_probabilites: dict[str, float]
    probabilities: dict[str, float]

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool


dl_model = {}

def load_model_background():
    try:
        print('Ensuring model artifacts are ready...')
        download_artifact(TOKENIZER_PATH, TOKENIZER_DOWNLOAD_URL)
        download_artifact(MODEL_PATH, MODEL_DOWNLOAD_URL)

        print('Loading model and tokenizer into memory...')
        patch_keras_deserialization()
        dl_model["BiGRU"] = load_model(str(MODEL_PATH), compile=False)
        with open(TOKENIZER_PATH, 'rb') as file:
            dl_model["Tokenizer"] = pickle.load(file)
        gc.collect()
        print('Model and tokenizer loaded successfully into memory!')
    except Exception as e:
        print(f'Error loading model: {e}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Non-blocking background loader so Uvicorn opens port 8000 instantly (<0.1s)
    loader_thread = threading.Thread(target=load_model_background, daemon=True)
    loader_thread.start()

    yield

    dl_model.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse(STATIC_DIR / 'index.html')

@app.get('/style.css', include_in_schema=False)
def serve_css():
    return FileResponse(STATIC_DIR / 'style.css')

@app.get('/script.js', include_in_schema=False)
def serve_js():
    return FileResponse(STATIC_DIR / 'script.js')

@app.get('/health', response_model=HealthResponse)
@app.get('/health/', response_model=HealthResponse, include_in_schema=False)
def health_check():
    is_loaded = bool(dl_model.get("BiGRU") and dl_model.get("Tokenizer"))
    status_msg = "Model Online" if is_loaded else "Model Loading..."
    return HealthResponse(status=status_msg, model_loaded=is_loaded)

@app.post('/predict', response_model=PredictionResponse)
@app.post('/predict/', response_model=PredictionResponse, include_in_schema=False)
def predict_emotion(text_input: TextInput):
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing in memory. Please try again in 5 seconds.")

    cleaned_text = preprocess_text(text_input.text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty or only special characters/whitespace.")

    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    probabilities = BiGRU_model.predict(padded_sequence, verbose=0)[0]

    top_emotion_index = int(np.argmax(probabilities))
    predicted_label = emotion_labels[top_emotion_index]
    predicted_emoji = EMOTION_EMOJIS.get(predicted_label, "❓")
    all_probs = {
        label: round(float(prob), 4) for prob, label in zip(probabilities, emotion_labels)
    }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=predicted_label,
        emoji=predicted_emoji,
        confidence=round(float(probabilities[top_emotion_index]), 4),
        all_probabilites=all_probs,
        probabilities=all_probs
    )