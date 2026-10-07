
from pathlib import Path
from contextlib import asynccontextmanager
import pickle
import re
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

"""
1. We are going to make some constants like:
A. Model Path (BiGRU)
B. Tokenizer Path
C. Max Sequence Length
D. Emotion Labels
E. Emotion emojis
"""
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
STATIC_DIR = BASE_DIR / "static"

# A. Model Path (BiGRU)
model_path = ARTIFACTS_DIR / "BiGRU_Model.keras"

# B. Tokenizer Path
tokenizer_path = ARTIFACTS_DIR / "tokenizer.pkl"

# C. Max Sequence Length (66 is the trained sentence length from notebook)
max_sequence_length = 66

# D. Emotion Labels
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# E. Emotion emojis
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


"""
2. Preprocess the upcoming text
Cleans raw text so it matches the format used while training.
A. Convert the text to lowercase. -done
B. Remove apostrophes (e.g can't -> cant). -done
C. Remove Special Characters and Punctuation. -done
D. Remove extra spaces -done
"""

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


"""
3. Request and Response Schemas
A. Text Input -> Input schema the text sent by user. -done
B. Prediction Response -> Output schema the emotion to predict. -done
C. Health Response (Server health check)
"""

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
    status: str
    model_loaded: bool


"""
4. Model Loading and LifeSpan Management
Load the model and tokenizer once the server starts up.
"""
dl_model = {}  # {1. BiGRU, 2. Tokenizer}-> True , {} -> False

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading the model and tokenizer...')
    dl_model["BiGRU"] = load_model(str(model_path))                   # BiGRU Model
    with open(tokenizer_path, 'rb') as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print('Model are loaded successfully...')

    yield  # Pause, model is loaded and server is running and at this point model wait karega for request

    dl_model.clear()  # Ek baar server band ho gaya uske baad model ko memory se hata do.


"""
5. Mount the static files to the FastAPI app
A. Enable CORS (Cross-Origin Resource Sharing) to allow requests from different origins.
"""
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name="static")


"""
6. API Endpoints.
A. Server UI at homepage ('/')
B. Static asset fallbacks ('/style.css', '/script.js')
C. Health Check Endpoint ('/health')
D. Predict Emotion Endpoint ('/predict')
"""

# A. Server UI at homepage ('/')
@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse(STATIC_DIR / 'index.html')

# B. Direct fallback routes for root static requests (prevents 404 if index.html uses relative paths)
@app.get('/style.css', include_in_schema=False)
def serve_css():
    return FileResponse(STATIC_DIR / 'style.css')

@app.get('/script.js', include_in_schema=False)
def serve_js():
    return FileResponse(STATIC_DIR / 'script.js')

# C. Health Check Endpoint ('/health' and '/health/')
@app.get('/health', response_model=HealthResponse)
@app.get('/health/', response_model=HealthResponse, include_in_schema=False)
def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))

# D. Predict Emotion Endpoint ('/predict' and '/predict/')
@app.post('/predict', response_model=PredictionResponse)
@app.post('/predict/', response_model=PredictionResponse, include_in_schema=False)
def predict_emotion(text_input: TextInput):
    """
    1. Cleans the input sentences.
    2. Convert the words into numeric using tokenizer.
    3. Pad the sequences to ensure uniform length.
    4. Run prediction using the BiGRU model.
    5. Return the top emotion and full probability breakdown.
    """

    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    # 1. Preprocess and validate
    cleaned_text = preprocess_text(text_input.text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty or only special characters/whitespace.")

    # 2. and 3. Tokenize and pad
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    # 4. Predict
    probabilities = BiGRU_model.predict(padded_sequence, verbose=0)[0]

    # 5. Format results
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