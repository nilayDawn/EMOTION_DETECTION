import logging
import pickle
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (
    EMOTION_EMOJIS,
    LABEL_NAMES,
    MAX_SEQUENCE_LENGTH,
    MODEL_DOWNLOAD_URL,
    MODEL_PATH,
    TOKENIZER_DOWNLOAD_URL,
    TOKENIZER_PATH,
)

logger = logging.getLogger(__name__)


def download_artifact(local_path: Path, url: str) -> None:
    """Downloads an artifact from a remote URL if it does not exist locally."""
    if local_path.is_file():
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Artifact {local_path.name} not found locally. Downloading from {url}...")

    # Download to a temporary file first to avoid partial file corruptions
    temp_path = local_path.with_suffix(local_path.suffix + ".downloading")
    try:
        # Create request with a browser-like User-Agent for reliable Hugging Face downloads
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EmotionDetectionApp)"})
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB blocks

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    if downloaded % (50 * 1024 * 1024) < block_size or downloaded == total_size:
                        logger.info(f"Downloading {local_path.name}: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")

        temp_path.replace(local_path)
        logger.info(f"Successfully downloaded and saved {local_path.name}.")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Failed to download {local_path.name} from {url}: {e}")
        raise RuntimeError(f"Could not download {local_path.name} from {url}: {e}") from e


def preprocess_text(text: str) -> str:
    """
    Cleans raw text to standardize format:
    1. Lowercase text
    2. Remove apostrophes (e.g., can't -> cant)
    3. Remove special characters and punctuation
    4. Remove extra whitespaces
    """
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def patch_keras_deserialization() -> None:
    """
    Patches Keras layer classes to gracefully handle configs containing
    newer parameters (such as `quantization_config`) across different Keras versions.
    """
    modules = []
    try:
        import keras
        modules.append(keras.layers)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        if hasattr(tf, "keras") and hasattr(tf.keras, "layers"):
            modules.append(tf.keras.layers)
    except ImportError:
        pass

    for mod in modules:
        for cls_name in ["Embedding", "Dense", "GRU", "Bidirectional", "SpatialDropout1D", "Dropout"]:
            layer_cls = getattr(mod, cls_name, None)
            if layer_cls is not None and hasattr(layer_cls, "__init__"):
                if not getattr(layer_cls, "_is_patched_for_compat", False):
                    orig_init = layer_cls.__init__
                    def _make_init(fn):
                        def _patched_init(self, *args, **kwargs):
                            kwargs.pop("quantization_config", None)
                            return fn(self, *args, **kwargs)
                        return _patched_init
                    layer_cls.__init__ = _make_init(orig_init)
                    layer_cls._is_patched_for_compat = True


class EmotionPredictor:
    """
    Service responsible for loading the trained BiGRU model and tokenizer,
    preprocessing input text, and producing emotion predictions with emojis.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        tokenizer_path: Path = TOKENIZER_PATH,
        labels: List[str] = LABEL_NAMES,
        emojis: Dict[str, str] = EMOTION_EMOJIS,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.labels = labels
        self.emojis = emojis
        self.max_len = max_sequence_length

        self.model: Optional[tf.keras.Model] = None
        self.tokenizer = None

    def load(self) -> None:
        """Loads the tokenizer and Keras model, downloading from Hugging Face if needed."""
        # Ensure artifacts are present locally before loading
        download_artifact(self.tokenizer_path, TOKENIZER_DOWNLOAD_URL)
        download_artifact(self.model_path, MODEL_DOWNLOAD_URL)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        if not self.tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found at: {self.tokenizer_path}")

        logger.info(f"Loading tokenizer from {self.tokenizer_path}...")
        with open(self.tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        logger.info(f"Loading Keras model from {self.model_path}...")
        # Apply compatibility patches and load without compilation for inference
        patch_keras_deserialization()
        try:
            import keras
            self.model = keras.models.load_model(str(self.model_path), compile=False)
        except Exception:
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)

        logger.info("Model and tokenizer loaded successfully.")

    @property
    def is_ready(self) -> bool:
        """Returns True if both model and tokenizer are loaded into memory."""
        return self.model is not None and self.tokenizer is not None

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Runs batch inference on a list of input texts.

        Returns:
            List of prediction dictionaries containing:
            - text: Original text
            - predicted_emotion: Top predicted label
            - emoji: Emoji corresponding to predicted emotion
            - confidence: Confidence probability of top class
            - probabilities: Dictionary of label -> float
            - breakdown: List of {emotion, emoji, probability} sorted descending
        """
        if not self.is_ready:
            raise RuntimeError("Model and tokenizer are not loaded. Call load() before predicting.")

        cleaned_texts = [preprocess_text(t) for t in texts]
        if not cleaned_texts or any(len(t) == 0 for t in cleaned_texts):
            raise ValueError("Input texts cannot be empty or whitespace only.")

        # Preprocess texts into padded integer sequences
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding="post",
            truncating="post",
        )

        # Run model inference
        predictions = self.model.predict(padded_sequences, verbose=0)

        results = []
        for orig_text, pred in zip(texts, predictions):
            probabilities = {
                label: round(float(pred[idx]), 4)
                for idx, label in enumerate(self.labels)
            }
            top_idx = int(np.argmax(pred))
            predicted_emotion = self.labels[top_idx]
            predicted_emoji = self.emojis.get(predicted_emotion, "❓")
            confidence = round(float(pred[top_idx]), 4)

            # Structured breakdown sorted by highest probability first
            breakdown = [
                {
                    "emotion": label,
                    "emoji": self.emojis.get(label, "❓"),
                    "probability": round(float(pred[idx]), 4),
                }
                for idx, label in enumerate(self.labels)
            ]
            breakdown.sort(key=lambda x: x["probability"], reverse=True)

            results.append({
                "text": orig_text,
                "predicted_emotion": predicted_emotion,
                "emoji": predicted_emoji,
                "confidence": confidence,
                "probabilities": probabilities,
                "breakdown": breakdown,
            })

        return results

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Runs inference on a single text string.
        Reuses batch inference for identical preprocessing and consistency.
        """
        return self.predict_batch([text])[0]


# Default singleton instance for the application
predictor = EmotionPredictor()
