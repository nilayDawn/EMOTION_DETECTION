import os
from pathlib import Path
from typing import Dict, List

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Artifacts directory and file paths
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = Path(os.getenv("MODEL_PATH", ARTIFACTS_DIR / "BiGRU_Model.keras"))
TOKENIZER_PATH = Path(os.getenv("TOKENIZER_PATH", ARTIFACTS_DIR / "tokenizer.pkl"))

# Model hyperparameters
MAX_SEQUENCE_LENGTH: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "66"))

LABEL_NAMES: List[str] = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]

# Emotion to emoji mapping
EMOTION_EMOJIS: Dict[str, str] = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}
