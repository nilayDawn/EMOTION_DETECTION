import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URL = "https://huggingface.co/Nilaydawn/emotion-bigru-model/resolve/main/BiGRU_Model.keras"
TOKENIZER_URL = "https://huggingface.co/Nilaydawn/emotion-bigru-model/resolve/main/tokenizer.pkl"

def download_file(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"Artifact already exists: {dest.name} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        return
    print(f"Downloading {dest.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EmotionDetectionApp)"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"Downloaded {dest.name} successfully ({dest.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    download_file(TOKENIZER_URL, ARTIFACTS_DIR / "tokenizer.pkl")
    download_file(MODEL_URL, ARTIFACTS_DIR / "BiGRU_Model.keras")
    print("All artifacts ready.")
