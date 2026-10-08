import os
import sys
from pathlib import Path

# Suppress TensorFlow verbose CPU/oneDNN logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRADIO_SSR_MODE"] = "False"

import gradio as gr
from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Safely import spaces for Hugging Face ZeroGPU support
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator

from config import (
    EMOTION_EMOJIS,
    LABEL_NAMES,
    MAX_SEQUENCE_LENGTH,
    MODEL_PATH,
    TOKENIZER_PATH,
)
from predictor import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor(
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    labels=LABEL_NAMES,
    emojis=EMOTION_EMOJIS,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)

print("Starting EmotiSense Application...")
predictor.load()
print("Model and Tokenizer loaded successfully.")

# Load custom UI static assets
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

with open(STATIC_DIR / "style.css", "r", encoding="utf-8") as f:
    custom_css = f.read()

with open(STATIC_DIR / "script.js", "r", encoding="utf-8") as f:
    custom_js = f.read()

with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
    raw_html = f.read()

# Extract the inner body content from index.html
start_idx = raw_html.find("<body")
if start_idx != -1:
    start_idx = raw_html.find(">", start_idx) + 1
    end_idx = raw_html.rfind("</body>")
    body_content = raw_html[start_idx:end_idx].strip()
    # Remove local script tag since Gradio executes js parameter directly
    body_content = body_content.replace('<script src="/static/script.js"></script>', '')
else:
    body_content = raw_html

head_tags = """
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Playfair+Display:ital,wght@0,500;0,600;1,500;1,600&display=swap" rel="stylesheet" />
"""

# ZeroGPU-compatible prediction function
@spaces.GPU
def predict_emotion_core(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")
    return predictor.predict_single(text)


# Build Gradio Block wrapper
with gr.Blocks(
    head=head_tags,
    css=custom_css,
    js=custom_js,
    title="EmotiSense — Feel the Words",
    theme=gr.themes.Base(),
) as demo:
    # Inject EmotiSense Custom HTML structure
    gr.HTML(body_content)

    # Hidden Gradio components to satisfy ZeroGPU requirements
    with gr.Row(visible=False):
        hidden_input = gr.Textbox(visible=False)
        hidden_output = gr.JSON(visible=False)
        hidden_btn = gr.Button(visible=False)
        hidden_btn.click(
            fn=predict_emotion_core,
            inputs=[hidden_input],
            outputs=[hidden_output],
        )

# Mount static files and API endpoints to underlying FastAPI app
demo.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@demo.app.get("/health")
def health_check():
    return {"status": "Server is running", "model_loaded": predictor.is_loaded}

@demo.app.post("/predict")
@spaces.GPU
async def predict_api(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        if not text or not text.strip():
            return JSONResponse(status_code=400, content={"detail": "Input text cannot be empty."})
        result = predictor.predict_single(text)
        return {
            "text": text,
            "predicted_emotion": result["predicted_emotion"],
            "emoji": result["emoji"],
            "confidence": result["confidence"],
            "all_probabilites": result["probabilities"],
            "probabilities": result["probabilities"],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        ssr_mode=False,
    )
