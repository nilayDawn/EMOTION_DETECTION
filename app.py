import os
import sys

# Suppress TensorFlow verbose CPU/oneDNN logs and disable experimental Gradio SSR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRADIO_SSR_MODE"] = "False"

import gradio as gr
from config import (
    EMOTION_EMOJIS,
    LABEL_NAMES,
    MAX_SEQUENCE_LENGTH,
    MODEL_DOWNLOAD_URL,
    MODEL_PATH,
    TOKENIZER_DOWNLOAD_URL,
    TOKENIZER_PATH,
)
from predictor import EmotionPredictor, download_artifact

# Initialize predictor singleton
predictor = EmotionPredictor(
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    labels=LABEL_NAMES,
    emojis=EMOTION_EMOJIS,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)

print("Starting Emotion Detection App...")
predictor.load()
print("Predictor loaded successfully.")


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


@spaces.GPU
def predict_emotion_gradio(text: str):
    """
    Callback for Gradio prediction.
    Takes input text and returns formatted top emotion string and probability dictionary for gr.Label.
    """
    if not text or not text.strip():
        return "⚠️ Please enter some text to analyze.", {}

    try:
        result = predictor.predict_single(text)
        emotion = result["predicted_emotion"]
        emoji = result["emoji"]
        confidence = result["confidence"] * 100
        probabilities = result["probabilities"]

        top_display = f"### Top Emotion: **{emotion.upper()}** {emoji}  \n**Confidence:** `{confidence:.2f}%`"
        return top_display, probabilities
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}", {}


# Custom CSS for modern styling
custom_css = """
.container { max-width: 850px; margin: auto; }
.header-box { text-align: center; margin-bottom: 20px; }
.primary-output { font-size: 1.25rem; padding: 12px; border-radius: 8px; }
"""

# Example test sentences
examples = [
    ["I just got promoted at work and I could not be happier today!"],
    ["I miss my old friends so much, everything feels lonely lately."],
    ["I am furious that they canceled the flight without any prior notice!"],
    ["I have an interview in ten minutes and my heart is beating so fast."],
    ["Spending the evening with my family makes my life feel so complete."],
    ["I opened the door and everyone jumped out shouting happy birthday!"],
]

# Build Gradio Blocks UI
with gr.Blocks(css=custom_css, title="🎭 Emotion Detection AI") as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown(
            """
            # 🎭 Emotion Detection with Deep Learning (BiGRU)
            Analyze the emotional tone of your text in real time using a **Bidirectional GRU (BiGRU)** neural network.
            """,
            elem_classes=["header-box"],
        )

        with gr.Row():
            with gr.Column(scale=5):
                text_input = gr.Textbox(
                    label="Enter your text / sentence:",
                    placeholder="Type or paste any sentence here (e.g., 'I am so excited for this weekend!')...",
                    lines=4,
                    max_lines=10,
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Analyze Emotion ✨", variant="primary")

            with gr.Column(scale=5):
                top_output = gr.Markdown(
                    value="*Enter text and click 'Analyze Emotion' to see results.*",
                    elem_classes=["primary-output"],
                )
                prob_output = gr.Label(
                    num_top_classes=6,
                    label="Emotion Probability Distribution",
                )

        gr.Examples(
            examples=examples,
            inputs=[text_input],
            outputs=[top_output, prob_output],
            fn=predict_emotion_gradio,
            cache_examples=False,
        )

        # Wire up click and submit events
        submit_btn.click(
            fn=predict_emotion_gradio,
            inputs=[text_input],
            outputs=[top_output, prob_output],
        )
        text_input.submit(
            fn=predict_emotion_gradio,
            inputs=[text_input],
            outputs=[top_output, prob_output],
        )
        clear_btn.click(
            fn=lambda: ("", "*Enter text and click 'Analyze Emotion' to see results.*", {}),
            inputs=[],
            outputs=[text_input, top_output, prob_output],
        )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )
