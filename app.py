import os
import sys

# Suppress TensorFlow verbose CPU/oneDNN logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import uvicorn
from main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting EmotiSense custom web application on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
