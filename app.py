import os
import uvicorn
from main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting EmotiSense server on http://0.0.0.0:{port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
