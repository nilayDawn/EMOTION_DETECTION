# 🎭 EmotiSense — Deep Learning Emotion Perception Engine

A full-stack, containerized Natural Language Processing (NLP) web application that perceives and visualizes the emotional tone of text across 6 primary emotions in real time:
- 😢 **Sadness**
- 😄 **Joy**
- ❤️ **Love**
- 😠 **Anger**
- 😨 **Fear**
- 😲 **Surprise**

---

### 🌟 Features
- **Modern Bespoke UI:** Custom living bokeh canvas, glowing orb perception, glassmorphism cards, and animated probability distributions.
- **Deep Learning Model:** Bidirectional GRU (BiGRU) trained on emotion-labeled text.
- **Production Backend:** FastAPI with asynchronous request handling and resilient asset management.
- **Fully Containerized:** Multi-stage, non-root Docker build ready for Azure, AWS, GCP, or Docker Hub.

---

### 🚀 Quick Start (Docker)

```bash
# 1. Build the Docker image
docker build -t emotisense-app .

# 2. Run container locally
docker run -p 8000:8000 emotisense-app
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

### ☁️ Azure Deployment

1. **Tag and push image to Docker Hub / ACR:**
   ```bash
   docker tag emotisense-app <your-username>/emotisense-app:latest
   docker push <your-username>/emotisense-app:latest
   ```
2. **Deploy via Azure App Service (Web App for Containers):**
   - Publish: **Container**
   - Operating System: **Linux**
   - Image: `<your-username>/emotisense-app:latest`
   - Port: `8000`
