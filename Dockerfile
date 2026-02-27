FROM python:3.12-slim

WORKDIR /app

# Install system deps for soundfile/sounddevice
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 libportaudio2 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model files if not present
RUN if [ ! -f kokoro-v1.0.onnx ]; then \
      curl -L -o kokoro-v1.0.onnx https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx; \
    fi && \
    if [ ! -f voices-v1.0.bin ]; then \
      curl -L -o voices-v1.0.bin https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin; \
    fi

EXPOSE 8080

CMD ["python", "-m", "kokoro_tts.web", "--host", "0.0.0.0", "--port", "8080"]
