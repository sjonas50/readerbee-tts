#!/usr/bin/env python3
"""Simple web UI for ReaderBee TTS."""

from __future__ import annotations

import io
import sys
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from kokoro_onnx import Kokoro

app = FastAPI(title="ReaderBee TTS")

# Global model instance, loaded at startup
kokoro: Kokoro | None = None

# Language prefix -> language code mapping
VOICE_LANG_MAP = {
    "af_": "en-us", "am_": "en-us",
    "bf_": "en-gb", "bm_": "en-gb",
    "ff_": "fr-fr",
    "if_": "it", "im_": "it",
    "jf_": "ja", "jm_": "ja",
    "zf_": "cmn", "zm_": "cmn",
}


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    lang: str = "en-us"


def get_voice_language(voice_name: str) -> str:
    """Determine language from voice name prefix."""
    for prefix, lang in VOICE_LANG_MAP.items():
        if voice_name.startswith(prefix):
            return lang
    return "en-us"


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/languages")
async def get_languages():
    if kokoro is None:
        raise HTTPException(503, "Model not loaded")
    languages = sorted(kokoro.get_languages())
    return {"languages": languages}


@app.get("/api/voices")
async def get_voices():
    if kokoro is None:
        raise HTTPException(503, "Model not loaded")
    voices = sorted(kokoro.get_voices())
    # Group by language
    grouped: dict[str, list[str]] = {}
    for v in voices:
        lang = get_voice_language(v)
        grouped.setdefault(lang, []).append(v)
    return {"voices": grouped}


@app.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest):
    if kokoro is None:
        raise HTTPException(503, "Model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Text is required")
    if len(text) > 10000:
        raise HTTPException(400, "Text too long (max 10,000 characters)")

    voice = req.voice
    speed = max(0.5, min(2.0, req.speed))

    try:
        # Handle voice blending (e.g. "af_sarah:60,am_adam:40")
        if "," in voice:
            parts = voice.split(",")
            if len(parts) != 2:
                raise ValueError("Voice blending requires exactly two voices")
            voices_list = []
            weights = []
            for part in parts:
                if ":" in part:
                    v, w = part.strip().split(":")
                    voices_list.append(v.strip())
                    weights.append(float(w.strip()))
                else:
                    voices_list.append(part.strip())
                    weights.append(50.0)
            total = sum(weights)
            weights = [w * (100 / total) for w in weights]
            style1 = kokoro.get_voice_style(voices_list[0])
            style2 = kokoro.get_voice_style(voices_list[1])
            voice = np.add(style1 * (weights[0] / 100), style2 * (weights[1] / 100))
        else:
            supported = set(kokoro.get_voices())
            if voice not in supported:
                raise ValueError(f"Unsupported voice: {voice}")

        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=req.lang)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Synthesis failed: {e}")

    # Write WAV to memory buffer
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=kokoro_tts_output.wav"},
    )


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="ReaderBee TTS Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--model", default="kokoro-v1.0.onnx", help="Path to ONNX model file")
    parser.add_argument("--voices", default="voices-v1.0.bin", help="Path to voices file")
    args = parser.parse_args()

    # Validate model files exist
    for path, label in [(args.model, "Model"), (args.voices, "Voices")]:
        if not os.path.exists(path):
            print(f"Error: {label} file not found: {path}")
            print(f"Download from: https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/")
            sys.exit(1)

    global kokoro
    print(f"Loading model from {args.model}...")
    kokoro = Kokoro(args.model, args.voices)
    print(f"Model loaded. Starting server at http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
