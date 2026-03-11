"""
FastAPI service for SAM Audio model inference.
Returns target and residual audio file paths from audio separation.

import requests

response = requests.post(
    "http://localhost:8014/separate",
    data={
        "audio": "workspace/utterance.wav",  # Must be a path the server can read
        "description": "main voice"
    }
)
result = response.json()
print("Target audio:", result["target"])
print("Residual audio:", result["residual"])

"""
import gc
import tempfile
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, Form, HTTPException

from sam_audio import SAMAudio, SAMAudioProcessor


MODEL_NAME = "facebook/sam-audio-base"
model = None
processor = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMAudio.from_pretrained(MODEL_NAME)
    model = model.to(device=device, dtype=torch.float16).eval()
    processor = SAMAudioProcessor.from_pretrained(MODEL_NAME)
    gc.collect()
    torch.cuda.empty_cache()
    yield


app = FastAPI(title="SAM Audio Service", lifespan=lifespan)


@app.post("/separate")
async def separate(
    audio: str = Form(..., description="Path to input audio file"),
    description: str = Form(..., description="Text description of the sound to isolate, e.g. 'man speaking' or 'vocals'"),
):
    """Separate audio into target (isolated sound) and residual (everything else). Returns JSON with temp file paths."""
    try:
        waveform, sr = torchaudio.load(audio)

        target_sr = processor.audio_sampling_rate
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        inputs = processor(audios=[waveform], descriptions=[description]).to(device)

        with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
            result = model.separate(inputs, predict_spans=False, reranking_candidates=1)

        target = result.target[0].unsqueeze(0).cpu()
        residual = result.residual[0].unsqueeze(0).cpu()
        sample_rate = processor.audio_sampling_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            target_path = f.name
        torchaudio.save(target_path, target, sample_rate, format="wav")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            residual_path = f.name
        torchaudio.save(residual_path, residual, sample_rate, format="wav")

        return {"target": target_path, "residual": residual_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8014)