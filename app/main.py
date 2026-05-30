"""
main.py — FastAPI backend untuk Voice CS System
Endpoint utama: POST /voice-chat
"""

import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.stt   import transcribe_speech_to_text
from app.llm   import generate_response
from app.tts   import transcribe_text_to_speech
from app.utils import normalize_text, detect_language_tags, save_transcript

app = FastAPI(
    title="Voice Code-Switching System",
    description="Pipeline STT → LLM → TTS untuk code-switching ID-EN-AR | Mode: preserve / normalized / translate",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = tempfile.gettempdir()
VALID_MODES = {"preserve", "normalized", "translate"}


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Voice CS System 🎙️",
        "modes": list(VALID_MODES),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/voice-chat")
async def voice_chat(
    file: UploadFile = File(...),
    mode: str = Query(default="preserve", description="Mode: preserve | normalized | translate"),
):
    """
    Pipeline lengkap: Audio → STT → Normalisasi → LLM → TTS → Audio response.

    Query param:
        mode: 'preserve' (default) | 'normalized' | 'translate'
    """
    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Mode tidak valid: '{mode}'. Pilih: {VALID_MODES}"
        )

    # 1. Baca audio
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File audio kosong.")

    filename = file.filename or "audio.wav"
    ext = os.path.splitext(filename)[1].lower() or ".wav"

    # 2. STT
    print(f"[STT] File: {filename} ({len(audio_bytes)} bytes)")
    transcript_raw = transcribe_speech_to_text(audio_bytes, ext)
    print(f"[STT] Raw: {transcript_raw}")

    if transcript_raw.startswith("[ERROR]"):
        raise HTTPException(status_code=500, detail=transcript_raw)

    # 3. Normalisasi + language tagging
    transcript_norm = normalize_text(transcript_raw)
    lang_tags = detect_language_tags(transcript_norm)
    print(f"[NLP] Normalized: {transcript_norm}")
    print(f"[NLP] Lang tags: {lang_tags}")

    # Simpan transkripsi ke data/corpus/transcripts/
    save_transcript(filename, transcript_raw, transcript_norm, lang_tags, mode)

    # 4. LLM
    print(f"[LLM] Mode={mode}, Input: {transcript_norm}")
    llm_response = generate_response(transcript_norm, mode=mode)
    print(f"[LLM] Response: {llm_response}")

    if llm_response.startswith("[ERROR]"):
        raise HTTPException(status_code=500, detail=llm_response)

    # 5. TTS
    print(f"[TTS] Synthesize: {llm_response}")
    audio_output_path = transcribe_text_to_speech(llm_response)

    if isinstance(audio_output_path, str) and audio_output_path.startswith("[ERROR]"):
        raise HTTPException(status_code=500, detail=audio_output_path)

    if not os.path.isfile(audio_output_path):
        raise HTTPException(status_code=500, detail="File audio TTS tidak ditemukan.")

    return FileResponse(
        path=audio_output_path,
        media_type="audio/wav",
        filename="response.wav",
    )


@app.post("/transcribe")
async def transcribe_only(file: UploadFile = File(...)):
    """Debug endpoint: STT saja, kembalikan transkripsi + language tags."""
    audio_bytes = await file.read()
    ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    transcript_raw  = transcribe_speech_to_text(audio_bytes, ext)
    transcript_norm = normalize_text(transcript_raw)
    lang_tags = detect_language_tags(transcript_norm)
    return JSONResponse({
        "transcript_raw":        transcript_raw,
        "transcript_normalized": transcript_norm,
        "language_tags":         lang_tags,
    })


@app.post("/generate")
async def generate_only(
    text: str,
    mode: str = Query(default="preserve"),
):
    """Debug endpoint: LLM saja (tanpa STT/TTS)."""
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Mode tidak valid: {mode}")
    response = generate_response(text, mode=mode)
    return JSONResponse({"mode": mode, "input": text, "response": response})
