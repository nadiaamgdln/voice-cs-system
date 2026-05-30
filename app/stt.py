import os
import uuid
import tempfile
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Whisper.cpp harus di-clone di dalam app/whisper.cpp/
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp")

# Path ke binary whisper-cli (hasil build cmake)
WHISPER_BINARY = os.path.join(WHISPER_DIR, "build", "bin", "Release", "whisper-cli.exe")

# Path ke model Whisper large-v3-turbo (sesuaikan setelah download)
# Letakkan model di models/ → whisper.cpp/models/ggml-large-v3-turbo.bin
WHISPER_MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "whisper.cpp", "ggml-large-v3-turbo.bin"
)
# Normalisasi path
WHISPER_MODEL_PATH = os.path.normpath(WHISPER_MODEL_PATH)


def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    """
    Transkripsi file audio menggunakan whisper.cpp CLI.

    Args:
        file_bytes (bytes): Isi file audio (harus sudah WAV 16kHz mono PCM-16).
        file_ext   (str)  : Ekstensi file, default ".wav".

    Returns:
        str: Teks hasil transkripsi, atau pesan error.
    """
    # Periksa ketersediaan binary & model
    if not os.path.isfile(WHISPER_BINARY):
        return (
            f"[ERROR] whisper-cli binary tidak ditemukan di: {WHISPER_BINARY}\n"
            "Pastikan whisper.cpp sudah di-clone dan di-build. Lihat README.md."
        )
    if not os.path.isfile(WHISPER_MODEL_PATH):
        return (
            f"[ERROR] Model Whisper tidak ditemukan di: {WHISPER_MODEL_PATH}\n"
            "Download model dengan: ./models/download-ggml-model.sh large-v3-turbo"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path  = os.path.join(tmpdir, f"{uuid.uuid4()}{file_ext}")
        output_base = os.path.join(tmpdir, "transcription")
        result_path = output_base + ".txt"

        # Simpan audio ke file temporer
        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        cmd = [
            WHISPER_BINARY,
            "-m", WHISPER_MODEL_PATH,
            "-f", audio_path,
            "-l", "auto",
            "-t", "8",
            "-otxt",
            "-of", output_base,
            "--no-timestamps",
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600
            )
        except subprocess.CalledProcessError as e:
            return f"[ERROR] Whisper gagal (exit {e.returncode}): {e.stderr}"
        except subprocess.TimeoutExpired:
            return "[ERROR] Whisper timeout setelah 600 detik."

        try:
            with open(result_path, "r", encoding="utf-8") as result_file:
                transcript = result_file.read().strip()
            return transcript if transcript else "[WARN] Transkripsi kosong."
        except FileNotFoundError:
            return "[ERROR] File hasil transkripsi tidak ditemukan."


def transcribe_file_path(audio_path: str) -> str:
    """
    Versi alternatif: transkripsi langsung dari path file.
    Berguna untuk pipeline batch (analisis_pipeline.py).
    """
    if not os.path.isfile(audio_path):
        return f"[ERROR] File tidak ditemukan: {audio_path}"

    with open(audio_path, "rb") as f:
        file_bytes = f.read()

    ext = os.path.splitext(audio_path)[1] or ".wav"
    return transcribe_speech_to_text(file_bytes, ext)
