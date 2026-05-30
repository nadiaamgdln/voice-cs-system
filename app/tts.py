import os
import re
import uuid
import tempfile
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder aset model Coqui TTS
# Letakkan file model di app/coqui_tts/
COQUI_DIR = os.path.join(BASE_DIR, "coqui_tts")

# Path ke file model checkpoint (checkpoint_*.pth)
# Nama file disesuaikan dengan model yang di-download
COQUI_MODEL_PATH  = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")

# Path ke file konfigurasi model
COQUI_CONFIG_PATH = os.path.join(COQUI_DIR, "config.json")

# Nama speaker (sesuaikan dengan isi speakers.pth — biasanya "wibowo" untuk Indonesian-TTS)
COQUI_SPEAKER = "wibowo"

# ─── Utilitas ────────────────────────────────────────────────────────────────

def _detect_language_segments(text: str) -> list[tuple[str, str]]:
    """
    Pecah teks menjadi segmen berdasarkan script/bahasa.
    Return: list of (lang, segment_text)
      - 'ar'  → teks Arab (Unicode block Arabic U+0600–U+06FF)
      - 'id'  → teks Latin (Indonesia/Inggris)

    Strategi sederhana: split di batas antara Arab ↔ Latin.
    """
    segments = []
    # Pattern untuk mengelompokkan karakter Arab vs bukan Arab
    arabic_pattern = re.compile(r'([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+(?:\s+[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+)*)')
    parts = arabic_pattern.split(text)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.search(r'[\u0600-\u06FF]', part):
            segments.append(("ar", part))
        else:
            segments.append(("id", part))

    return segments if segments else [("id", text)]


def transcribe_text_to_speech(text: str) -> str:
    """
    Konversi teks ke audio menggunakan Coqui TTS.
    Mendukung teks code-switching dengan pemisahan per segmen bahasa.

    Args:
        text (str): Teks respons (bisa mengandung ID, EN, AR).

    Returns:
        str: Path ke file WAV output, atau pesan error.
    """
    # Periksa ketersediaan model
    if not os.path.isfile(COQUI_MODEL_PATH):
        return (
            f"[ERROR] Model TTS tidak ditemukan: {COQUI_MODEL_PATH}\n"
            "Download model Indonesian-TTS dari:\n"
            "https://github.com/wikimedia/wikipedia-tts-indonesian\n"
            "atau ikuti README.md untuk langkah instalasi."
        )

    segments = _detect_language_segments(text)

    # Jika satu segmen saja, langsung render
    if len(segments) == 1:
        return _tts_with_coqui(segments[0][1])

    # Multi-segmen: render tiap segmen, lalu gabungkan dengan ffmpeg
    tmp_files = []
    try:
        for lang, seg_text in segments:
            if not seg_text.strip():
                continue
            seg_path = _tts_with_coqui(seg_text)
            if seg_path.startswith("[ERROR]"):
                return seg_path
            tmp_files.append(seg_path)

        if not tmp_files:
            return "[ERROR] Tidak ada segmen teks yang berhasil di-render."

        if len(tmp_files) == 1:
            return tmp_files[0]

        # Gabungkan file-file audio dengan ffmpeg concat
        return _concat_audio_files(tmp_files)

    finally:
        # Bersihkan file temp perantara (kecuali output akhir)
        # File output akhir dibersihkan oleh caller (main.py)
        pass


# ─── Engine Coqui TTS ────────────────────────────────────────────────────────

def _tts_with_coqui(text: str) -> str:
    """Render teks menjadi WAV menggunakan Coqui TTS CLI."""
    tmp_dir     = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")

    cmd = [
        "tts",
        "--text",        text,
        "--model_path",  COQUI_MODEL_PATH,
        "--config_path", COQUI_CONFIG_PATH,
        # "--speaker_idx", COQUI_SPEAKER,
        "--out_path",    output_path,
    ]

    try:
        subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True, 
            encoding="utf-8",
            errors="replace",
            timeout=300)
        return output_path
    except subprocess.CalledProcessError as e:
        return f"[ERROR] Coqui TTS gagal: {e.stderr}"
    except subprocess.TimeoutExpired:
        return "[ERROR] TTS timeout setelah 180 detik."


# ─── Concat helper ───────────────────────────────────────────────────────────

def _concat_audio_files(file_paths: list[str]) -> str:
    """Gabungkan beberapa WAV file menjadi satu menggunakan ffmpeg concat."""
    tmp_dir     = tempfile.gettempdir()
    list_file   = os.path.join(tmp_dir, f"concat_{uuid.uuid4()}.txt")
    output_path = os.path.join(tmp_dir, f"tts_merged_{uuid.uuid4()}.wav")

    # Buat file daftar untuk ffmpeg concat demuxer
    with open(list_file, "w") as f:
        for fp in file_paths:
            f.write(f"file '{fp}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e:
        return f"[ERROR] Gagal menggabungkan audio: {e.stderr}"
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)

def synthesize_speech(text: str, output_path: str) -> str:
    """
    Wrapper untuk kompatibilitas pipeline analisis.
    Generate TTS lalu salin ke output_path.
    """
    import shutil

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    generated_path = transcribe_text_to_speech(text)

    if generated_path.startswith("[ERROR]"):
        return generated_path

    shutil.copy2(generated_path, output_path)

    return output_path