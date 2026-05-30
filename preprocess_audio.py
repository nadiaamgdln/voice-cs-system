"""
preprocess_audio.py
-------------------
Preprocessing seluruh corpus audio sebelum digunakan di pipeline.

Yang dilakukan:
  1. Deteksi format asli tiap file (WAV valid, MP3, M4A, MP4, dll.)
  2. Remux/convert semua file → WAV PCM 16-bit, mono, 16000 Hz
     (format yang diharapkan whisper.cpp)
  3. Simpan hasil ke data/corpus/preprocessed/
  4. Buat manifest JSON di data/manifests/corpus_manifest.json

Cara pakai:
  python preprocess_audio.py
  python preprocess_audio.py --input data/corpus/audio --output data/corpus/preprocessed
"""

import os
import json
import shutil
import subprocess
import argparse
from pathlib import Path

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"
FFPROBE = r"C:\ffmpeg\bin\ffprobe.exe"

# ─── Konfigurasi default ────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
INPUT_DIR   = BASE_DIR / "data" / "corpus" / "audio"
OUTPUT_DIR  = BASE_DIR / "data" / "corpus" / "preprocessed"
MANIFEST    = BASE_DIR / "data" / "manifests" / "corpus_manifest.json"
LOG_FILE    = BASE_DIR / "log" / "preprocess.log"

# Target spesifikasi audio (standar Whisper)
TARGET_SR       = 16000   # sample rate Hz
TARGET_CHANNELS = 1       # mono
TARGET_BITDEPTH = "s16"   # signed 16-bit PCM (ffmpeg sample_fmt)

# ─── Helpers ────────────────────────────────────────────────────────────────
def detect_format(filepath: Path) -> str:
    """Deteksi format nyata file menggunakan `file` command."""
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=format_name", "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def ffprobe_info(filepath: Path) -> dict:
    """Ambil info audio stream via ffprobe."""
    cmd = [
        FFPROBE, "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(filepath)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        if audio_streams:
            s = audio_streams[0]
            return {
                "codec":    s.get("codec_name", "unknown"),
                "channels": s.get("channels", 0),
                "sample_rate": int(s.get("sample_rate", 0)),
                "bit_depth":   s.get("bits_per_sample", "unknown"),
            }
    except Exception:
        pass
    return {}

def needs_conversion(filepath: Path, probe: dict) -> bool:
    """Return True jika file perlu diconvert."""
    fmt = detect_format(filepath).lower()
    
    # File bukan WAV RIFF → pasti perlu convert
    if "riff" not in fmt or "wave" not in fmt:
        return True
    
    # WAV tapi salah spesifikasi
    if probe.get("channels", 0) != TARGET_CHANNELS:
        return True
    if probe.get("sample_rate", 0) != TARGET_SR:
        return True
    if probe.get("codec", "") not in ("pcm_s16le", "pcm_s16be", ""):
        return True
    
    return False

def convert_audio(src: Path, dst: Path, log_lines: list) -> bool:
    """
    Convert audio ke WAV PCM 16-bit mono 16kHz menggunakan ffmpeg.
    Return True jika berhasil.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        FFMPEG, "-y",
        "-i", str(src),
        "-ar", str(TARGET_SR),
        "-ac", str(TARGET_CHANNELS),
        "-sample_fmt", TARGET_BITDEPTH,
        "-acodec", "pcm_s16le",
        str(dst)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        log_lines.append(f"[OK]  {src.name} → {dst.name}")
        return True
    else:
        err = result.stderr.split("\n")[-3] if result.stderr else "unknown error"
        log_lines.append(f"[ERR] {src.name}: {err}")
        print(f"  [ERROR] {src.name}: {err}")
        return False

# ─── Main ────────────────────────────────────────────────────────────────────
def preprocess(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    # Kumpulkan semua file audio (semua ekstensi yang mungkin)
    extensions = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac", ".aac", ".opus"}
    all_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
        # juga handle file tanpa ekstensi yang tepat (e.g. *.m4a.wav)
    ])
    # Tambahkan semua file tanpa memandang ekstensi (untuk kasus .m4a.wav dll)
    all_wav = sorted(input_dir.glob("*.wav"))
    all_files = sorted(set(all_files) | set(all_wav))

    print(f"\n{'='*60}")
    print(f" Preprocessing Audio Corpus")
    print(f" Input  : {input_dir}")
    print(f" Output : {output_dir}")
    print(f" Total  : {len(all_files)} file ditemukan")
    print(f"{'='*60}\n")

    log_lines = []
    manifest  = []
    stats = {"ok": 0, "skip": 0, "error": 0, "converted": 0}

    for src in all_files:
        # Nama output selalu .wav bersih (hapus ekstensi ganda seperti .m4a.wav)
        clean_stem = src.stem
        # Hapus ekstensi tersembunyi (e.g. "2344_audio1.m4a" → "2344_audio1")
        while Path(clean_stem).suffix in {".m4a", ".mp3", ".mp4", ".aac", ".ogg"}:
            clean_stem = Path(clean_stem).stem
        dst = output_dir / f"{clean_stem}.wav"

        fmt    = detect_format(src)
        probe  = ffprobe_info(src)
        convert = needs_conversion(src, probe)

        print(f"  {'CONVERT' if convert else 'COPY   '} {src.name}")
        print(f"           format : {fmt[:60]}")
        if probe:
            print(f"           codec={probe.get('codec')}, "
                  f"sr={probe.get('sample_rate')}Hz, "
                  f"ch={probe.get('channels')}")

        if convert:
            ok = convert_audio(src, dst, log_lines)
            if ok:
                stats["converted"] += 1
                stats["ok"] += 1
            else:
                stats["error"] += 1
        else:
            # File sudah valid, copy saja
            shutil.copy2(src, dst)
            log_lines.append(f"[COPY] {src.name} → {dst.name}")
            stats["skip"] += 1
            stats["ok"] += 1

        if dst.exists():
            final_probe = ffprobe_info(dst)
            manifest.append({
                "id":          clean_stem,
                "original":    str(src.relative_to(BASE_DIR)),
                "preprocessed": str(dst.relative_to(BASE_DIR)),
                "original_format": fmt,
                "converted":   convert,
                "final_sr":    final_probe.get("sample_rate", TARGET_SR),
                "final_ch":    final_probe.get("channels", TARGET_CHANNELS),
                "final_codec": final_probe.get("codec", "pcm_s16le"),
            })
        print()

    # ── Tulis log ──
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # ── Tulis manifest ──
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f" Selesai!")
    print(f"   Total file  : {len(all_files)}")
    print(f"   Berhasil    : {stats['ok']}")
    print(f"   Di-convert  : {stats['converted']}")
    print(f"   Di-copy     : {stats['skip']}")
    print(f"   Error       : {stats['error']}")
    print(f"   Manifest    : {MANIFEST}")
    print(f"   Log         : {LOG_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess semua audio corpus ke WAV 16kHz mono PCM-16"
    )
    parser.add_argument(
        "--input",  type=Path, default=INPUT_DIR,
        help="Folder input audio mentah"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help="Folder output audio hasil preprocessing"
    )
    args = parser.parse_args()
    preprocess(args.input, args.output)
