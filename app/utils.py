"""
utils.py — Utilitas umum untuk Voice CS System
"""

import os
import re
import json
import time
import unicodedata
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = BASE_DIR / "data" / "corpus" / "transcripts"


# ─── Normalisasi Teks ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Normalisasi teks transkripsi sebelum dikirim ke LLM:
    - Hapus timestamp Whisper [00:00.000 → ...]
    - Normalisasi unicode NFC
    - Hapus karakter kontrol
    - Collapse whitespace berlebih
    """
    # Hapus timestamp format Whisper
    text = re.sub(r'\[\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}[.,]\d{3}\]', '', text)
    # Hapus tag lain dalam kurung siku
    text = re.sub(r'\[.*?\]', '', text)
    # Normalisasi unicode ke NFC
    text = unicodedata.normalize("NFC", text)
    # Hapus karakter kontrol (selain spasi dan newline)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse spasi berlebih
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def detect_language_tags(text: str) -> list[dict]:
    """
    Language tagging sederhana berbasis Unicode script.
    Return: list of {"lang": "ar"|"id", "text": "..."}

    - Arab  → Unicode U+0600–U+06FF
    - Latin → ID atau EN (tidak dibedakan lebih lanjut)
    """
    segments = []
    arabic_re = re.compile(
        r'([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
        r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]*)'
    )
    parts = arabic_re.split(text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.search(r'[\u0600-\u06FF]', part):
            segments.append({"lang": "ar", "text": part})
        else:
            segments.append({"lang": "id", "text": part})
    return segments if segments else [{"lang": "id", "text": text}]


# ─── Simpan Transkripsi ──────────────────────────────────────────────────────

def save_transcript(
    filename: str,
    transcript_raw: str,
    transcript_norm: str,
    lang_tags: list[dict],
    mode: str = "preserve",
):
    """
    Simpan hasil transkripsi STT ke data/corpus/transcripts/ dalam format JSON.
    Satu file JSON per audio input.
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    out_path = TRANSCRIPTS_DIR / f"{stem}.json"

    data = {
        "source_file":     filename,
        "timestamp":       datetime.now().isoformat(),
        "mode":            mode,
        "transcript_raw":  transcript_raw,
        "transcript_norm": transcript_norm,
        "language_tags":   lang_tags,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── WER / CER ───────────────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Hitung Word Error Rate (WER).
    WER = (S + D + I) / N
    S=substitutions, D=deletions, I=insertions, N=jumlah kata referensi.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    return _edit_distance_rate(ref_words, hyp_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Hitung Character Error Rate (CER).
    Sama seperti WER tapi pada level karakter.
    """
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))
    return _edit_distance_rate(ref_chars, hyp_chars)


def _edit_distance_rate(ref: list, hyp: list) -> float:
    """Dynamic programming edit distance, dinormalisasi terhadap panjang ref."""
    if not ref:
        return float(len(hyp))
    n, m = len(ref), len(hyp)
    # Tabel DP ukuran (n+1) x (m+1)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )
    return dp[n][m] / n


# ─── File Utilities ──────────────────────────────────────────────────────────

def clean_temp_file(path: str):
    """Hapus file temporer dengan aman."""
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def load_manifest(manifest_path: str = None) -> list[dict]:
    if manifest_path is None:
        manifest_path = BASE_DIR / "data" / "manifests" / "corpus_manifest.json"
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_log(log_data: list[dict], log_path: str = None):
    if log_path is None:
        log_path = BASE_DIR / "log" / "pipeline_log.json"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)


# ─── Rate Limiter ────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Simple rate limiter untuk Gemini API.
    Jaga agar RPM (requests per minute) tidak terlampaui.
    """
    def __init__(self, rpm: int = 10):
        self.rpm = rpm
        self.min_interval = 60.0 / rpm
        self._last_call = 0.0

    def wait(self):
        elapsed  = time.time() - self._last_call
        to_wait  = self.min_interval - elapsed
        if to_wait > 0:
            print(f"[RateLimit] Menunggu {to_wait:.1f}s agar tidak exceed RPM...")
            time.sleep(to_wait)
        self._last_call = time.time()
