"""
analisis_pipeline.py
────────────────────
Pipeline analisis & uji coba SELURUH corpus audio yang telah direkam.

Flow per file:
  preprocessed_audio
    → STT (whisper large-v3-turbo)
    → normalisasi + language tagging
    → LLM (Gemini) — mode: preserve / normalized / translate
    → hitung WER/CER jika referensi tersedia
    → simpan hasil ke log/

Output:
  log/pipeline_results.json  — hasil lengkap per file
  log/pipeline_summary.csv   — ringkasan tabular
  log/pipeline_report.txt    — laporan human-readable

Cara pakai:
  # Jalankan preprocessing dulu!
  python preprocess_audio.py

  # Jalankan pipeline mode preserve (default)
  python analisis_pipeline.py

  # Mode normalized:
  python analisis_pipeline.py --mode normalized

  # Test 10 file pertama saja:
  python analisis_pipeline.py --limit 10

  # Dengan referensi transkripsi untuk hitung WER/CER:
  python analisis_pipeline.py --ref-file data/corpus/transcripts/reference.json
"""

import os
import sys
import json
import re
import time
import argparse
import csv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.stt   import transcribe_file_path
from app.llm   import generate_response_stateless
from app.tts   import synthesize_speech
from app.utils import (
    normalize_text, detect_language_tags,
    save_log, load_manifest, RateLimiter,
    compute_wer, compute_cer,
)

# ─── Konfigurasi ─────────────────────────────────────────────────────────────
PREPROCESSED_DIR = BASE_DIR / "data" / "corpus" / "preprocessed"
LOG_DIR          = BASE_DIR / "log"
RESULTS_JSON     = LOG_DIR / "pipeline_results.json"
SUMMARY_CSV      = LOG_DIR / "pipeline_summary.csv"
REPORT_TXT       = LOG_DIR / "pipeline_report.txt"
TTS_OUTPUT_DIR   = BASE_DIR / "data" / "corpus" / "tts_output"

# Rate limiter: gemini-2.0-flash free tier ~10 RPM aman
RATE_LIMITER = RateLimiter(rpm=3)

VALID_MODES = {"preserve", "normalized", "translate"}

# Koreksi ringan khusus domain voice customer service.
# Tujuannya hanya memperbaiki salah dengar STT yang sering muncul,
# tanpa mengubah struktur pipeline utama.
DOMAIN_CORRECTIONS = {
    "flag": "flight",
    "flaik": "flight",
    "flait": "flight",
    "fly": "flight",
    "flight ticket": "flight",
    "buk": "book",
    "dubuk": "book",
    "bukfaik": "book flight",
    "bukflaik": "book flight",
    "book fly": "book flight",
    "book tiket": "book ticket",
    "skedul": "schedule",
    "sekedul": "schedule",
    "range": "arrange",
    "orintz": "arrange",
    "orinj": "arrange",
    "alrits": "arrange",
    "erindz": "arrange",
    "orange transport": "arrange transport",
    "range transport": "arrange transport",
    "transportasi": "transport",
    "jeda": "Jeddah",
    "jedah": "Jeddah",
    "jendah": "Jeddah",
    "jidah": "Jeddah",
    "jidda": "Jeddah",
    "jaddah": "Jeddah",
    "jiddah": "Jeddah",
    "jeddah": "Jeddah",
    "jeddha": "Jeddah",
    "judea": "Jeddah",
    "jabodah": "Jeddah",
    "madina": "Madinah",
    "madinda": "Madinah",
    "madinah": "Madinah",
    "madinahh": "Madinah",
    "medina": "Madinah",
    "medinah": "Madinah",
    "mekah": "Makkah",
    "meka": "Makkah",
    "mekkah": "Makkah",
    "makkahh": "Makkah",
    "menjedah": "min Jeddah",
    "minjida": "min Jeddah",
    "ilajadah": "ila Jeddah",
    "ilajedah": "ila Jeddah",
    "ilah": "ila",
    "kogadan": "ghadan",
    "wadan": "ghadan",
    "hadan": "ghadan",
    "ghodan": "ghadan",
    "khadat": "ghadan",
    "khadim": "qadim",
    "al qi": "akhi",
    "yaakhi": "ya akhi",
    "uridubu": "uridu book",
    "uri du": "uridu",
    "umroh": "umrah",
    "omrah": "umrah",
    "umra": "umrah",
    "umraah": "umrah",
    "umur": "umrah",
    "haji": "hajj",
    "haj": "hajj",
    "travel umroh": "travel umrah",
    "travel umrah simple": "travel umrah simple",
    "madinah visit": "Madinah visit",
    "medina visit": "Madinah visit",
    "haramain railway": "Haramain High Speed Railway",
    "haramain train": "Haramain High Speed Railway",
    "saudi arabia": "Saudi",
    "visa saudi": "visa Saudi",
    "visa ksa": "visa Saudi",
    "ksa": "KSA",
    "fisit": "visit",
    "visitasi": "visit",
    "travell": "travel",
    "travle": "travel",
    "prepare dokumen": "prepare documents",
    "cheklist": "checklist",
    "ceklist": "checklist",
    "budjet": "budget",
    "otel": "hotel",
    "hotell": "hotel",
    "get": "guide",
    "guid": "guide",
    "gaid": "guide",
    "fasting": "fasting",
    "ramadhan": "Ramadan",
    "to morrow": "tomorrow",
    "overwelmed": "overwhelmed",
    "ovrwhelmed": "overwhelmed",
    "mulai mana": "mulai dari mana",
    "mulay": "mulai",
    "di bawah": "dibawa",
    "expand step by step": "explain step by step",
}

EN_WORDS = {
    "can", "you", "help", "arrange", "transport", "from", "to", "tomorrow",
    "explain", "step", "by", "apply", "visa", "saudi", "booking", "book",
    "flight", "online", "schedule", "travel", "app", "website", "destination",
    "payment", "passenger", "details", "official", "portal", "account",
    "application", "form", "upload", "required", "documents", "passport",
    "approval", "email", "high", "speed", "railway", "fast", "trip",
    "include", "simple", "budget", "checklist", "prepare", "documents",
    "valid", "agent", "package", "packages", "basic", "economy", "city",
    "tour", "private", "car", "bus", "taxi", "train", "hotel", "near",
    "budget", "tips", "overwhelmed"
}

ID_WORDS = {
    "aku", "dari", "ke", "dengan", "benar", "cara", "secara", "jelaskan",
    "bisa", "bantu", "pilih", "isi", "lalu", "setelah", "pertama",
    "terakhir", "tiket", "atau", "mau", "butuh", "minggu", "depan", "umrah", "haji", "puasa",
    "ramadan", "wajib", "muslim", "persiapan", "barang", "dibawa",
    "dekat", "terbatas", "sekarang", "proses", "visa", "hotel"
}

ARABIZI_WORDS = {
    "uridu", "ila", "min", "gadan", "usbu", "qadim", "afdal",
    "mubasyurah", "hal", "ana", "li", "al", "akhi", "ghadan", "tafaddal", 
    "abshir", "na'am", "ahlan", "sahlan", "turidu", "tufaddil", "aw", "innahu",
    "sa", "u", "awinuk",
}


def apply_domain_normalization(text: str) -> str:
    """
    Normalisasi tambahan khusus hasil STT.
    Fungsi ini dipakai setelah normalize_text() dari app.utils.
    """
    normalized = text

    # Koreksi frasa dulu, lalu kata tunggal.
    for wrong, correct in DOMAIN_CORRECTIONS.items():
        pattern = r"\b" + re.escape(wrong) + r"\b"
        normalized = re.sub(pattern, correct, normalized, flags=re.IGNORECASE)

    # Rapikan spasi ganda yang mungkin muncul setelah koreksi.
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def detect_language_tags_preserve(text: str) -> list[dict]:
    """
    Deteksi bahasa sederhana untuk mode preserve.
    Dibuat agar code-switching Latin Indonesia/English/Arabizi tidak selalu
    jatuh ke satu label saja.
    """
    tokens = re.findall(r"[\w']+|[^\w\s]", text, flags=re.UNICODE)
    if not tokens:
        return []

    tagged_parts = []
    current_lang = None
    current_tokens = []

    def guess_lang(token: str) -> str:
        low = token.lower()

        if re.search(r"[\u0600-\u06FF]", token):
            return "ar"
        if low in EN_WORDS:
            return "en"
        if low in ID_WORDS:
            return "id"
        if low in ARABIZI_WORDS:
            return "ar-Latn"
        if re.fullmatch(r"[^\w\s]", token):
            return current_lang or "id"
        return "id"

    for token in tokens:
        lang = guess_lang(token)

        if current_lang is None:
            current_lang = lang

        if lang != current_lang and current_tokens:
            tagged_parts.append({
                "lang": current_lang,
                "text": " ".join(current_tokens).replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
            })
            current_tokens = []
            current_lang = lang

        current_tokens.append(token)

    if current_tokens:
        tagged_parts.append({
            "lang": current_lang,
            "text": " ".join(current_tokens).replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
        })

    return tagged_parts



# ─── Pipeline satu file ───────────────────────────────────────────────────────

def process_file(idx: int, audio_path: Path, mode: str, references: dict) -> dict:
    result = {
        "index": idx,
        "file":             audio_path.name,
        "path":             str(audio_path.relative_to(BASE_DIR)),
        "timestamp":        datetime.now().isoformat(),
        "mode":             mode,
        "stt_raw":          "",
        "stt_normalized":   "",
        "lang_tags":        [],
        "langs_detected":   [],
        "llm_response":     "",
        "tts_output_path":  "",
        "tts_latency_s":    0.0,
        "final_audio_status": "skipped",
        "wer":              None,
        "cer":              None,
        "stt_latency_s":    0.0,
        "llm_latency_s":    0.0,
        "total_latency_s":  0.0,
        "status":           "ok",
        "error":            "",
    }

    # ── 1. STT ──────────────────────────────────────────────────────────────
    t0 = time.time()
    transcript_raw = transcribe_file_path(str(audio_path))
    result["stt_latency_s"] = round(time.time() - t0, 2)

    if transcript_raw.startswith("[ERROR]"):
        result["status"] = "stt_error"
        result["error"]  = transcript_raw
        return result

    result["stt_raw"] = transcript_raw

    # ── 2. Normalisasi & Language Tagging ───────────────────────────────────
    normalized = normalize_text(transcript_raw)
    normalized = apply_domain_normalization(normalized)
    result["stt_normalized"] = normalized

    # Pakai tagging preserve yang lebih peka code-switching.
    # Jika hasilnya kosong, fallback ke fungsi bawaan dari app.utils.
    lang_tags = detect_language_tags_preserve(normalized) or detect_language_tags(normalized)
    result["lang_tags"]      = lang_tags
    result["langs_detected"] = sorted(set(t["lang"] for t in lang_tags))

    if not normalized:
        result["status"] = "empty_transcript"
        return result

    # ── 3. WER / CER (jika referensi tersedia) ──────────────────────────────
    stem = audio_path.stem
    if stem in references:
        ref_text = references[stem]
        result["wer"] = round(compute_wer(ref_text, normalized), 4)
        result["cer"] = round(compute_cer(ref_text, normalized), 4)

    # ── 4. LLM ──────────────────────────────────────────────────────────────
    RATE_LIMITER.wait()
    t1 = time.time()
    llm_resp = generate_response_stateless(normalized, mode=mode)
    result["llm_latency_s"] = round(time.time() - t1, 2)

    if llm_resp.startswith("[ERROR]"):
        result["status"] = "llm_error"
        result["error"]  = llm_resp
    else:
        result["llm_response"] = llm_resp

    # ── 5. TTS ──────────────────────────────────────────────────────────────
        TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        t2 = time.time()
        tts_output_path = TTS_OUTPUT_DIR / f"{audio_path.stem}_response.wav"

        synthesize_speech(llm_resp, str(tts_output_path))

        result["tts_latency_s"] = round(time.time() - t2, 2)
        result["tts_output_path"] = str(tts_output_path.relative_to(BASE_DIR))
        result["final_audio_status"] = "ok"

    result["total_latency_s"] = round(
        result["stt_latency_s"] + result["llm_latency_s"] + result["tts_latency_s"], 2
    )
    return result


# ─── Collect files ────────────────────────────────────────────────────────────

def collect_audio_files(audio_dir: Path, limit: int = None) -> list[Path]:
    files = sorted(audio_dir.glob("*.wav"))
    if not files:
        print(f"[WARN] Tidak ada .wav di {audio_dir}")
    if limit:
        files = files[:limit]
    return files


def load_references(ref_file: str) -> dict:
    """
    Load file referensi transkripsi untuk WER/CER.
    Format JSON: {"stem_filename": "teks referensi", ...}
    Contoh: {"2128_audio1": "Saya sedang belajar machine learning di kampus"}
    """
    if not ref_file or not Path(ref_file).exists():
        return {}
    with open(ref_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Laporan ──────────────────────────────────────────────────────────────────

def write_report(results: list[dict], mode: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ok     = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]

    def avg(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    avg_stt = avg(ok, "stt_latency_s")
    avg_llm = avg(ok, "llm_latency_s")
    avg_tot = avg(ok, "total_latency_s")
    avg_wer = avg([r for r in ok if r["wer"] is not None], "wer")
    avg_cer = avg([r for r in ok if r["cer"] is not None], "cer")

    # ── JSON ──
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── CSV ──
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "mode", "status",
            "langs_detected", "stt_normalized_snippet",
            "llm_response_snippet",
            "wer", "cer",
            "stt_latency_s", "llm_latency_s", "total_latency_s",
            "error_snippet"
        ])
        for r in results:
            writer.writerow([
                r["file"], r["mode"], r["status"],
                "|".join(r.get("langs_detected", [])),
                r["stt_normalized"][:80],
                r["llm_response"][:60],
                r.get("wer"), r.get("cer"),
                r["stt_latency_s"], r["llm_latency_s"], r["total_latency_s"],
                r["error"][:80],
            ])

    # ── TXT Report ──
    lines = [
        "=" * 70,
        " LAPORAN PIPELINE ANALISIS — Voice CS System",
        f" Tanggal      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f" Mode LLM     : {mode}",
        f" Total file   : {len(results)}",
        f" Berhasil     : {len(ok)}",
        f" Error        : {len(errors)}",
        "",
        " METRIK RATA-RATA:",
        f"   Avg STT latency : {avg_stt}s",
        f"   Avg LLM latency : {avg_llm}s",
        f"   Avg Total       : {avg_tot}s",
        f"   Avg WER         : {avg_wer if avg_wer is not None else 'N/A (tidak ada referensi)'}",
        f"   Avg CER         : {avg_cer if avg_cer is not None else 'N/A (tidak ada referensi)'}",
        "=" * 70,
        "",
    ]

    # Distribusi bahasa
    all_langs = []
    for r in ok:
        all_langs.extend(r.get("langs_detected", []))
    lang_dist = {}
    for l in all_langs:
        lang_dist[l] = lang_dist.get(l, 0) + 1
    lines.append("DISTRIBUSI BAHASA TERDETEKSI:")
    for lang, cnt in sorted(lang_dist.items()):
        lines.append(f"  {lang}: {cnt} file")
    lines.append("")

    # Detail per file
    lines.append("DETAIL PER FILE:")
    lines.append("─" * 70)
    for r in results:
        lines.append(f"FILE : {r['file']}")
        lines.append(f"  Status         : {r['status']}")
        lines.append(f"  Bahasa         : {', '.join(r.get('langs_detected', []))}")
        if r["stt_normalized"]:
            lines.append(f"  Transkripsi    : {r['stt_normalized'][:100]}")
        if r["llm_response"]:
            lines.append(f"  LLM Respons    : {r['llm_response'][:100]}")
        if r.get("wer") is not None:
            lines.append(f"  WER={r['wer']}  CER={r['cer']}")
        if r["error"]:
            lines.append(f"  Error          : {r['error'][:100]}")
        lines.append(
            f"  Latency        : STT={r['stt_latency_s']}s | "
            f"LLM={r['llm_latency_s']}s | Total={r['total_latency_s']}s"
        )
        lines.append("")

    # Error summary
    if errors:
        lines.append("─" * 70)
        lines.append("ERROR SUMMARY:")
        for r in errors:
            lines.append(f"  [{r['status']}] {r['file']}: {r['error'][:80]}")
        lines.append("")

    # Catatan eksperimen
    lines += [
        "─" * 70,
        "CATATAN EKSPERIMEN:",
        "  - Model STT    : whisper.cpp large-v3-turbo (deteksi bahasa otomatis)",
        f"  - Mode LLM     : {mode}",
        "  - Model LLM    : gemini-2.0-flash (atau sesuai .env GEMINI_MODEL)",
        "  - Preprocessing: semua audio di-convert ke WAV 16kHz mono PCM-16 via ffmpeg",
        "  - File bermasalah (MP3/M4A/MP4 berekstensi .wav) otomatis di-convert",
        "  - Rate limiter : 10 RPM untuk menjaga batas free tier Gemini",
        "  - Language tagging berbasis Unicode script (Arab vs Latin)",
        "  - WER/CER: hanya dihitung jika file referensi disediakan via --ref-file",
        "=" * 70,
    ]

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OUTPUT]")
    print(f"  JSON    : {RESULTS_JSON}")
    print(f"  CSV     : {SUMMARY_CSV}")
    print(f"  Report  : {REPORT_TXT}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline analisis STT→LLM pada seluruh corpus audio"
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=PREPROCESSED_DIR,
        help="Folder audio (default: data/corpus/preprocessed)"
    )
    parser.add_argument(
        "--mode", default="preserve", choices=list(VALID_MODES),
        help="Mode LLM: preserve | normalized | translate (default: preserve)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Batasi jumlah file (untuk testing)"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Mulai dari indeks file ke-n untuk batch processing"
    )
    parser.add_argument(
        "--ref-file", type=str, default=None,
        help="Path ke JSON referensi transkripsi untuk hitung WER/CER"
    )
    args = parser.parse_args()

    audio_dir = args.audio_dir
    if not audio_dir.exists():
        print(f"[ERROR] Folder tidak ditemukan: {audio_dir}")
        print("Jalankan dulu: python preprocess_audio.py")
        sys.exit(1)

    all_files = collect_audio_files(audio_dir, None)
    files = all_files[args.start: args.start + args.limit] if args.limit else all_files[args.start:]
    if not files:
        print(f"[ERROR] Tidak ada .wav di {audio_dir}")
        sys.exit(1)

    references = load_references(args.ref_file)
    if references:
        print(f"[INFO] Referensi dimuat: {len(references)} entri")

    print(f"\n{'='*60}")
    print(f" Pipeline Analisis — {len(files)} file | Mode: {args.mode}")
    print(f" Audio dir : {audio_dir}")
    print(f"{'='*60}\n")

    results = []
    for i, fpath in enumerate(files, 1):
        print(f"[{i:03d}/{len(files):03d}] {fpath.name}")
        result = process_file(i, fpath, args.mode, references)
        results.append(result)

        if result["status"] == "ok":
            wer_str = f"  WER={result['wer']}" if result.get("wer") is not None else ""
            print(f"  STT  : {result['stt_normalized'][:70]}")
            print(f"  LLM  : {result['llm_response'][:70]}")
            print(f"  TTS  : {result['tts_output_path']}")
            print(f"  Bahasa: {result['langs_detected']}  "
                  f"Latency={result['total_latency_s']}s{wer_str}")
        else:
            print(f"  [!] {result['status']} — {result['error'][:70]}")
        print()

    write_report(results, args.mode)
    print(f"\n✓ Selesai: {len(results)} file diproses.")


if __name__ == "__main__":
    main()
