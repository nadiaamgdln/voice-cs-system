# Voice Code-Switching System (UAS NLP 2025/2026)

Sistem multilingual **Speech-to-Speech end-to-end** yang mendukung code-switching Bahasa Indonesia, English, dan Arab.

Project ini mengimplementasikan workflow berikut:
Audio Input -> STT -> Normalisasi/Tagging -> LLM -> TTS -> Audio Output

---

## Ringkasan Proyek

Sistem ini bertujuan untuk menghadirkan pipeline voice-based code-switching yang:
- mendukung input campuran Bahasa Indonesia + English + Arab
- bisa mempertahankan code-switching (`preserve`)
- bisa menormalisasi hasil transkripsi ke Bahasa Indonesia (`normalized`)
- bisa menerjemahkan input ke Bahasa Indonesia (`translate`)

Sistem terdiri dari backend FastAPI, modul STT/LLM/TTS, dan prototipe frontend Gradio.

---

## Struktur Proyek

```
voice-cs-system/
├── app/
│   ├── main.py           # FastAPI backend dengan endpoint voice-chat dan debug
│   ├── stt.py            # STT dengan whisper.cpp
│   ├── llm.py            # LLM dengan Google Gemini API
│   ├── tts.py            # TTS dengan Coqui TTS
│   ├── utils.py          # Normalisasi, language tagging, utilities
│   ├── coqui_tts/        # Model Coqui TTS
│   │   ├── config.json
│   │   ├── checkpoint_1260000-inference.pth
│   │   └── speakers.pth
│   └── whisper.cpp/      # whisper.cpp source/build (opsional)
├── data/
│   ├── corpus/
│   │   ├── audio/        # Audio mentah
│   │   |── preprocessed/ # Audio hasil preprocessing
│   │   |── transcripts/
│   │   |      └── reference.json
│   │   └── tts_output/   # Audio hasil tts
│   └── manifests/
│       └── corpus_manifest.json
├── gradio_app/
│   └── app.py            # Prototype UI Gradio
├── log/                  # Log pipeline dan evaluasi
├── models/
│   └── whisper.cpp/      # Model whisper lokal
├── preprocess_audio.py   # Script preprocessing audio corpus
├── analisis_pipeline.py  # Pipeline analisis batch STT -> LLM
├── requirements.txt
├── .env                  # API key lokal (jangan commit)
├── .gitignore
└── README.md
```

---

## Setup & Penggunaan

### 1. Buat Virtual Environment

```bash
python -m venv env
# Windows:
env\Scripts\activate
# Linux/macOS:
source env/bin/activate

pip install -r requirements.txt
pip install -U google-genai
```

> Disarankan menggunakan **Python 3.11**.

---

### 2. Konfigurasi Gemini API Key

Buat API key di Google AI Studio lalu simpan di `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

> Jangan commit file `.env`.

---

### 3. Jalankan Backend FastAPI

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoint utama:
- `POST /voice-chat` -> pipeline audio -> audio
- `POST /transcribe` -> debug STT
- `POST /generate` -> debug LLM

---

### 4. Jalankan Gradio UI

Pastikan backend sudah berjalan, lalu jalankan:

```bash
python gradio_app/app.py
```

Buka browser: `http://localhost:7860`

---

## Alur Sistem

1. `Audio` direkam dari pengguna
2. `STT` jalankan whisper.cpp untuk transkripsi
3. `Normalisasi` dan `language tagging`
4. `LLM` menghasilkan respons teks
5. `TTS` mengubah teks menjadi suara
6. `Audio Output` dikembalikan ke UI

---

## Struktur Data & Model

- `models/whisper.cpp/` -> tempat model Whisper lokal
- `app/coqui_tts/` -> tempat model Coqui TTS
- `data/corpus/audio/` -> audio mentah
- `data/corpus/preprocessed/` -> audio hasil preprocessing
- `log/` -> output pipeline dan laporan evaluasi

---

## Preprocessing Audio Corpus

Jalankan:

```bash
python preprocess_audio.py
```

Output:
- `data/corpus/preprocessed/`
- `data/manifests/corpus_manifest.json`

Fungsi:
- convert semua file audio ke WAV 16kHz mono PCM-16
- deteksi format MP3/M4A/MP4 yang disamarkan sebagai WAV
- buat manifest data otomatis

---

## Analisis Batch

Jalankan semua data:

```bash
python analisis_pipeline.py
```

Output:
- `log/pipeline_results.json`
- `log/pipeline_summary.csv`
- `log/pipeline_report.txt`

---

## Mode Sistem

- `preserve` -> pertahankan pola code-switching input
- `normalized` -> respons dalam Bahasa Indonesia baku
- `translate` -> terjemahkan ke Bahasa Indonesia

---

## Fitur Utama

- Support code-switching Indo/EN/Arab
- Debug STT dan LLM terpisah untuk validasi
- UI Gradio untuk prototipe interaktif
- Backend FastAPI untuk integrasi endpoint
- Output evaluasi tersimpan di `log/`

---

## Tips

- Pastikan backend sudah jalan sebelum buka UI.
- Jika Gemini quota habis, beri jeda di pipeline batch.
- Gunakan `transformers==5.0` untuk stabilitas Coqui TTS.
- Jangan simpan API key langsung di kode sumber.

---

## Catatan Penting

Beberapa komponen tidak disertakan dalam repository ini karena ukuran file yang besar atau alasan keamanan, yaitu:

- Folder `app/whisper.cpp`
- Model Whisper (`*.bin`)
- Model Coqui TTS (`*.pth`)
- File konfigurasi `.env`

Untuk menggunakan modul Speech-to-Text, lakukan clone Whisper.cpp secara manual:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git app/whisper.cpp
```

Setelah itu, unduh model Whisper yang diperlukan dan tempatkan pada direktori model yang sesuai.

## Referensi

- [whisper.cpp — Local Speech-to-Text Engine](https://github.com/ggml-org/whisper.cpp)
- [OpenAI Whisper Models](https://github.com/openai/whisper)
- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Google GenAI Python SDK](https://github.com/googleapis/python-genai)
- [Coqui TTS — Text-to-Speech Toolkit](https://github.com/idiap/coqui-ai-TTS)
- [Indonesian TTS Model (Wikidepia)](https://huggingface.co/Wikidepia/indonesian-tts/tree/main)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://www.gradio.app/docs)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [JiWER — WER/CER Evaluation](https://github.com/jitsi/jiwer)