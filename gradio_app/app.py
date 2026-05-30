"""
gradio_app/app.py — Prototype Gradio UI untuk Voice CS System
Jalankan: python gradio_app/app.py
Pastikan FastAPI backend sudah jalan: uvicorn app.main:app --reload --port 8000
"""

import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
VALID_MODES = ["preserve", "normalized", "translate"]

CUSTOM_CSS = """
:root {
    color-scheme: light;
    font-family: Inter, 'Segoe UI', sans-serif;
}
body {
    background: radial-gradient(circle at top left, #d7ecff 0%, transparent 30%),
                linear-gradient(180deg, #f5f9ff 0%, #eef4ff 40%, #f8fbff 100%);
}
.gradio-container {
    max-width: 1160px;
    margin: auto;
    padding: 16px;
}
.gradio-block {
    border-radius: 24px;
    border: 1px solid rgba(200, 216, 242, 0.7);
    background: rgba(255, 255, 255, 0.92);
    box-shadow: 0 24px 54px rgba(42, 94, 158, 0.10);
    backdrop-filter: blur(20px);
}
.panel {
    background: linear-gradient(180deg, rgba(245, 251, 255, 0.95), rgba(226, 239, 255, 0.9));
    border: 1px solid rgba(98, 138, 233, 0.16);
    padding: 22px;
    border-radius: 26px;
}
.panel .gradio-column {
    background: transparent;
}
.voice-card {
    padding: 20px;
    border-radius: 24px;
    background: rgba(250, 253, 255, 0.98);
    border: 1px solid rgba(179, 199, 232, 0.7);
    box-shadow: 0 18px 34px rgba(68, 110, 164, 0.06);
}
.voice-card h3 {
    margin-top: 0;
    margin-bottom: 12px;
    font-size: 1.18rem;
    color: #1c3865;
}
.voice-card .card-note {
    margin-top: 0;
    margin-bottom: 18px;
    color: #5b6e8f;
    font-size: 0.95rem;
    line-height: 1.6;
}
.voice-action-button {
    display: flex;
    justify-content: center;
    width: 100%;
    margin-top: 12px;
}
.voice-action-button .gr-button {
    width: auto !important;
    min-width: 260px;
    max-width: 320px;
    padding: 14px 28px;
}
.voice-action-button .gr-button-primary {
    border-radius: 18px;
    box-shadow: 0 14px 30px rgba(62, 116, 255, 0.18);
}
.gradio-row, .gradio-column {
    gap: 18px;
}
.gradio-tabs, .gradio-tab, .gradio-tab-panel {
    border-radius: 24px;
}
.gr-button, .gr-button-primary {
    border-radius: 16px;
    font-weight: 600;
    letter-spacing: 0.01em;
    padding: 14px 18px;
}
.gr-button-primary {
    background: linear-gradient(135deg, #2858d6 0%, #3e74ff 100%);
    color: white;
}
.gr-button-primary:hover {
    filter: brightness(1.08);
}
.gradio-input, .gr-input, .gr-dropdown, .gr-textbox, .gradio-audio {
    border-radius: 18px;
    border: 1px solid rgba(149, 170, 205, 0.35);
    background: rgba(255, 255, 255, 0.95);
}
.gradio-input:focus, .gr-dropdown:focus, .gr-textbox:focus {
    border-color: rgba(62, 116, 255, 0.8);
    box-shadow: 0 0 0 4px rgba(62, 116, 255, 0.12);
}
.gradio-markdown, .gr-markdown, .gr-html {
    color: #0f1f3f;
    font-family: Inter, 'Segoe UI', sans-serif;
}
.gradio-markdown h1, .gr-markdown h1 {
    font-size: 2.4rem;
    letter-spacing: -0.04em;
}
.gradio-markdown h2, .gr-markdown h2 {
    color: #1d3168;
}
.hero-section {
    padding: 24px 26px 18px;
    margin-bottom: 12px;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(192, 211, 240, 0.7);
    box-shadow: 0 16px 32px rgba(38, 90, 170, 0.08);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 10px;
    line-height: 1.05;
}
.hero-subtitle {
    margin-top: 0;
    color: #385285;
    font-size: 1rem;
    line-height: 1.7;
    max-width: 820px;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.hero-badge {
    padding: 10px 16px;
    border-radius: 999px;
    background: rgba(64, 117, 255, 0.12);
    color: #26406e;
    font-size: 0.95rem;
    font-weight: 600;
}
.card-panel {
    padding: 24px;
    background: rgba(255, 255, 255, 0.92);
    border-radius: 24px;
    border: 1px solid rgba(199, 214, 238, 0.7);
}
.card-title {
    margin: 0 0 10px;
    font-size: 1.15rem;
    font-weight: 700;
}
.card-note {
    color: #556a96;
    line-height: 1.7;
}
"""


def voice_chat(audio, mode: str):
    """Kirim audio ke backend dengan mode yang dipilih, kembalikan audio + info."""
    if audio is None:
        return None, "⚠️ Belum ada audio yang direkam."

    sr, audio_data = audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{BACKEND_URL}/voice-chat",
                files={"file": ("voice.wav", f, "audio/wav")},
                params={"mode": mode},
                timeout=600,
            )

        if resp.status_code == 200:
            out = os.path.join(tempfile.gettempdir(), f"gradio_out_{mode}.wav")
            with open(out, "wb") as f:
                f.write(resp.content)
            return out, f"✅ Mode: **{mode}** | Respons berhasil diterima."
        else:
            return None, f"❌ Error {resp.status_code}: {resp.text[:200]}"

    except requests.exceptions.ConnectionError:
        return None, (
            f"❌ Tidak bisa terhubung ke backend di {BACKEND_URL}\n"
            "Pastikan FastAPI sudah dijalankan."
        )
    except Exception as e:
        return None, f"❌ Error: {str(e)}"
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def transcribe_debug(audio):
    """Debug: STT + language tagging tanpa LLM/TTS."""
    if audio is None:
        return "Belum ada audio.", ""

    sr, audio_data = audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{BACKEND_URL}/transcribe",
                files={"file": ("voice.wav", f, "audio/wav")},
                timeout=120,
            )
        if resp.status_code == 200:
            data = resp.json()
            transcript = data.get("transcript_normalized", "")
            lang_info   = str(data.get("language_tags", []))
            return transcript, lang_info
        return f"Error {resp.status_code}: {resp.text}", ""
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def llm_debug(text: str, mode: str):
    """Debug: LLM saja, tampilkan respons teks per mode."""
    if not text.strip():
        return "Teks kosong."
    try:
        resp = requests.post(
            f"{BACKEND_URL}/generate",
            params={"text": text, "mode": mode},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        return f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"Error: {str(e)}"


# ─── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Code-Switching Speech-to-Speech System", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=CUSTOM_CSS) as demo:
    gr.HTML("""
    <div class='hero-section'>
        <div class='hero-title'>Code-Switching Speech-to-Speech System</div>
        <div class='hero-subtitle'>Sistem multilingual end-to-end untuk speech-based code-switching: Speech → STT → normalisasi/tagging → LLM → TTS → Speech.</div>
        <div class='hero-badges'>
            <span class='hero-badge'>Multilingual</span>
            <span class='hero-badge'>End-to-end</span>
            <span class='hero-badge'>Preserve & Normalize</span>
            <span class='hero-badge'>Eksperimen & Evaluasi</span>
        </div>
    </div>
    """)

    with gr.Tab("Voice Chat"):
        gr.Markdown("""
        ### Suara ke Respon
        Gunakan mikrofon untuk merekam pertanyaan Anda, lalu pilih mode sistem untuk menghasilkan balasan suara yang sesuai.
        """)
        with gr.Row(elem_classes="panel"):
            with gr.Column(scale=3, elem_classes="voice-card"):
                gr.HTML("<h3>Input Audio</h3><p class='card-note'>Rekam suara langsung dari mikrofon lalu pilih mode action sebelum mengirim permintaan.</p>")
                audio_in = gr.Audio(sources="microphone", type="numpy", label="Rekam Pertanyaan")
                mode_sel = gr.Dropdown(
                    choices=VALID_MODES,
                    value="preserve",
                    label="Mode Sistem",
                    info="preserve=pertahankan code-switching | normalized=normalisasi bahasa | translate=terjemahkan ke Bahasa Indonesia"
                )
                submit_btn = gr.Button("Proses Audio", variant="primary", elem_classes="voice-action-button")
            with gr.Column(scale=2, elem_classes="voice-card"):
                gr.HTML("<h3>Balasan Sistem</h3><p class='card-note'>Dengarkan jawaban suara dari sistem dan lihat catatan status ketika hasil siap.</p>")
                audio_out = gr.Audio(type="filepath", label="Balasan Asisten")
                status_out = gr.Markdown()

        submit_btn.click(
            fn=voice_chat,
            inputs=[audio_in, mode_sel],
            outputs=[audio_out, status_out],
        )

    with gr.Tab("Debug STT"):
        gr.Markdown("Lihat output transkripsi dan tagging bahasa sebelum diproses lebih lanjut oleh LLM.")
        with gr.Row(elem_classes="panel"):
            with gr.Column(scale=3, elem_classes="voice-card"):
                gr.HTML("<h3>Input STT</h3><p class='card-note'>Rekam audio untuk ditranskripsi dan melihat tagging bahasa.</p>")
                dbg_audio = gr.Audio(sources="microphone", type="numpy", label="Rekam Audio")
                submit_stt = gr.Button("Jalankan Transkripsi", variant="primary", elem_classes="voice-action-button")
            with gr.Column(scale=2, elem_classes="voice-card"):
                gr.HTML("<h3>Hasil Transkripsi</h3><p class='card-note'>Lihat teks hasil transkripsi serta tag bahasa yang dikenali.</p>")
                dbg_transcript = gr.Textbox(label="Transkripsi Ter-normalisasi", lines=5)
                dbg_lang       = gr.Textbox(label="Bahasa & Tagging", lines=3)

        submit_stt.click(
            fn=transcribe_debug,
            inputs=dbg_audio,
            outputs=[dbg_transcript, dbg_lang],
        )

    with gr.Tab("Debug LLM"):
        gr.Markdown("Uji respons teks dari LLM dan bandingkan hasil untuk setiap mode.")
        with gr.Row(elem_classes="panel"):
            with gr.Column(scale=3, elem_classes="voice-card"):
                gr.HTML("<h3>Input Teks</h3><p class='card-note'>Masukkan teks dan pilih mode untuk melihat keluaran LLM.</p>")
                llm_input = gr.Textbox(label="Teks Input", lines=3,
                                       placeholder="Contoh: Jelaskan cara kerja model code-switching.")
                llm_mode  = gr.Dropdown(choices=VALID_MODES, value="preserve", label="Mode Sistem")
                submit_llm = gr.Button("Proses Teks", variant="primary", elem_classes="voice-action-button")
            with gr.Column(scale=2, elem_classes="voice-card"):
                gr.HTML("<h3>Respons LLM</h3><p class='card-note'>Output teks dari model LLM akan muncul di sini.</p>")
                llm_output = gr.Textbox(label="Respons LLM", lines=7)

        submit_llm.click(
            fn=llm_debug,
            inputs=[llm_input, llm_mode],
            outputs=llm_output,
        )

    with gr.Tab("Info"):
        with gr.Row(elem_classes="panel"):
            with gr.Column(scale=3, elem_classes="voice-card"):
                gr.HTML("<h3>Sistem & Pipeline</h3><p class='card-note'>Sistem speech-to-speech multilingual dengan alur: Speech → STT → normalisasi/tagging → LLM → TTS → Speech.</p>")
                gr.Markdown(f"""
                - **STT:** whisper.cpp (model lokal)
                - **LLM:** Google Gemini API
                - **TTS:** Coqui TTS
                - **Backend:** FastAPI @ `{BACKEND_URL}`
                - **Output:** suara + laporan eksperimen
                """)
            with gr.Column(scale=2, elem_classes="voice-card"):
                gr.HTML("<h3>Mode Sistem</h3><p class='card-note'>Pilih mode untuk sesuaikan respons antara preserve CS, normalisasi, atau terjemahan.</p>")
                gr.Markdown("""
                - **preserve:** pertahankan pola code-switching
                - **normalized:** respons Bahasa Indonesia baku
                - **translate:** terjemahkan ke Bahasa Indonesia
                """)
        with gr.Row(elem_classes="panel"):
            with gr.Column(elem_classes="voice-card"):
                gr.HTML("<h3>Contoh Input</h3>")
                gr.Markdown("""
                - *"Bisa jelaskan feature engineering dalam machine learning?"*
                - *"Ana ingin tahu كيف أستخدم data augmentation."*
                - *"This system perlu dukungan multilingual dan evaluasi real-time."*
                """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
