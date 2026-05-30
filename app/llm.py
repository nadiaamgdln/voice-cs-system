import os
import time
from google import genai
from google.genai import types
from pydantic import TypeAdapter
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "models/gemma-4-31b-it")
DEFAULT_MODE = os.getenv("LLM_MODE", "preserve")

API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
    os.getenv("GEMINI_API_KEY_6"),
    os.getenv("GEMINI_API_KEY_7"),
    os.getenv("GEMINI_API_KEY_8"),
    os.getenv("GEMINI_API_KEY_9"),
    os.getenv("GEMINI_API_KEY_10"),
]

API_KEYS = [k for k in API_KEYS if k]

if not API_KEYS:
    raise ValueError(
        "API key Gemini tidak ditemukan di file .env.\n"
        "Pastikan .env berisi GEMINI_API_KEY_1 dan/atau GEMINI_API_KEY_2"
    )

clients = [genai.Client(api_key=k) for k in API_KEYS]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

SYSTEM_PROMPT_PRESERVE = """
You are an intelligent multilingual assistant that communicates using code-switching
between Indonesian (ID), English (EN), and Arabic (AR).

RULES:
- Detect the language mixture pattern in the user's input.
- Respond using THE SAME code-switching pattern as the user.
- If user mixes ID+EN, respond mixing ID+EN.
- If user mixes ID+AR, respond mixing ID+AR.
- If user mixes all three, respond mixing all three.
- Keep responses short and direct, maximum 2–3 sentences.
- Never repeat the user's question; answer directly.
- Be polite and conversational.
"""

SYSTEM_PROMPT_NORMALIZED = """
You are an intelligent virtual assistant that always responds in STANDARD INDONESIAN.

RULES:
- Always respond in clear, formal Indonesian.
- Keep responses short and direct, maximum 2–3 sentences.
- Never repeat the user's question; answer directly.
- Be polite and easy to understand.
"""

SYSTEM_PROMPT_TRANSLATE = """
You are a multilingual assistant.

RULES:
- Understand input in Indonesian, English, Arabic, or mixed language.
- Respond in Bahasa Indonesia only.
- Keep responses short, maximum 2–3 sentences.
- Be polite and direct.
"""

MODE_PROMPTS = {
    "preserve": SYSTEM_PROMPT_PRESERVE,
    "normalized": SYSTEM_PROMPT_NORMALIZED,
    "translate": SYSTEM_PROMPT_TRANSLATE,
}

history_adapter = TypeAdapter(list[types.Content])


def _make_config(mode: str = DEFAULT_MODE) -> types.GenerateContentConfig:
    prompt = MODE_PROMPTS.get(mode, SYSTEM_PROMPT_PRESERVE)
    return types.GenerateContentConfig(
        system_instruction=prompt,
        temperature=0.3,
        max_output_tokens=800,
    )


def _make_chat(mode: str = DEFAULT_MODE):
    return clients[0].chats.create(
        model=MODEL,
        config=_make_config(mode)
    )


def export_chat_history(chat) -> str:
    return history_adapter.dump_json(chat.get_history()).decode("utf-8")


def save_chat_history(chat):
    json_history = export_chat_history(chat)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write(json_history)


def load_chat_history(mode: str = DEFAULT_MODE):
    if not os.path.exists(CHAT_HISTORY_FILE) or os.path.getsize(CHAT_HISTORY_FILE) == 0:
        return _make_chat(mode)

    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        json_str = f.read().strip()

    if not json_str:
        return _make_chat(mode)

    try:
        history = history_adapter.validate_json(json_str)
        return clients[0].chats.create(
            model=MODEL,
            config=_make_config(mode),
            history=history
        )
    except Exception as e:
        print(f"[WARN] Gagal load history chat: {e} — mulai sesi baru.")
        return _make_chat(mode)


chat = load_chat_history()


def generate_response(prompt: str, mode: str = DEFAULT_MODE) -> str:
    global chat

    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = chat.send_message(prompt)
            save_chat_history(chat)
            time.sleep(5)
            return response.text.strip()

        except Exception as e:
            err_str = str(e).lower()

            if any(k in err_str for k in ("quota", "rate", "429", "resource exhausted")):
                wait = 20 * (attempt + 1)
                print(f"[WARN] Rate limit hit. Menunggu {wait}s...")
                time.sleep(wait)
            else:
                return f"[ERROR] Gemini: {repr(e)}"

    return "[ERROR] Gagal mendapat respons setelah beberapa percobaan."


def generate_response_stateless(prompt: str, mode: str = DEFAULT_MODE) -> str:
    config = _make_config(mode)

    max_retries_per_key = 3
    last_error = None

    for key_index, active_client in enumerate(clients):
        for attempt in range(max_retries_per_key):
            try:
                response = active_client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=config,
                )

                text = getattr(response, "text", None)

                if text and text.strip():
                    time.sleep(5)
                    return text.strip()

                last_error = "Empty response"

                print(
                    f"[WARN] Gemini key {key_index + 1} mengembalikan respons kosong "
                    f"(attempt {attempt + 1}/{max_retries_per_key})"
                )

                time.sleep(10)
                continue

            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                print(
                    f"[WARN] Gemini key {key_index + 1} gagal "
                    f"(attempt {attempt + 1}/{max_retries_per_key}): {repr(e)}"
                )

                if any(
                    k in err_str
                    for k in (
                        "quota",
                        "rate",
                        "429",
                        "resource exhausted",
                        "500",
                        "internal"
                    )
                ):
                    time.sleep(20)
                    continue
                else:
                    break

    return f"[ERROR] Semua Gemini API key gagal. Last error: {repr(last_error)}"