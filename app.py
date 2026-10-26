import os
import re
import time

import httpx
from fastapi import FastAPI, Request

app = FastAPI()

# ---------- НАСТРОЙКИ (берутся из переменных окружения Render) ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "3.5"))
# Пароль для проверочного адреса /debug. Если не задан — адрес выключен.
DEBUG_TOKEN = os.environ.get("DEBUG_TOKEN", "")

# ---------- ТЕКСТЫ ----------
SYSTEM_PROMPT = (
    "Ты голосовой помощник в колонке. Отвечай по-русски, коротко: одно-два "
    "предложения. Без списков, без markdown, без эмодзи, без ссылок — текст "
    "будет озвучен голосом. Не называй себя Алисой. Если не знаешь ответа, "
    "честно скажи об этом."
)
WELCOME_TEXT = (
    "Привет! Я отвечаю на вопросы коротко. Спроси что-нибудь, например: "
    "сколько воды пить в день. Чтобы узнать команды, скажи «помощь»."
)
HELP_TEXT = (
    "Просто задай вопрос своими словами, и я коротко отвечу. "
    "Чтобы выйти, скажи «хватит»."
)
ABILITIES_TEXT = (
    "Я отвечаю на вопросы в одно-два предложения: объясняю понятия, "
    "даю советы и подсказки. Ответы готовит нейросеть, в них возможны неточности."
)
FALLBACK_TEXT = "Сейчас не получается ответить, попробуй ещё раз."

HELP_COMMANDS = {"помощь", "помоги", "справка"}
ABILITIES_COMMANDS = {"что ты умеешь", "что умеешь", "что ты можешь"}

http_client = httpx.AsyncClient()


def make_response(text: str, end_session: bool = False) -> dict:
    """Собирает ответ в формате Яндекс Диалогов."""
    return {
        "version": "1.0",
        "response": {"text": text, "end_session": end_session},
    }


def clean_for_voice(text: str) -> str:
    """Убирает markdown-символы и обрезает до лимита Яндекса."""
    text = re.sub(r"[*#_`>]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 1000:
        text = text[:1000]
        last_dot = text.rfind(".")
        if last_dot > 0:
            text = text[: last_dot + 1]
    return text


def describe_error(e: Exception) -> str:
    """Название ошибки + текст, чтобы в логах было понятно, что случилось."""
    return f"{type(e).__name__}: {e}"


async def ask_gemini(question: str, timeout: float) -> str:
    """Отправляет вопрос в Gemini и возвращает текст ответа."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {"maxOutputTokens": 400, "temperature": 0.7},
    }

    resp = await http_client.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    answer = " ".join(p.get("text", "") for p in parts if not p.get("thought"))
    answer = clean_for_voice(answer)
    if not answer:
        raise RuntimeError(f"Gemini вернул пустой ответ: {str(data)[:300]}")
    return answer


@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok"}


# Проверочный адрес: открываешь в браузере, ждёт Gemini до 25 секунд
@app.get("/debug")
async def debug(token: str = "", q: str = "Сколько планет в Солнечной системе?"):
    if not DEBUG_TOKEN or token != DEBUG_TOKEN:
        return {"error": "нет доступа"}
    start = time.monotonic()
    try:
        answer = await ask_gemini(q, timeout=25)
        return {
            "ok": True,
            "model": GEMINI_MODEL,
            "seconds": round(time.monotonic() - start, 2),
            "answer": answer,
        }
    except Exception as e:
        return {
            "ok": False,
            "model": GEMINI_MODEL,
            "seconds": round(time.monotonic() - start, 2),
            "error": describe_error(e),
            "key_is_set": bool(GEMINI_API_KEY),  # только да/нет, сам ключ не показываем
        }


@app.post("/")
async def alice_webhook(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    req = data.get("request", {})
    session = data.get("session", {})
    command = req.get("command", "").strip().lower()
    utterance = req.get("original_utterance", "").strip()

    if utterance.lower() == "ping":
        return make_response("pong")

    if session.get("new") and not command:
        return make_response(WELCOME_TEXT)

    if command in HELP_COMMANDS:
        return make_response(HELP_TEXT)
    if command in ABILITIES_COMMANDS:
        return make_response(ABILITIES_TEXT)

    if not command:
        return make_response("Не расслышал, повтори, пожалуйста.")

    start = time.monotonic()
    try:
        answer = await ask_gemini(utterance or command, timeout=GEMINI_TIMEOUT)
        print(f"Вопрос: {utterance!r} | Gemini за {time.monotonic() - start:.2f} с", flush=True)
        return make_response(answer)
    except Exception as e:
        print(f"ОШИБКА за {time.monotonic() - start:.2f} с: {describe_error(e)}", flush=True)
        return make_response(FALLBACK_TEXT)
