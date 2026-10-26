import os
import re
import time

import httpx
from fastapi import FastAPI, Request

app = FastAPI()

# ---------- НАСТРОЙКИ (берутся из переменных окружения Render) ----------
# Ключ: только из окружения, в коде его нет
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Модель можно сменить на Render, не трогая код
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Сколько секунд ждём Gemini. У Алисы всего 4,5 с на весь ответ.
GEMINI_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "3.5"))

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

# Один клиент на всё время работы: соединение переиспользуется, так быстрее
http_client = httpx.AsyncClient()


def make_response(text: str, end_session: bool = False) -> dict:
    """Собирает ответ в формате Яндекс Диалогов."""
    return {
        "version": "1.0",
        "response": {"text": text, "end_session": end_session},
    }


def clean_for_voice(text: str) -> str:
    """Убирает markdown-символы и обрезает до лимита Яндекса (1024 символа)."""
    text = re.sub(r"[*#_`>]", "", text)      # звёздочки, решётки и т.п.
    text = re.sub(r"\s+", " ", text).strip()  # лишние пробелы и переносы
    if len(text) > 1000:
        text = text[:1000]
        last_dot = text.rfind(".")
        if last_dot > 0:
            text = text[: last_dot + 1]
    return text


async def ask_gemini(question: str) -> str:
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

    resp = await http_client.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT)
    if resp.status_code != 200:
        # Пишем в лог код и начало текста ошибки (ключ туда не попадает)
        raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    # Берём только текст ответа, пропуская служебные "мысли" модели
    answer = " ".join(p.get("text", "") for p in parts if not p.get("thought"))
    answer = clean_for_voice(answer)
    if not answer:
        raise RuntimeError("Gemini вернул пустой ответ")
    return answer


# Проверка, что сервер жив (сюда стучит будильник cron-job.org)
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok"}


# Сюда Яндекс присылает фразы пользователя
@app.post("/")
async def alice_webhook(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    req = data.get("request", {})
    session = data.get("session", {})
    command = req.get("command", "").strip().lower()          # фраза, очищенная Яндексом
    utterance = req.get("original_utterance", "").strip()     # фраза как есть

    # Служебная проверка от Яндекса — отвечаем мгновенно
    if utterance.lower() == "ping":
        return make_response("pong")

    # Навык только что запустили без вопроса — приветствие
    if session.get("new") and not command:
        return make_response(WELCOME_TEXT)

    if command in HELP_COMMANDS:
        return make_response(HELP_TEXT)
    if command in ABILITIES_COMMANDS:
        return make_response(ABILITIES_TEXT)

    if not command:
        return make_response("Не расслышал, повтори, пожалуйста.")

    # Основное: спрашиваем Gemini
    start = time.monotonic()
    try:
        answer = await ask_gemini(utterance or command)
        print(f"Вопрос: {utterance!r} | Gemini за {time.monotonic() - start:.2f} с", flush=True)
        return make_response(answer)
    except Exception as e:
        print(f"ОШИБКА за {time.monotonic() - start:.2f} с: {e}", flush=True)
        return make_response(FALLBACK_TEXT)
