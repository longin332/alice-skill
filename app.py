import asyncio
import os
import re
import time

import httpx
from fastapi import FastAPI, Request

app = FastAPI()

# =====================================================================
# НАСТРОЙКИ. Всё берётся из переменных окружения Render.
# Если переменной нет — используется значение по умолчанию (второе в скобках).
# =====================================================================

# Порядок моделей: первая — основная, дальше запасные (через запятую)
PROVIDER_ORDER = [
    p.strip()
    for p in os.environ.get("PROVIDER_ORDER", "groq,gemini,openrouter").split(",")
    if p.strip()
]
# Сколько секунд всего даём на все попытки (у Алисы лимит 4,5 с на весь путь)
TOTAL_BUDGET = float(os.environ.get("TOTAL_BUDGET", "3.8"))
# Если осталось меньше этого — следующую модель уже не пробуем
MIN_ATTEMPT_TIME = 0.5

# --- Groq ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TIMEOUT = float(os.environ.get("GROQ_TIMEOUT", "2.0"))

# --- Gemini ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "1.5"))

# --- OpenRouter ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "")
OPENROUTER_TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", "1.5"))

# Пароль для проверочного адреса /debug. Если не задан — адрес выключен.
DEBUG_TOKEN = os.environ.get("DEBUG_TOKEN", "")

# =====================================================================
# ТЕКСТЫ
# =====================================================================
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


# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================
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


# =====================================================================
# МОДЕЛИ
# =====================================================================
async def ask_gemini(question: str) -> str:
    """Запрос к Gemini."""
    if not GEMINI_API_KEY:
        raise RuntimeError("нет ключа GEMINI_API_KEY")
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
    resp = await http_client.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    answer = clean_for_voice(" ".join(p.get("text", "") for p in parts if not p.get("thought")))
    if not answer:
        raise RuntimeError("пустой ответ")
    return answer


async def ask_openai_style(url: str, api_key: str, model: str, question: str) -> str:
    """Запрос к сервисам с форматом как у OpenAI (Groq и OpenRouter устроены так)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": 300,
        "temperature": 0.7,
    }
    resp = await http_client.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    answer = clean_for_voice(data["choices"][0]["message"].get("content") or "")
    if not answer:
        raise RuntimeError("пустой ответ")
    return answer


async def ask_groq(question: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("нет ключа GROQ_API_KEY")
    return await ask_openai_style(
        "https://api.groq.com/openai/v1/chat/completions", GROQ_API_KEY, GROQ_MODEL, question
    )


async def ask_openrouter(question: str) -> str:
    if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
        raise RuntimeError("не заданы OPENROUTER_API_KEY или OPENROUTER_MODEL")
    return await ask_openai_style(
        "https://openrouter.ai/api/v1/chat/completions",
        OPENROUTER_API_KEY,
        OPENROUTER_MODEL,
        question,
    )


# Название -> (функция, сколько максимум ждём эту модель)
PROVIDERS = {
    "groq": (ask_groq, GROQ_TIMEOUT),
    "gemini": (ask_gemini, GEMINI_TIMEOUT),
    "openrouter": (ask_openrouter, OPENROUTER_TIMEOUT),
}


async def ask_with_fallback(question: str, total_budget: float):
    """
    Пробует модели по очереди из PROVIDER_ORDER.
    Возвращает (ответ, имя_модели, журнал). Если никто не ответил — (None, None, журнал).
    """
    start = time.monotonic()
    log = []
    for name in PROVIDER_ORDER:
        if name not in PROVIDERS:
            log.append(f"{name}: неизвестное имя, пропуск")
            continue
        func, provider_timeout = PROVIDERS[name]

        remaining = total_budget - (time.monotonic() - start)
        if remaining < MIN_ATTEMPT_TIME:
            log.append("время вышло")
            break
        timeout = min(provider_timeout, remaining)

        t0 = time.monotonic()
        try:
            # wait_for жёстко обрывает запрос, если он идёт дольше timeout
            answer = await asyncio.wait_for(func(question), timeout=timeout)
            log.append(f"{name}: ок за {time.monotonic() - t0:.2f} с")
            return answer, name, log
        except Exception as e:
            log.append(f"{name}: {describe_error(e)[:200]} ({time.monotonic() - t0:.2f} с)")
    return None, None, log


# =====================================================================
# АДРЕСА СЕРВЕРА
# =====================================================================
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok"}


@app.get("/debug")
async def debug(token: str = "", q: str = "Сколько планет в Солнечной системе?", only: str = ""):
    """
    Проверка в браузере.
    /debug?token=ПАРОЛЬ            — вся цепочка, как для Алисы (с лимитом времени)
    /debug?token=ПАРОЛЬ&only=groq  — только одна модель, ждём до 25 секунд
    """
    if not DEBUG_TOKEN or token != DEBUG_TOKEN:
        return {"error": "нет доступа"}

    start = time.monotonic()
    if only:
        if only not in PROVIDERS:
            return {"error": f"нет модели {only}, есть: {list(PROVIDERS)}"}
        try:
            answer = await asyncio.wait_for(PROVIDERS[only][0](q), timeout=25)
            return {"ok": True, "provider": only, "seconds": round(time.monotonic() - start, 2), "answer": answer}
        except Exception as e:
            return {"ok": False, "provider": only, "seconds": round(time.monotonic() - start, 2), "error": describe_error(e)}

    answer, name, log = await ask_with_fallback(q, TOTAL_BUDGET)
    return {
        "ok": answer is not None,
        "answered_by": name,
        "seconds": round(time.monotonic() - start, 2),
        "answer": answer or FALLBACK_TEXT,
        "log": log,
        "order": PROVIDER_ORDER,
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

    answer, name, log = await ask_with_fallback(utterance or command, TOTAL_BUDGET)
    print(f"Вопрос: {utterance!r} | " + " | ".join(log), flush=True)
    return make_response(answer if answer else FALLBACK_TEXT)
