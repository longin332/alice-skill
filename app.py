import asyncio
import json
import os
import re
import time

import httpx
from fastapi import BackgroundTasks, FastAPI, Request

app = FastAPI()

# =====================================================================
# НАСТРОЙКИ. Всё берётся из переменных окружения Render.
# =====================================================================

# --- Порядок моделей и общий лимит времени ---
PROVIDER_ORDER = [
    p.strip()
    for p in os.environ.get("PROVIDER_ORDER", "groq,gemini,openrouter").split(",")
    if p.strip()
]
TOTAL_BUDGET = float(os.environ.get("TOTAL_BUDGET", "3.8"))
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

# --- Память (Upstash Redis) ---
UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
HISTORY_MAX_MESSAGES = int(os.environ.get("HISTORY_MAX_MESSAGES", "6"))
HISTORY_TTL = int(os.environ.get("HISTORY_TTL", "3600"))
MEMORY_TIMEOUT = float(os.environ.get("MEMORY_TIMEOUT", "0.5"))

# --- Пароль для /debug ---
DEBUG_TOKEN = os.environ.get("DEBUG_TOKEN", "")

# =====================================================================
# ТЕКСТЫ
# =====================================================================
SYSTEM_PROMPT = (
    "Ты голосовой помощник в колонке. Отвечай по-русски. Обычно отвечай коротко: "
    "одно-два предложения. Если пользователь просит подробнее или пояснить, "
    "ответь развёрнуто, но не длиннее пяти предложений. Учитывай предыдущие "
    "реплики разговора. Без списков, без markdown, без эмодзи, без ссылок — "
    "текст будет озвучен голосом. Не называй себя Алисой. Если не знаешь ответа, "
    "честно скажи об этом."
)
WELCOME_TEXT = (
    "Привет! Я отвечаю на вопросы коротко. Спроси что-нибудь, например: "
    "сколько воды пить в день. Чтобы узнать команды, скажи «помощь»."
)
HELP_TEXT = (
    "Просто задай вопрос своими словами, и я коротко отвечу. Если нужно больше, "
    "скажи «поясни подробнее». Чтобы выйти, скажи «хватит»."
)
ABILITIES_TEXT = (
    "Я отвечаю на вопросы в одно-два предложения, помню, о чём мы говорили, "
    "и могу пояснить подробнее. Ответы готовит нейросеть, в них возможны неточности."
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
# ПАМЯТЬ ДИАЛОГА (Upstash Redis)
# =====================================================================
def memory_enabled() -> bool:
    return bool(UPSTASH_URL and UPSTASH_TOKEN)


async def redis_command(*args):
    """Отправляет одну команду в Upstash, например ("GET", "ключ")."""
    resp = await http_client.post(
        UPSTASH_URL,
        headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
        json=[str(a) for a in args],
        timeout=5,
    )
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Upstash: {data['error']}")
    return data.get("result")


def history_key(session_id: str) -> str:
    return f"hist:{session_id}"


async def load_history(session_id: str) -> list:
    """Достаёт историю сессии. При любой проблеме — пустая история, навык работает дальше."""
    if not memory_enabled() or not session_id:
        return []
    try:
        raw = await asyncio.wait_for(
            redis_command("GET", history_key(session_id)), timeout=MEMORY_TIMEOUT
        )
        return json.loads(raw) if raw else []
    except Exception as e:
        print(f"Память (чтение) не сработала: {describe_error(e)}", flush=True)
        return []


async def save_history(session_id: str, history: list, question: str, answer: str):
    """Добавляет пару вопрос-ответ и сохраняет. Вызывается уже ПОСЛЕ ответа Алисе."""
    if not memory_enabled() or not session_id:
        return
    new_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    new_history = new_history[-HISTORY_MAX_MESSAGES:]  # храним только последние
    try:
        await redis_command(
            "SET",
            history_key(session_id),
            json.dumps(new_history, ensure_ascii=False),
            "EX",
            HISTORY_TTL,
        )
    except Exception as e:
        print(f"Память (запись) не сработала: {describe_error(e)}", flush=True)


# =====================================================================
# МОДЕЛИ. Все принимают messages — список реплик:
# [{"role": "user" или "assistant", "content": "текст"}, ...]
# =====================================================================
async def ask_gemini(messages: list) -> str:
    """Запрос к Gemini."""
    if not GEMINI_API_KEY:
        raise RuntimeError("нет ключа GEMINI_API_KEY")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    # У Gemini ответы модели называются "model", а не "assistant"
    contents = [
        {
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": [{"text": m["content"]}],
        }
        for m in messages
    ]
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 500, "temperature": 0.7},
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


async def ask_openai_style(url: str, api_key: str, model: str, messages: list) -> str:
    """Запрос к сервисам с форматом как у OpenAI (Groq и OpenRouter)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "max_tokens": 500,
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


async def ask_groq(messages: list) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("нет ключа GROQ_API_KEY")
    return await ask_openai_style(
        "https://api.groq.com/openai/v1/chat/completions", GROQ_API_KEY, GROQ_MODEL, messages
    )


async def ask_openrouter(messages: list) -> str:
    if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
        raise RuntimeError("не заданы OPENROUTER_API_KEY или OPENROUTER_MODEL")
    return await ask_openai_style(
        "https://openrouter.ai/api/v1/chat/completions",
        OPENROUTER_API_KEY,
        OPENROUTER_MODEL,
        messages,
    )


PROVIDERS = {
    "groq": (ask_groq, GROQ_TIMEOUT),
    "gemini": (ask_gemini, GEMINI_TIMEOUT),
    "openrouter": (ask_openrouter, OPENROUTER_TIMEOUT),
}


async def ask_with_fallback(messages: list, total_budget: float):
    """Пробует модели по очереди. Возвращает (ответ, имя_модели, журнал)."""
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
            answer = await asyncio.wait_for(func(messages), timeout=timeout)
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
async def debug(
    background_tasks: BackgroundTasks,
    token: str = "",
    q: str = "Сколько планет в Солнечной системе?",
    only: str = "",
    s: str = "",
):
    """
    /debug?token=ПАРОЛЬ                 — вся цепочка, без памяти
    /debug?token=ПАРОЛЬ&only=groq       — одна модель, ждём до 25 секунд
    /debug?token=ПАРОЛЬ&s=test&q=...    — с памятью: s — имя тестовой сессии
    """
    if not DEBUG_TOKEN or token != DEBUG_TOKEN:
        return {"error": "нет доступа"}

    start = time.monotonic()
    if only:
        if only not in PROVIDERS:
            return {"error": f"нет модели {only}, есть: {list(PROVIDERS)}"}
        try:
            answer = await asyncio.wait_for(
                PROVIDERS[only][0]([{"role": "user", "content": q}]), timeout=25
            )
            return {"ok": True, "provider": only, "seconds": round(time.monotonic() - start, 2), "answer": answer}
        except Exception as e:
            return {"ok": False, "provider": only, "seconds": round(time.monotonic() - start, 2), "error": describe_error(e)}

    session_id = f"debug-{s}" if s else ""
    history = await load_history(session_id)
    messages = history + [{"role": "user", "content": q}]
    answer, name, log = await ask_with_fallback(messages, TOTAL_BUDGET)
    if answer and session_id:
        background_tasks.add_task(save_history, session_id, history, q, answer)
    return {
        "ok": answer is not None,
        "answered_by": name,
        "seconds": round(time.monotonic() - start, 2),
        "answer": answer or FALLBACK_TEXT,
        "history_before": len(history),
        "memory_enabled": memory_enabled(),
        "log": log,
    }


@app.post("/")
async def alice_webhook(request: Request, background_tasks: BackgroundTasks):
    request_start = time.monotonic()
    try:
        data = await request.json()
    except Exception:
        data = {}

    req = data.get("request", {})
    session = data.get("session", {})
    command = req.get("command", "").strip().lower()
    utterance = req.get("original_utterance", "").strip()
    session_id = session.get("session_id", "")

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

    question = utterance or command
    history = await load_history(session_id)
    messages = history + [{"role": "user", "content": question}]

    # Время, потраченное на чтение памяти, вычитаем из общего лимита
    budget = TOTAL_BUDGET - (time.monotonic() - request_start)
    answer, name, log = await ask_with_fallback(messages, budget)
    print(f"Вопрос: {utterance!r} | история: {len(history)} | " + " | ".join(log), flush=True)

    if answer:
        # Сохраняем уже ПОСЛЕ того, как ответ ушёл Алисе
        background_tasks.add_task(save_history, session_id, history, question, answer)
        return make_response(answer)
    return make_response(FALLBACK_TEXT)
