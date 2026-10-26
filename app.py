from fastapi import FastAPI, Request

# Создаём веб-приложение
app = FastAPI()


# Проверка, что сервер жив. Сюда же будет "стучать" будильник cron-job.org.
# HEAD добавлен, потому что некоторые сервисы проверки шлют именно его.
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok"}


# Сюда Яндекс присылает фразы пользователя (методом POST)
@app.post("/")
async def alice_webhook(request: Request):
    # Пытаемся прочитать JSON, который прислал Яндекс
    try:
        data = await request.json()
    except Exception:
        data = {}

    # Что сказал пользователь (пока не используем, но уже достаём)
    user_text = data.get("request", {}).get("original_utterance", "")
    print("Пользователь сказал:", user_text, flush=True)  # видно в логах Render

    # Ответ в формате, который требует Яндекс
    return {
        "version": "1.0",
        "response": {
            "text": "Привет!",
            "end_session": False,
        },
    }
