import requests
import uuid
import json
import time
from datetime import datetime


def get_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': 'Basic KEY=='
    }
    resp = requests.post(url, headers=headers, data={'scope': 'GIGACHAT_API_PERS'}, verify=False)
    return resp.json().get('access_token')


def chat(token, message, model="GigaChat", temperature=0.7, max_tokens=500):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": False,
        "repetition_penalty": 1,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    start_time = time.time()
    resp = requests.post(
        'https://gigachat.devices.sberbank.ru/api/v1/chat/completions',
        headers={'Authorization': f'Bearer {token}'},
        json=payload,
        verify=False
    )
    elapsed_time = time.time() - start_time

    return resp.json(), elapsed_time


QUERY = """У меня есть 3 яблока. Я отдал тебе 2 яблока, а одно переложил в сумку.
Сколько яблок у меня осталось?
А теперь представь, что у тебя было 5 груш, ты съел 2, а потом купил ещё 4, и одну переложил в сумку.
Сколько груш у тебя стало?
ОТВЕТЬ ТОЛЬКО ЧИСЛАМИ через запятую."""


if __name__ == '__main__':
    print(f"Запрос: {QUERY}\n")

    token = get_token()

    models = [
        {"name": "GigaChat-2-Lite", "id": "GigaChat", "type": "Слабая (Lite)"},
        {"name": "GigaChat-2-Pro", "id": "GigaChat-Pro", "type": "Средняя (Pro)"},
        {"name": "GigaChat-2-Max", "id": "GigaChat-Max", "type": "Сильная (Max)"}
    ]

    results = []

    for model in models:
        print("\n" + "=" * 80)
        print(f"🤖 МОДЕЛЬ: {model['name']} ({model['type']})")
        print("=" * 80)

        answers = []
        times = []
        tokens_list = []

        print(f"\n--- Запрос  ---")
        response, elapsed = chat(token, QUERY, model=model['id'], max_tokens=5000)

        if 'choices' in response:
            answer = response['choices'][0]['message']['content']
            usage = response.get('usage', {})
            tokens = usage.get('total_tokens', usage.get('completion_tokens', 0))

            print(f"✅ Время ответа: {elapsed:.2f} сек")
            print(f"📊 Токенов: {tokens}")
            print(f"📝 Ответ:  {answer}")

            answers.append(answer)
            times.append(elapsed)
            tokens_list.append(tokens)
        else:
            print(f"❌ Ошибка: {response}")
            answers.append(f"ERROR: {response}")
            times.append(0)
            tokens_list.append(0)

        time.sleep(1)

        # Сохраняем статистику по модели
        results.append({
            "model": model['name'],
            "type": model['type'],
            "avg_time": sum(times) / len(times),
            "avg_tokens": sum(tokens_list) / len(tokens_list),
            "answers": answers
        })

    print(results)
    with open("model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
