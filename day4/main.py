import requests
import uuid
import json
import time


def get_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': 'Basic KEY'
    }
    resp = requests.post(url, headers=headers, data={'scope': 'GIGACHAT_API_PERS'}, verify=False)
    return resp.json().get('access_token')


def chat(token, message, temperature=0.7, max_tokens=500):
    payload = {
        "model": 'GigaChat',
        "messages": [{"role": "user", "content": message}],
        "stream": False,
        "repetition_penalty": 1,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    resp = requests.post(
        'https://gigachat.devices.sberbank.ru/api/v1/chat/completions',
        headers={'Authorization': f'Bearer {token}'},
        json=payload,
        verify=False
    )
    return resp.json()


QUERY = "Опиши, как мог бы выглядеть город будущего через 100 лет. Дай 3 конкретные идеи"

if __name__ == '__main__':
    print(f"Запрос: {QUERY}\n")

    token = get_token()

    temperatures = [0, 0.7, 1.2]
    results = {}

    for temp in temperatures:
        print("\n" + "=" * 80)
        print(f"🌡️  ТЕМПЕРАТУРА = {temp}")
        print("=" * 80)

        answers = []
        print(f"\n--- Запрос (temp={temp}) ---")
        response = chat(token, QUERY, temperature=temp)

        if 'choices' in response:
            answer = response['choices'][0]['message']['content']
            print(f"Ответ: {answer}")
            answers.append(answer)
        else:
            print(f"Ошибка: {response}")
            answers.append(f"ERROR: {response}")

        time.sleep(1)

        results[temp] = answers

    with open("temperature_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

