import requests
import uuid


def get_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': 'TOKEN'
    }
    resp = requests.post(url, headers=headers, data={'scope': 'GIGACHAT_API_PERS'}, verify=False)
    return resp.json().get('access_token')


def chat(token, message, max_tokens=None, stop_sequences=None):
    payload = {
        "model": 'GigaChat',
        "messages": [{"role": "user", "content": message}],
        "stream": False,
        "repetition_penalty": 1
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens
    if stop_sequences:
        payload["stop"] = stop_sequences

    resp = requests.post(
        'https://gigachat.devices.sberbank.ru/api/v1/chat/completions',
        headers={'Authorization': f'Bearer {token}'},
        json=payload,
        verify=False
    )
    return resp.json()


if __name__ == '__main__':
    token = get_token()
    base_question = "Расскажи о трёх основных направлениях искусственного интеллекта"

    # 📌 Запрос 1: БЕЗ ограничений
    print("=" * 50)
    print("ЗАПРОС 1: Без ограничений")
    response1 = chat(token, base_question)
    print("Ответ:", response1['choices'][0]['message']['content'])
    print("Длина:", len(response1['choices'][0]['message']['content']))

    # 📌 Запрос 2: С ФОРМАТОМ ответа
    print("\n" + "=" * 50)
    print("ЗАПРОС 2: С явным форматом ответа")
    message2 = base_question + """

    Ответ должен быть строго в таком формате:
    Направление 1: [название] - [описание]
    Направление 2: [название] - [описание]  
    Направление 3: [название] - [описание]
    """
    response2 = chat(token, message2)
    print("Ответ:", response2['choices'][0]['message']['content'])

    # 📌 Запрос 3: С ОГРАНИЧЕНИЕМ длины
    print("\n" + "=" * 50)
    print("ЗАПРОС 3: С ограничением длины (max_tokens=50)")
    response3 = chat(token, base_question, max_tokens=50)
    print("Ответ:", response3['choices'][0]['message']['content'])

    # 📌 Запрос 4: С STOP SEQUENCE
    print("ЗАПРОС 4: С условием завершения (инструкция: остановиться после перечисления)")
    message4 = base_question + """

    ВАЖНОЕ УСЛОВИЕ ЗАВЕРШЕНИЯ:
    Останови свой ответ сразу после того, как перечислишь три направления. 
    Не пиши "В заключение", "Таким образом" или любые другие завершающие фразы.
    Просто назови три направления и остановись.
    """
    response4 = chat(token, message4)
    print(response4['choices'][0]['message']['content'])