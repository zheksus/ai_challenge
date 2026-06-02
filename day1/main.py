import requests
import uuid

def get_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    rq_uid = str(uuid.uuid4())

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': f'{rq_uid}',
        'Authorization': 'Basic <auth-key>'
    }
    resp = requests.post(
        url,
        headers=headers,
        data={'scope': 'GIGACHAT_API_PERS'},
        verify=False,
    )
    data = resp.json()
    return data.get('access_token')

def chat(token, message):
    resp = requests.post(
        'https://gigachat.devices.sberbank.ru/api/v1/chat/completions',
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        },
        json={
            "model": 'GigaChat',
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            "repetition_penalty": 1
        },
        verify=False
    )
    return resp.json()

if __name__ == '__main__':
    token = get_token()
    response = chat(token, "Привет")
    print(response['choices'][0]['message']['content'])
