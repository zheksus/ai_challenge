import requests
import uuid
import json


def get_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': 'KEY'
    }
    resp = requests.post(url, headers=headers, data={'scope': 'GIGACHAT_API_PERS'}, verify=False)
    return resp.json().get('access_token')


def chat(token, message, max_tokens=None):
    payload = {
        "model": 'GigaChat',
        "messages": [{"role": "user", "content": message}],
        "stream": False,
        "repetition_penalty": 1,
        "temperature": 0.3  # Низкая температура для логических задач
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

    resp = requests.post(
        'https://gigachat.devices.sberbank.ru/api/v1/chat/completions',
        headers={'Authorization': f'Bearer {token}'},
        json=payload,
        verify=False
    )
    return resp.json()


# Задача для решения
TASK = """У вас есть 3 ведра: 8 литров (полное), 5 литров (пустое) и 3 литра (пустое). 
Как отмерить ровно 6 литров воды? Опишите последовательность действий."""

if __name__ == '__main__':
    print("=" * 80)
    print("AI CHALLENGE: Сравнение 4 способов решения задачи")
    print("=" * 80)
    print(f"Задача: {TASK}\n")

    # Получаем токен
    token = get_token()

    # ============================================================
    # СПОСОБ 1: Прямой ответ без дополнительных инструкций
    # ============================================================
    print("\n" + "=" * 80)
    print("📌 СПОСОБ 1: Прямой ответ (без дополнительных инструкций)")
    print("=" * 80)

    response1 = chat(token, TASK)
    if 'choices' in response1:
        answer1 = response1['choices'][0]['message']['content']
        print(answer1)
    else:
        print(f"Ошибка: {response1}")

    # ============================================================
    # СПОСОБ 2: Добавлена инструкция "решай пошагово"
    # ============================================================
    print("\n" + "=" * 80)
    print("📌 СПОСОБ 2: С инструкцией «решай пошагово»")
    print("=" * 80)

    # prompt2 = TASK + "\n\nВАЖНО: Решай задачу пошагово, описывая каждое действие."

    prompt2 = TASK + """

    ВАЖНО:
    1. Решай задачу пошагово
    2. Максимум 10 шагов
    3. На каждом шаге показывай состояние всех трех ведер
    4. Как только получишь 6 литров в любом ведре — ОСТАНОВИСЬ и напиши "ГОТОВО"

    Формат шага:
    Шаг X: [действие] → (8л: X, 5л: Y, 3л: Z)"""

    response2 = chat(token, prompt2)

    if 'choices' in response2:
        answer2 = response2['choices'][0]['message']['content']
        print(answer2)
    else:
        print(f"Ошибка: {response2}")

    # ============================================================
    # СПОСОБ 3: LLM сама составляет промпт, потом решает
    # ============================================================
    print("\n" + "=" * 80)
    print("📌 СПОСОБ 3: Сначала составляем промпт, потом решаем")
    print("=" * 80)

    # Шаг 3.1: LLM составляет промпт для решения
    prompt_generator = f"""Ты — эксперт по составлению промптов для решения логических задач. 
Составь идеальный промпт (инструкцию) для решения следующей задачи:

{TASK}

Промпт должен быть подробным, пошаговым, помогать LLM правильно решить задачу.
Выведи ТОЛЬКО промпт, без лишних комментариев."""

    print("🔨 Генерируем промпт...")
    response_gen = chat(token, prompt_generator)
    if 'choices' in response_gen:
        generated_prompt = response_gen['choices'][0]['message']['content']
        print(f"✅ Сгенерированный промпт:\n{generated_prompt}\n")
    else:
        generated_prompt = TASK + "\n\nРеши задачу пошагово, объясняя каждое действие."
        print(f"⚠️ Используем стандартный промпт\n")

    # Шаг 3.2: Решаем задачу с помощью сгенерированного промпта
    print("💡 Решаем задачу с использованием сгенерированного промпта...")
    response3 = chat(token, generated_prompt)
    if 'choices' in response3:
        answer3 = response3['choices'][0]['message']['content']
        print(answer3)
    else:
        print(f"Ошибка: {response3}")

    # ============================================================
    # СПОСОБ 4: Группа экспертов (аналитик, инженер, критик)
    # ============================================================
    print("\n" + "=" * 80)
    print("📌 СПОСОБ 4: Группа экспертов (аналитик, инженер, критик)")
    print("=" * 80)

    prompt4 = f"""Реши следующую задачу, но представь, что работают три эксперта:

{TASK}

Формат ответа:

[АНАЛИТИК]
Анализирую задачу: (опиши условие, известные данные, цель, какие операции возможны)

[ИНЖЕНЕР]
Предлагаю алгоритм решения: (пошаговая последовательность действий, показывающая, как получить 6 литров)

[КРИТИК]
Проверяю решение: (найди ошибки, неточности, проверь каждый шаг)
Если есть ошибки, исправь их и предложи правильное решение.

[ИТОГОВОЕ РЕШЕНИЕ]
(конечный ответ после обсуждения всех экспертов)"""

    response4 = chat(token, prompt4)
    if 'choices' in response4:
        answer4 = response4['choices'][0]['message']['content']
        print(answer4)
    else:
        print(f"Ошибка: {response4}")

    # ============================================================
    # Сохраняем результаты для сравнения
    # ============================================================
    results = {
        "task": TASK,
        "direct_answer": answer1 if 'answer1' in dir() else None,
        "step_by_step": answer2 if 'answer2' in dir() else None,
        "generated_prompt": {
            "prompt": generated_prompt if 'generated_prompt' in dir() else None,
            "answer": answer3 if 'answer3' in dir() else None
        },
        "experts": answer4 if 'answer4' in dir() else None
    }

    with open("gigachat_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("✅ Результаты сохранены в gigachat_results.json")
    print("=" * 80)