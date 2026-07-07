#!/usr/bin/env python3
"""
AI Challenge - Консольная утилита для работы с локальной Ollama
Интеграция с локальной LLM (qwen2.5-coder:1.5b)
Работает полностью без облачных моделей
"""

import requests
import json
import sys
import os
from datetime import datetime

# Конфигурация
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:1.5b"
HISTORY_FILE = "chat_history.json"


class LocalAIChat:
    def __init__(self):
        self.history = []
        self.load_history()
        self.check_ollama()

    def check_ollama(self):
        """Проверяет, доступна ли Ollama"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if MODEL_NAME not in model_names:
                    print(f"⚠️  Модель '{MODEL_NAME}' не найдена в Ollama")
                    print(f"📥 Установите её командой: ollama pull {MODEL_NAME}")
                    sys.exit(1)
                print(f"✅ Модель '{MODEL_NAME}' успешно загружена")
                return True
        except requests.exceptions.ConnectionError:
            print("❌ Ошибка: Ollama не запущена!")
            print("🔧 Запустите Ollama и попробуйте снова")
            sys.exit(1)
        return False

    def ask(self, prompt, stream=False):
        """
        Отправляет запрос к локальной LLM

        Args:
            prompt (str): Текст запроса
            stream (bool): Включить потоковую передачу

        Returns:
            str: Ответ модели
        """
        # Формируем контекст из истории
        context = self.build_context()
        full_prompt = f"{context}\nUser: {prompt}\nAssistant:"

        payload = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500
            }
        }

        try:
            if stream:
                return self.ask_stream(payload)
            else:
                response = requests.post(OLLAMA_URL, json=payload, timeout=60)
                response.raise_for_status()
                return response.json()['response']
        except requests.exceptions.Timeout:
            return "⏱️ Превышено время ожидания ответа от модели"
        except requests.exceptions.RequestException as e:
            return f"❌ Ошибка при запросе к Ollama: {str(e)}"

    def ask_stream(self, payload):
        """Потоковый режим для постепенного вывода ответа"""
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        full_response = ""
        print("\n🤖 ", end="", flush=True)

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        chunk = data['response']
                        print(chunk, end="", flush=True)
                        full_response += chunk
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue

        print()  # Переход на новую строку
        return full_response

    def build_context(self):
        """Строит контекст из последних сообщений"""
        if not self.history:
            return "You are a helpful AI assistant. Answer concisely and accurately."

        context = "Previous conversation:\n"
        for msg in self.history[-5:]:  # Берем последние 5 сообщений
            context += f"{msg['role']}: {msg['content']}\n"
        return context

    def add_to_history(self, role, content):
        """Добавляет сообщение в историю"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save_history()

    def save_history(self):
        """Сохраняет историю в файл"""
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Не удалось сохранить историю: {e}")

    def load_history(self):
        """Загружает историю из файла"""
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"⚠️ Не удалось загрузить историю: {e}")
            self.history = []

    def clear_history(self):
        """Очищает историю диалога"""
        self.history = []
        self.save_history()
        print("🧹 История диалога очищена")

    def show_history(self):
        """Показывает последние сообщения"""
        if not self.history:
            print("📭 История пуста")
            return

        print("\n📜 Последние сообщения:")
        for msg in self.history[-10:]:
            time = msg.get('timestamp', '')[:16]
            print(f"{time} | {msg['role']}: {msg['content'][:50]}...")


def print_help():
    """Выводит справку по командам"""
    print("""
📋 Команды:
  /help    - Показать эту справку
  /clear   - Очистить историю диалога
  /history - Показать последние сообщения
  /model   - Показать текущую модель
  /exit    - Выйти из программы

💡 Просто введите текст, чтобы задать вопрос модели
    """)


def main():
    print("=" * 50)
    print("🤖  Local AI Chat - Консольная утилита")
    print(f"📦 Модель: {MODEL_NAME}")
    print("🌐 Работает полностью локально (без облачных моделей)")
    print("=" * 50)

    chat = LocalAIChat()
    print_help()

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()

            if not user_input:
                continue

            # Обработка команд
            if user_input.startswith('/'):
                command = user_input.lower()

                if command == '/exit':
                    print("👋 До свидания!")
                    break
                elif command == '/help':
                    print_help()
                elif command == '/clear':
                    chat.clear_history()
                elif command == '/history':
                    chat.show_history()
                elif command == '/model':
                    print(f"🧠 Текущая модель: {MODEL_NAME}")
                    print(f"📊 Количество запросов в сессии: {len(chat.history) // 2}")
                else:
                    print(f"⚠️ Неизвестная команда: {command}")
                    print("   Введите /help для списка команд")
                continue

            # Добавляем вопрос в историю
            chat.add_to_history("user", user_input)

            # Получаем ответ от модели
            print("\n🤔 Думаю...", end=" ", flush=True)
            response = chat.ask(user_input, stream=False)
            print(f"\n🤖 {response}")  # <-- ДОБАВИТЬ ЭТУ СТРОКУ!

            # Добавляем ответ в историю
            chat.add_to_history("assistant", response)

        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()