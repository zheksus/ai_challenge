import requests
import uuid
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class GigaChatAgent:
    """
    Агент для работы с GigaChat API.
    Инкапсулирует логику авторизации, отправки запросов, управления историей.
    """

    def __init__(self, auth_key: str, model: str = "GigaChat", max_history: int = 10):
        """
        Инициализация агента.

        Args:
            auth_key: Ключ авторизации (Basic ...)
            model: Модель GigaChat (GigaChat, GigaChat-Pro, GigaChat-Max)
            max_history: Максимальное количество сообщений в истории
        """
        self.auth_key = auth_key
        self.model = model
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
        self._token = None
        self._token_expires_at = None

    def _get_token(self) -> str:
        """
        Получение или обновление токена доступа.
        Токен кэшируется и обновляется автоматически при истечении.
        """
        # Проверяем, не истёк ли текущий токен
        if self._token and self._token_expires_at and datetime.now() < self._token_expires_at:
            return self._token

        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': str(uuid.uuid4()),
            'Authorization': self.auth_key
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                data={'scope': 'GIGACHAT_API_PERS'},
                verify=False,
                timeout=30
            )
            data = resp.json()
            self._token = data.get('access_token')
            # Токен живёт 30 минут, установим на 25 минут для запаса
            from datetime import timedelta
            self._token_expires_at = datetime.now() + timedelta(minutes=25)
            return self._token
        except Exception as e:
            print(f"❌ Ошибка авторизации: {e}")
            raise

    def _call_api(self, message: str, temperature: float = 0.7, max_tokens: int = 2000) -> Dict:
        """
        Внутренний метод для вызова API.

        Args:
            message: Текст сообщения
            temperature: Креативность ответа (0-1)
            max_tokens: Максимальная длина ответа
        """
        token = self._get_token()

        # Формируем сообщения из истории + новое сообщение
        messages = self.history + [{"role": "user", "content": message}]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": 1
        }

        try:
            resp = requests.post(
                'https://gigachat.devices.sberbank.ru/api/v1/chat/completions',
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {token}'
                },
                json=payload,
                verify=False,
                timeout=60
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def ask(self, question: str, temperature: float = 0.7) -> str:
        """
        Основной метод агента: задать вопрос и получить ответ.

        Args:
            question: Вопрос пользователя
            temperature: Креативность ответа

        Returns:
            Ответ агента
        """
        # Добавляем вопрос в историю
        self.history.append({"role": "user", "content": question})

        # Ограничиваем историю
        if len(self.history) > self.max_history * 2:  # user + assistant
            self.history = self.history[-self.max_history * 2:]

        # Вызываем API
        response = self._call_api(question, temperature)

        if "error" in response:
            return f"[Ошибка] {response['error']}"

        if "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["message"]["content"]
            # Добавляем ответ в историю
            self.history.append({"role": "assistant", "content": answer})
            return answer

        return "[Ошибка] Неожиданный формат ответа от API"

    def clear_history(self):
        """Очистка истории диалога"""
        self.history = []
        print("🧹 История диалога очищена")

    def get_history(self) -> List[Dict[str, str]]:
        """Получение истории диалога"""
        return self.history.copy()

    def set_model(self, model: str):
        """Смена модели"""
        self.model = model
        print(f"🔄 Модель изменена на: {model}")

    def set_temperature(self, temp: float):
        """Установка температуры"""
        print(f"🌡️ Температура изменена на: {temp}")
        # Температура передаётся в _call_api, здесь просто уведомление


class SimpleCLI:
    """
    Простой CLI интерфейс для агента.
    """

    def __init__(self, agent: GigaChatAgent):
        self.agent = agent
        self.running = True

    def _print_help(self):
        """Вывод справки"""
        print("\n" + "=" * 60)
        print("📖 ДОСТУПНЫЕ КОМАНДЫ:")
        print("=" * 60)
        print("  /help      - показать эту справку")
        print("  /clear     - очистить историю диалога")
        print("  /history   - показать историю диалога")
        print("  /model     - показать текущую модель")
        print("  /model pro - переключить на GigaChat-Pro")
        print("  /model max - переключить на GigaChat-Max")
        print("  /temp X    - установить температуру (0-1.5)")
        print("  /exit      - выйти из чата")
        print("=" * 60 + "\n")

    def _print_history(self):
        """Вывод истории"""
        history = self.agent.get_history()
        if not history:
            print("📭 История пуста")
            return

        print("\n" + "=" * 60)
        print("📜 ИСТОРИЯ ДИАЛОГА:")
        print("=" * 60)
        for i, msg in enumerate(history):
            role = "👤 ПОЛЬЗОВАТЕЛЬ" if msg["role"] == "user" else "🤖 АГЕНТ"
            print(f"\n[{i + 1}] {role}:")
            print(f"    {msg['content'][:200]}..." if len(msg['content']) > 200 else f"    {msg['content']}")
        print("=" * 60 + "\n")

    def run(self):
        """Запуск CLI интерфейса"""
        print("\n" + "=" * 60)
        print("🤖 ДОБРО ПОЖАЛОВАТЬ В GigaChat АГЕНТ")
        print("=" * 60)
        print(f"📌 Модель: {self.agent.model}")
        print("📌 Введите вопрос или команду ( /help для справки )")
        print("=" * 60 + "\n")

        while self.running:
            try:
                # Получаем ввод пользователя
                user_input = input("👤 Вы: ").strip()

                if not user_input:
                    continue

                # Обработка команд
                if user_input.startswith("/"):
                    cmd = user_input.lower()

                    if cmd == "/exit":
                        print("👋 До свидания!")
                        self.running = False
                        break

                    elif cmd == "/help":
                        self._print_help()

                    elif cmd == "/clear":
                        self.agent.clear_history()

                    elif cmd == "/history":
                        self._print_history()

                    elif cmd == "/model":
                        print(f"📌 Текущая модель: {self.agent.model}")

                    elif cmd.startswith("/model pro"):
                        self.agent.set_model("GigaChat-Pro")

                    elif cmd.startswith("/model max"):
                        self.agent.set_model("GigaChat-Max")

                    elif cmd.startswith("/temp"):
                        parts = cmd.split()
                        if len(parts) > 1:
                            try:
                                temp = float(parts[1])
                                if 0 <= temp <= 1.5:
                                    # Передаём температуру через атрибут
                                    self.agent.current_temp = temp
                                    print(f"🌡️ Температура установлена: {temp}")
                                else:
                                    print("❌ Температура должна быть от 0 до 1.5")
                            except ValueError:
                                print("❌ Используйте число, например: /temp 0.7")
                        else:
                            print("📌 Используйте: /temp 0.7")

                    else:
                        print(f"❌ Неизвестная команда: {user_input}")
                        print("   Введите /help для списка команд")

                # Обычный вопрос
                else:
                    print("🤖 Агент: ", end="", flush=True)

                    # Получаем температуру, если установлена
                    temp = getattr(self.agent, 'current_temp', 0.7)

                    answer = self.agent.ask(user_input, temperature=temp)
                    print(answer)
                    print()  # Пустая строка после ответа

            except KeyboardInterrupt:
                print("\n\n👋 Прервано пользователем. До свидания!")
                self.running = False
                break
            except Exception as e:
                print(f"\n❌ Неожиданная ошибка: {e}")
                print("   Попробуйте ещё раз или введите /exit")


def main():
    """Главная функция"""
    # Конфигурация
    AUTH_KEY = "Basic <ваш_base64_ключ>"  # Замените на ваш ключ

    if AUTH_KEY == "Basic <ваш_base64_ключ>":
        print("⚠️ ВНИМАНИЕ: Не указан ключ авторизации!")
        print("   Отредактируйте файл и добавьте ваш ключ в AUTH_KEY")
        print("   Формат: Basic base64(ClientID:Secret)")
        return

    # Создаём агента
    agent = GigaChatAgent(
        auth_key=AUTH_KEY,
        model="GigaChat",
        max_history=20
    )

    # Запускаем CLI
    cli = SimpleCLI(agent)
    cli.run()


if __name__ == "__main__":
    main()