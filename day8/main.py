import requests
import uuid
import json
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class TokenCounter:
    """
    Класс для подсчёта токенов.
    Для GigaChat используем приблизительную оценку:
    - Русский текст: ~1.8 символа на токен
    - Латиница/цифры: ~4 символа на токен
    - Среднее: ~2 символа на токен
    """

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Подсчёт количества токенов в тексте.
        Возвращает приблизительное количество токенов.
        """
        if not text:
            return 0

        # Приблизительная оценка для смешанного текста
        # По оценкам OpenAI: 1 токен ≈ 4 символа для английского, 1-2 для русского
        # Используем среднее 1.8 символа на токен для русского текста
        return len(text) // 2  # Упрощённая формула для демонстрации

    @staticmethod
    def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
        """Подсчёт токенов во всех сообщениях"""
        total = 0
        for msg in messages:
            total += TokenCounter.count_tokens(msg.get("content", ""))
        return total

    @staticmethod
    def format_tokens(tokens: int) -> str:
        """Форматирование числа токенов"""
        if tokens < 1000:
            return f"{tokens} токенов"
        else:
            return f"{tokens / 1000:.1f}K токенов"


class GigaChatAPIError(Exception):
    """Исключение для ошибок API GigaChat"""
    pass


class PersistentGigaChatAgentWithTokens:
    """
    Агент с подсчётом токенов и контролем лимитов.
    """

    # Лимиты модели GigaChat
    MAX_TOKENS_PER_REQUEST = 8000000  # Максимум токенов на запрос (контекст + ответ)
    MAX_RESPONSE_TOKENS = 4000000  # Максимум токенов в ответе
    TOKEN_WARNING_THRESHOLD = 6000000  # Предупреждение при превышении

    def __init__(self, auth_key: str, model: str = "GigaChat",
                 db_path: str = "agent_history.db", session_id: str = "default"):
        """
        Инициализация агента с токен-счётчиком.
        """
        self.auth_key = auth_key
        self.model = model
        self.db_path = db_path
        self.session_id = session_id
        self._token = None
        self._token_expires_at = None
        self.token_counter = TokenCounter()

        # Статистика по токенам
        self.total_input_tokens = 0  # Всего токенов в запросах (история)
        self.total_output_tokens = 0  # Всего токенов в ответах
        self.request_count = 0  # Количество запросов

        # Инициализация базы данных
        self._init_database()

        # Загрузка истории из базы
        self.history = self._load_history()

        # Восстановление статистики из базы
        self._load_stats()

    def _init_database(self):
        """Создание таблиц базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Таблица для сессий
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                model TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')

        # Таблица для истории сообщений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                tokens INTEGER,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')

        # Таблица для статистики токенов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS token_stats (
                session_id TEXT PRIMARY KEY,
                total_input_tokens INTEGER DEFAULT 0,
                total_output_tokens INTEGER DEFAULT 0,
                request_count INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')

        conn.commit()
        conn.close()

        self._update_session_info()

    def _update_session_info(self):
        """Обновление информации о сессии"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute('''
            INSERT OR REPLACE INTO sessions (session_id, model, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM sessions WHERE session_id = ?), ?), ?)
        ''', (self.session_id, self.model, self.session_id, now, now))

        # Инициализация статистики, если её нет
        cursor.execute('''
            INSERT OR IGNORE INTO token_stats (session_id, total_input_tokens, total_output_tokens, request_count)
            VALUES (?, 0, 0, 0)
        ''', (self.session_id,))

        conn.commit()
        conn.close()

    def _load_stats(self):
        """Загрузка статистики токенов из базы"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT total_input_tokens, total_output_tokens, request_count
            FROM token_stats
            WHERE session_id = ?
        ''', (self.session_id,))

        row = cursor.fetchone()
        if row:
            self.total_input_tokens = row[0]
            self.total_output_tokens = row[1]
            self.request_count = row[2]

        conn.close()

    def _save_stats(self):
        """Сохранение статистики токенов в базу"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE token_stats
            SET total_input_tokens = ?, total_output_tokens = ?, request_count = ?
            WHERE session_id = ?
        ''', (self.total_input_tokens, self.total_output_tokens, self.request_count, self.session_id))

        conn.commit()
        conn.close()

    def _load_history(self) -> List[Dict[str, str]]:
        """Загрузка истории из базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT role, content, tokens FROM messages 
            WHERE session_id = ? 
            ORDER BY id ASC
        ''', (self.session_id,))

        rows = cursor.fetchall()
        conn.close()

        history = [{"role": row[0], "content": row[1]} for row in rows]

        if history:
            print(f"📂 Загружено {len(history)} сообщений из истории")

        return history

    def _save_message(self, role: str, content: str, tokens: int):
        """Сохранение сообщения в базу"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO messages (session_id, role, content, tokens, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (self.session_id, role, content, tokens, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def _get_token(self) -> str:
        """Получение или обновление токена доступа"""
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
            from datetime import timedelta
            self._token_expires_at = datetime.now() + timedelta(minutes=25)
            return self._token
        except Exception as e:
            print(f"❌ Ошибка авторизации: {e}")
            raise

    def _check_token_limit(self) -> Tuple[bool, int, str]:
        """
        Проверка, не превышает ли история лимит токенов.
        Возвращает: (превышен_ли_лимит, количество_токенов, сообщение)
        """
        history_tokens = TokenCounter.count_messages_tokens(self.history)

        if history_tokens > self.MAX_TOKENS_PER_REQUEST:
            return (True, history_tokens,
                    f"⚠️ ПРЕДУПРЕЖДЕНИЕ: История ({history_tokens} токенов) превышает лимит модели ({self.MAX_TOKENS_PER_REQUEST} токенов)!")
        elif history_tokens > self.TOKEN_WARNING_THRESHOLD:
            return (False, history_tokens,
                    f"⚠️ ВНИМАНИЕ: История близка к лимиту ({history_tokens}/{self.MAX_TOKENS_PER_REQUEST} токенов)")
        else:
            return (False, history_tokens,
                    f"✅ История в пределах лимита ({history_tokens}/{self.MAX_TOKENS_PER_REQUEST} токенов)")

    def _call_api(self, message: str, temperature: float = 0.7) -> Dict:
        """Внутренний метод для вызова API"""
        token = self._get_token()

        # Формируем сообщения из истории + новое сообщение
        messages = self.history + [{"role": "user", "content": message}]

        # Подсчёт токенов в запросе
        request_tokens = TokenCounter.count_messages_tokens(messages)

        # Проверка лимита
        is_over, token_count, warning_msg = self._check_token_limit()
        if is_over:
            print(warning_msg)
            raise GigaChatAPIError(
                f"Превышение лимита токенов: история ({token_count}) > {self.MAX_TOKENS_PER_REQUEST}. Очистите историю командой /clear")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": self.MAX_RESPONSE_TOKENS,
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
        Возвращает ответ с информацией о токенах.
        """
        # Подсчёт токенов вопроса
        question_tokens = TokenCounter.count_tokens(question)

        # Добавляем вопрос в историю
        self.history.append({"role": "user", "content": question})
        self._save_message("user", question, question_tokens)
        self.total_input_tokens += question_tokens
        self.request_count += 1

        # Вызываем API
        try:
            response = self._call_api(question, temperature)
        except GigaChatAPIError as e:
            return f"[ОШИБКА] {str(e)}"

        if "error" in response:
            return f"[Ошибка API] {response['error']}"

        if "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["message"]["content"]
            answer_tokens = TokenCounter.count_tokens(answer)

            # Добавляем ответ в историю
            self.history.append({"role": "assistant", "content": answer})
            self._save_message("assistant", answer, answer_tokens)
            self.total_output_tokens += answer_tokens
            self._save_stats()

            # Формируем ответ с информацией о токенах
            total_tokens = question_tokens + answer_tokens
            history_tokens = TokenCounter.count_messages_tokens(self.history)

            token_info = f"""
📊 Статистика токенов:
├─ Вопрос: {question_tokens} токенов
├─ Ответ: {answer_tokens} токенов
├─ Запрос: {total_tokens} токенов
├─ История: {history_tokens} токенов
├─ Всего в диалоге: {self.total_input_tokens + self.total_output_tokens} токенов
└─ Запросов: {self.request_count}

📈 Прогноз стоимости (по тарифу GigaChat 2, ~0.5₽ за 1K токенов):
├─ Сессия: ~{(self.total_input_tokens + self.total_output_tokens) * 0.0005:.2f}₽
└─ Этот запрос: ~{total_tokens * 0.0005:.4f}₽
"""

            # Проверка лимита для следующего запроса
            _, _, warning = self._check_token_limit()
            if "ПРЕДУПРЕЖДЕНИЕ" in warning or "ВНИМАНИЕ" in warning:
                token_info += f"\n{warning}"

            return answer + token_info

        return "[Ошибка] Неожиданный формат ответа от API"

    def clear_history(self):
        """Очистка истории диалога и сброс статистики"""
        self.history = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
        cursor.execute("DELETE FROM token_stats WHERE session_id = ?", (self.session_id,))
        cursor.execute(
            "INSERT OR IGNORE INTO token_stats (session_id, total_input_tokens, total_output_tokens, request_count) VALUES (?, 0, 0, 0)",
            (self.session_id,))
        conn.commit()
        conn.close()

        self._update_session_info()
        print("🧹 История диалога и статистика очищены")

    def get_stats(self) -> Dict:
        """Получение полной статистики"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "request_count": self.request_count,
            "history_tokens": TokenCounter.count_messages_tokens(self.history),
            "history_length": len(self.history)
        }

    def get_history(self) -> List[Dict[str, str]]:
        """Получение истории диалога"""
        return self.history.copy()

    def set_model(self, model: str):
        """Смена модели"""
        self.model = model
        self._update_session_info()
        print(f"🔄 Модель изменена на: {model}")


class PersistentCLIWithTokens:
    """CLI интерфейс для агента с подсчётом токенов"""

    def __init__(self, agent: PersistentGigaChatAgentWithTokens):
        self.agent = agent
        self.running = True

    def _print_help(self):
        print("\n" + "=" * 70)
        print("📖 ДОСТУПНЫЕ КОМАНДЫ:")
        print("=" * 70)
        print("  /help         - показать эту справку")
        print("  /clear        - очистить историю диалога и статистику")
        print("  /history      - показать историю диалога")
        print("  /stats        - показать детальную статистику токенов")
        print("  /model        - показать текущую модель")
        print("  /model pro    - переключить на GigaChat-Pro")
        print("  /model max    - переключить на GigaChat-Max")
        print("  /limit        - показать лимиты модели")
        print("  /big          - создать длинное сообщение для теста лимитов")
        print("  /exit         - выйти из чата")
        print("=" * 70 + "\n")

    def _print_stats(self):
        """Вывод детальной статистики"""
        stats = self.agent.get_stats()

        print("\n" + "=" * 70)
        print("📊 ДЕТАЛЬНАЯ СТАТИСТИКА ТОКЕНОВ")
        print("=" * 70)
        print(
            f"📥 Входные токены (история):     {stats['total_input_tokens']} ({stats['total_input_tokens'] / 1000:.2f}K)")
        print(
            f"📤 Выходные токены (ответы):     {stats['total_output_tokens']} ({stats['total_output_tokens'] / 1000:.2f}K)")
        print(f"📊 Всего токенов:                {stats['total_tokens']} ({stats['total_tokens'] / 1000:.2f}K)")
        print(f"🔢 Количество запросов:          {stats['request_count']}")
        print(f"💬 Токенов в текущей истории:    {stats['history_tokens']} ({stats['history_tokens'] / 1000:.2f}K)")
        print(f"📝 Сообщений в истории:          {stats['history_length']}")

        # Примерная стоимость
        cost_per_1k = 0.0005  # 0.5₽ за 1K токенов
        session_cost = stats['total_tokens'] * cost_per_1k

        print("\n💰 ОЦЕНКА СТОИМОСТИ (по тарифу GigaChat 2):")
        print(f"   Стоимость сессии: ~{session_cost:.4f}₽")
        print(f"   Стоимость за 1K токенов: ~{cost_per_1k * 1000:.2f}₽")
        print("=" * 70 + "\n")

    def _print_limit_info(self):
        """Вывод информации о лимитах модели"""
        print("\n" + "=" * 70)
        print("⚠️ ЛИМИТЫ МОДЕЛИ GigaChat")
        print("=" * 70)
        print(f"📌 Максимум токенов на запрос:  {PersistentGigaChatAgentWithTokens.MAX_TOKENS_PER_REQUEST}")
        print(f"📌 Максимум токенов в ответе:   {PersistentGigaChatAgentWithTokens.MAX_RESPONSE_TOKENS}")
        print(f"📌 Порог предупреждения:        {PersistentGigaChatAgentWithTokens.TOKEN_WARNING_THRESHOLD}")

        current_tokens = TokenCounter.count_messages_tokens(self.agent.history)
        print(f"\n📊 Текущее состояние истории: {current_tokens} токенов")

        if current_tokens > PersistentGigaChatAgentWithTokens.TOKEN_WARNING_THRESHOLD:
            print("⚠️ ВНИМАНИЕ: История близка к лимиту!")
        print("=" * 70 + "\n")

    def _print_history_with_tokens(self):
        """Вывод истории с подсчётом токенов для каждого сообщения"""
        history = self.agent.get_history()
        if not history:
            print("📭 История пуста")
            return

        print("\n" + "=" * 70)
        print(f"📜 ИСТОРИЯ ДИАЛОГА (сессия: {self.agent.session_id})")
        print("=" * 70)

        total_tokens = 0
        for i, msg in enumerate(history):
            role = "👤 ПОЛЬЗОВАТЕЛЬ" if msg["role"] == "user" else "🤖 АГЕНТ"
            tokens = TokenCounter.count_tokens(msg["content"])
            total_tokens += tokens

            print(f"\n[{i + 1}] {role} ({tokens} токенов):")
            content_preview = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            print(f"    {content_preview}")

        print(f"\n📊 Итого в истории: {total_tokens} токенов")
        print("=" * 70 + "\n")

    def _generate_big_message(self, file_path="./book.txt") -> str:
        """Генерация длинного сообщения для тестирования лимитов"""
        base_text = "Это тестовое сообщение для проверки лимитов модели. " * 500
        return base_text[:6000]  # Ограничим 6000 символов
        # Проверяем существование файла
        # if not os.path.exists(file_path):
        #     # Пробуем найти файл в текущей директории и родительских
        #     for root, dirs, files in os.walk('.'):
        #         if 'skills-all.md' in files:
        #             file_path = os.path.join(root, 'skills-all.md')
        #             break
        #     else:
        #         raise FileNotFoundError(f"Файл {file_path} не найден!")
        #
        # # Читаем файл
        # with open(file_path, 'r', encoding='utf-8') as f:
        #     content = f.read()
        #
        # # Подсчитываем примерное количество токенов
        # estimated_tokens = len(content) // 2
        #
        # # Формируем сообщение для отправки
        # message = f"""Пожалуйста, проанализируй содержимое этого файла.
        #
        # Содержимое файла (примерно {estimated_tokens} токенов):
        # ---
        # {content}
        # ---
        #
        # Вопрос: Какие основные темы освещены в этом файле? Дай краткий ответ (2-3 предложения)."""
        # return message

    def run(self):
        """Запуск CLI интерфейса"""
        print("\n" + "=" * 70)
        print("🤖 ДОБРО ПОЖАЛОВАТЬ В GIGACHAT АГЕНТ С ПОДСЧЁТОМ ТОКЕНОВ")
        print("=" * 70)
        print(f"📌 Модель: {self.agent.model}")
        print(f"📌 Сессия: {self.agent.session_id}")
        print(f"📌 История: {len(self.agent.get_history())} сообщений в памяти")
        print(f"📌 Токенов в истории: {TokenCounter.count_messages_tokens(self.agent.history)}")
        print("📌 Введите вопрос или команду ( /help для справки )")
        print("=" * 70 + "\n")

        while self.running:
            try:
                user_input = input("👤 Вы: ").strip()

                if not user_input:
                    continue

                # Обработка команд
                if user_input.startswith("/"):
                    cmd = user_input.lower()

                    if cmd == "/exit":
                        print("\n👋 До свидания! Итоговая статистика:")
                        self._print_stats()
                        self.running = False
                        break

                    elif cmd == "/help":
                        self._print_help()

                    elif cmd == "/clear":
                        self.agent.clear_history()

                    elif cmd == "/history":
                        self._print_history_with_tokens()

                    elif cmd == "/stats":
                        self._print_stats()

                    elif cmd == "/limit":
                        self._print_limit_info()

                    elif cmd == "/model":
                        print(f"📌 Текущая модель: {self.agent.model}")

                    elif cmd.startswith("/model pro"):
                        self.agent.set_model("GigaChat-Pro")
                        print("🔄 Для применения смены модели очистите историю (/clear)")

                    elif cmd.startswith("/model max"):
                        self.agent.set_model("GigaChat-Max")
                        print("🔄 Для применения смены модели очистите историю (/clear)")

                    elif cmd == "/big":
                        big_msg = self._generate_big_message()
                        print(
                            f"📦 Сгенерировано длинное сообщение ({len(big_msg)} символов, ~{len(big_msg) // 2} токенов)")
                        print("🤖 Агент: ", end="", flush=True)
                        answer = self.agent.ask(big_msg)
                        print(answer)

                    else:
                        print(f"❌ Неизвестная команда: {user_input}")
                        print("   Введите /help для списка команд")

                # Обычный вопрос
                else:
                    print("🤖 Агент: ", end="", flush=True)
                    answer = self.agent.ask(user_input)
                    print(answer)
                    print()

            except KeyboardInterrupt:
                print("\n\n👋 Прервано пользователем. Итоговая статистика:")
                self._print_stats()
                self.running = False
                break
            except Exception as e:
                print(f"\n❌ Неожиданная ошибка: {e}")
                print("   Попробуйте ещё раз или введите /exit")


def test_dialogs():
    """Функция для тестирования диалогов с разной длиной"""
    print("\n" + "=" * 70)
    print("🧪 ЗАПУСК ТЕСТОВЫХ ДИАЛОГОВ")
    print("=" * 70)

    AUTH_KEY = "Basic "

    # Тест 1: Короткий диалог
    print("\n📌 ТЕСТ 1: Короткий диалог (3 сообщения)")
    print("-" * 50)
    agent = PersistentGigaChatAgentWithTokens(AUTH_KEY, db_path="test.db", session_id="test_short")
    agent.clear_history()

    questions = ["Привет!", "Как тебя зовут?", "Что ты умеешь?"]
    for q in questions:
        print(f"\n👤 {q}")
        answer = agent.ask(q)
        print(f"🤖 {answer[:200]}...")

    print(f"\n📊 ИТОГО: {agent.get_stats()['total_tokens']} токенов за {agent.get_stats()['request_count']} запроса")

    # Тест 2: Длинный диалог
    print("\n" + "=" * 70)
    print("\n📌 ТЕСТ 2: Длинный диалог (10 сообщений)")
    print("-" * 50)
    agent = PersistentGigaChatAgentWithTokens(AUTH_KEY, db_path="test.db", session_id="test_long")
    agent.clear_history()

    for i in range(1, 11):
        q = f"Расскажи интересный факт номер {i} о космосе. Ответь кратко, в одном предложении."
        print(f"\n👤 {q}")
        answer = agent.ask(q)
        # Показываем только статистику, а не полный ответ
        stats = agent.get_stats()
        print(f"📊 Токенов: {stats['total_tokens']}, история: {stats['history_tokens']}")

    print(f"\n📊 ИТОГО: {agent.get_stats()['total_tokens']} токенов за {agent.get_stats()['request_count']} запросов")

    # Тест 3: Превышение лимита (искусственное)
    print("\n" + "=" * 70)
    print("\n📌 ТЕСТ 3: Проверка превышения лимита")
    print("-" * 50)
    agent = PersistentGigaChatAgentWithTokens(AUTH_KEY, db_path="test.db", session_id="test_overflow")
    agent.clear_history()

    # Добавляем очень длинное сообщение
    long_text = "Это очень длинное сообщение. " * 2000  # ~4000 токенов
    print(f"👤 Отправка длинного сообщения (~{len(long_text) // 2} токенов)")
    answer = agent.ask(long_text)
    print(f"🤖 {answer[:300]}...")

    # Пытаемся отправить ещё одно
    print("\n👤 Ещё один короткий вопрос")
    answer = agent.ask("Как дела?")
    print(f"🤖 {answer[:200]}...")


def main():
    """Главная функция"""
    AUTH_KEY = "Basic <ваш_base64_ключ>"  # Замените на ваш ключ

    if AUTH_KEY == "Basic <ваш_base64_ключ>":
        print("⚠️ ВНИМАНИЕ: Не указан ключ авторизации!")
        print("   Отредактируйте файл и добавьте ваш ключ в AUTH_KEY")
        print("   или установите переменную окружения GIGACHAT_AUTH_KEY")

        # Запрос на запуск тестов
        run_tests = input("\nЗапустить тестовые диалоги без ключа? (y/N): ").lower()
        if run_tests == 'y':
            test_dialogs()
        return

    # Создаём агента
    agent = PersistentGigaChatAgentWithTokens(
        auth_key=AUTH_KEY,
        model="GigaChat",
        db_path="agent_history.db",
        session_id="main"
    )

    # Запускаем CLI
    cli = PersistentCLIWithTokens(agent)
    cli.run()


if __name__ == "__main__":
    main()