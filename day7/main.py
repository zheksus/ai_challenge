import requests
import uuid
import json
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class PersistentGigaChatAgent:
    """
    Агент с сохранением истории в SQLite.
    При перезапуске восстанавливает историю диалога.
    """

    def __init__(self, auth_key: str, model: str = "GigaChat",
                 db_path: str = "agent_history.db", session_id: str = "default"):
        """
        Инициализация агента с постоянным хранилищем.

        Args:
            auth_key: Ключ авторизации (Basic ...)
            model: Модель GigaChat
            db_path: Путь к файлу базы данных SQLite
            session_id: Идентификатор сессии (позволяет иметь разные диалоги)
        """
        self.auth_key = auth_key
        self.model = model
        self.db_path = db_path
        self.session_id = session_id
        self._token = None
        self._token_expires_at = None

        # Инициализация базы данных
        self._init_database()

        # Загрузка истории из базы
        self.history = self._load_history()

    def _init_database(self):
        """Создание таблиц базы данных, если их нет"""
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
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')

        conn.commit()
        conn.close()

        # Обновляем информацию о сессии
        self._update_session_info()

    def _update_session_info(self):
        """Обновление информации о сессии в базе"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute('''
            INSERT OR REPLACE INTO sessions (session_id, model, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM sessions WHERE session_id = ?), ?), ?)
        ''', (self.session_id, self.model, self.session_id, now, now))

        conn.commit()
        conn.close()

    def _load_history(self) -> List[Dict[str, str]]:
        """Загрузка истории из базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT role, content FROM messages 
            WHERE session_id = ? 
            ORDER BY id ASC
        ''', (self.session_id,))

        rows = cursor.fetchall()
        conn.close()

        history = [{"role": row[0], "content": row[1]} for row in rows]

        if history:
            print(f"📂 Загружено {len(history)} сообщений из истории")

        return history

    def _save_message(self, role: str, content: str):
        """Сохранение одного сообщения в базу"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (self.session_id, role, content, datetime.now().isoformat()))

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

    def _call_api(self, message: str, temperature: float = 0.7, max_tokens: int = 2000) -> Dict:
        """Внутренний метод для вызова API"""
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
        """
        # Добавляем вопрос в историю
        self.history.append({"role": "user", "content": question})
        self._save_message("user", question)

        # Вызываем API
        response = self._call_api(question, temperature)

        if "error" in response:
            return f"[Ошибка] {response['error']}"

        if "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["message"]["content"]
            # Добавляем ответ в историю
            self.history.append({"role": "assistant", "content": answer})
            self._save_message("assistant", answer)
            return answer

        return "[Ошибка] Неожиданный формат ответа от API"

    def clear_history(self):
        """Очистка истории диалога (и в памяти, и в базе)"""
        self.history = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
        conn.commit()
        conn.close()

        self._update_session_info()
        print("🧹 История диалога очищена")

    def get_history(self) -> List[Dict[str, str]]:
        """Получение истории диалога"""
        return self.history.copy()

    def set_model(self, model: str):
        """Смена модели"""
        self.model = model
        self._update_session_info()
        print(f"🔄 Модель изменена на: {model}")

    def switch_session(self, session_id: str):
        """
        Переключение на другую сессию (другой диалог).
        Позволяет вести несколько независимых разговоров.
        """
        self.session_id = session_id
        self.history = self._load_history()
        self._update_session_info()
        print(f"🔄 Переключено на сессию: {session_id}")
        print(f"📂 Загружено {len(self.history)} сообщений")

    def list_sessions(self) -> List[Tuple[str, str, int]]:
        """Список всех сохранённых сессий"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT s.session_id, s.updated_at, COUNT(m.id) as msg_count
            FROM sessions s
            LEFT JOIN messages m ON s.session_id = m.session_id
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
        ''')

        sessions = cursor.fetchall()
        conn.close()
        return sessions


class PersistentCLI:
    """
    CLI интерфейс для агента с постоянной памятью.
    """

    def __init__(self, agent: PersistentGigaChatAgent):
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
        print("  /sessions  - показать список всех сессий")
        print("  /switch X  - переключиться на сессию X")
        print("  /new       - создать новую сессию")
        print("  /exit      - выйти из чата")
        print("=" * 60 + "\n")

    def _print_history(self):
        """Вывод истории"""
        history = self.agent.get_history()
        if not history:
            print("📭 История пуста")
            return

        print("\n" + "=" * 60)
        print(f"📜 ИСТОРИЯ ДИАЛОГА (сессия: {self.agent.session_id})")
        print("=" * 60)
        for i, msg in enumerate(history):
            role = "👤 ПОЛЬЗОВАТЕЛЬ" if msg["role"] == "user" else "🤖 АГЕНТ"
            print(f"\n[{i + 1}] {role}:")
            print(f"    {msg['content'][:300]}..." if len(msg['content']) > 300 else f"    {msg['content']}")
        print("=" * 60 + "\n")

    def _print_sessions(self):
        """Вывод списка сессий"""
        sessions = self.agent.list_sessions()
        if not sessions:
            print("📭 Нет сохранённых сессий")
            return

        print("\n" + "=" * 60)
        print("💬 СОХРАНЁННЫЕ СЕССИИ:")
        print("=" * 60)
        for session_id, updated_at, msg_count in sessions:
            current = " ← ТЕКУЩАЯ" if session_id == self.agent.session_id else ""
            print(f"  📌 {session_id} - {msg_count} сообщений, обновлена: {updated_at[:19]}{current}")
        print("=" * 60 + "\n")

    def run(self):
        """Запуск CLI интерфейса"""
        print("\n" + "=" * 60)
        print("🤖 ДОБРО ПОЖАЛОВАТЬ В PERSISTENT GIGACHAT АГЕНТ")
        print("=" * 60)
        print(f"📌 Модель: {self.agent.model}")
        print(f"📌 Сессия: {self.agent.session_id}")
        print(f"📌 История: {len(self.agent.get_history())} сообщений в памяти")
        print("📌 Введите вопрос или команду ( /help для справки )")
        print("=" * 60 + "\n")

        while self.running:
            try:
                user_input = input("👤 Вы: ").strip()

                if not user_input:
                    continue

                # Обработка команд
                if user_input.startswith("/"):
                    cmd = user_input.lower()

                    if cmd == "/exit":
                        print("👋 До свидания! История сохранена.")
                        self.running = False
                        break

                    elif cmd == "/help":
                        self._print_help()

                    elif cmd == "/clear":
                        self.agent.clear_history()

                    elif cmd == "/history":
                        self._print_history()

                    elif cmd == "/sessions":
                        self._print_sessions()

                    elif cmd == "/model":
                        print(f"📌 Текущая модель: {self.agent.model}")

                    elif cmd.startswith("/model pro"):
                        self.agent.set_model("GigaChat-Pro")

                    elif cmd.startswith("/model max"):
                        self.agent.set_model("GigaChat-Max")

                    elif cmd.startswith("/switch"):
                        parts = cmd.split()
                        if len(parts) > 1:
                            new_session = parts[1]
                            self.agent.switch_session(new_session)
                        else:
                            print("❌ Укажите имя сессии, например: /switch work")

                    elif cmd == "/new":
                        import uuid
                        new_session = f"session_{uuid.uuid4().hex[:8]}"
                        self.agent.switch_session(new_session)
                        print(f"✨ Создана новая сессия: {new_session}")

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
                print("\n\n👋 Прервано пользователем. История сохранена!")
                self.running = False
                break
            except Exception as e:
                print(f"\n❌ Неожиданная ошибка: {e}")
                print("   Попробуйте ещё раз или введите /exit")


def main():
    """Главная функция"""
    AUTH_KEY = "Basic <ваш_base64_ключ>"  # Замените на ваш ключ

    if AUTH_KEY == "Basic <ваш_base64_ключ>":
        print("⚠️ ВНИМАНИЕ: Не указан ключ авторизации!")
        print("   Отредактируйте файл и добавьте ваш ключ в AUTH_KEY")
        return

    # Создаём агента с постоянным хранилищем
    agent = PersistentGigaChatAgent(
        auth_key=AUTH_KEY,
        model="GigaChat",
        db_path="agent_history.db",
        session_id="main"  # Можно изменить на любое имя
    )

    # Запускаем CLI
    cli = PersistentCLI(agent)
    cli.run()


if __name__ == "__main__":
    main()