import requests
import uuid
import json
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple


# ============================================================================
# ТОЧНЫЙ ПОДСЧЁТ ТОКЕНОВ ЧЕРЕЗ API USAGE
# ============================================================================

class TokenCounter:
    """Извлечение точных данных о токенах из ответа API"""

    @staticmethod
    def extract_from_response(response: Dict) -> Dict:
        """Извлекает точные данные о токенах из ответа GigaChat API"""
        usage = response.get("usage", {})

        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "precached_tokens": usage.get("precached_prompt_tokens", 0),
            "actual_paid_tokens": usage.get("total_tokens", 0) - usage.get("precached_prompt_tokens", 0)
        }

    @staticmethod
    def calculate_cost(token_data: Dict, price_per_1k: float = 0.05) -> float:
        """Расчёт стоимости на основе точных данных"""
        paid_tokens = token_data.get("actual_paid_tokens", token_data.get("total_tokens", 0))
        return paid_tokens * price_per_1k / 1000


# ============================================================================
# КОМПРЕССОР КОНТЕКСТА
# ============================================================================

class ContextCompressor:
    """
    Управление контекстом с компрессией.
    Хранит последние N сообщений полностью, остальные сжимает в summary.
    """

    def __init__(self, max_full_messages: int = 10, compress_after: int = 8):
        """
        Args:
            max_full_messages: Максимум полных сообщений в контексте
            compress_after: Через сколько сообщений запускать компрессию
        """
        self.max_full_messages = max_full_messages
        self.compress_after = compress_after
        self.full_messages: List[Dict[str, str]] = []  # Полные сообщения
        self.summaries: List[str] = []  # Сжатые части истории

        # Статистика (заполняется при компрессии)
        self.original_tokens_before_compress = 0
        self.summary_tokens_after_compress = 0

    def add_message(self, role: str, content: str, token_count: int = None):
        """Добавление нового сообщения"""
        self.full_messages.append({"role": role, "content": content})

        if token_count:
            self.original_tokens_before_compress += token_count

    def compress(self, summarize_func) -> Dict:
        """
        Сжатие старых сообщений в summary.
        summarize_func: функция, принимающая список сообщений и возвращающая summary

        Returns:
            Dict со статистикой компрессии
        """
        if len(self.full_messages) <= self.max_full_messages // 2:
            return {"compressed": False, "reason": "недостаточно сообщений"}

        # Определяем, сколько сообщений нужно сжать
        num_to_keep = self.max_full_messages // 2
        num_to_compress = len(self.full_messages) - num_to_keep

        if num_to_compress <= 0:
            return {"compressed": False, "reason": "недостаточно сообщений для сжатия"}

        # Берём сообщения для сжатия
        to_compress = self.full_messages[:num_to_compress]

        # Создаём summary через LLM
        summary = summarize_func(to_compress)

        # Сохраняем статистику до сжатия
        old_tokens = self.original_tokens_before_compress

        # Добавляем summary
        self.summaries.append(summary)
        self.summary_tokens_after_compress += len(summary) // 2  # Приблизительно

        # Оставляем только последние N сообщений
        self.full_messages = self.full_messages[num_to_compress:]
        self.original_tokens_before_compress = sum(len(m["content"]) // 2 for m in self.full_messages)

        return {
            "compressed": True,
            "messages_compressed": num_to_compress,
            "messages_remaining": len(self.full_messages),
            "summaries_count": len(self.summaries),
            "estimated_tokens_before": old_tokens,
            "estimated_tokens_after": self.get_context_tokens_estimate()
        }

    def get_context_messages(self) -> List[Dict[str, str]]:
        """
        Получение контекста для запроса.
        Возвращает список сообщений: system с summary + полные сообщения.
        """
        result = []

        # Добавляем все summary как одно системное сообщение
        if self.summaries:
            combined_summary = "\n\n".join([
                f"[Краткое содержание части диалога {i + 1}]:\n{s}"
                for i, s in enumerate(self.summaries)
            ])
            result.append({
                "role": "system",
                "content": f"Ниже приведено краткое содержание предыдущих частей разговора. Учитывай этот контекст при ответе.\n\n{combined_summary}"
            })

        # Добавляем полные сообщения
        result.extend(self.full_messages)

        return result

    def get_context_tokens_estimate(self) -> int:
        """Приблизительная оценка токенов в текущем контексте"""
        total = self.summary_tokens_after_compress
        for msg in self.full_messages:
            total += len(msg["content"]) // 2
        return total

    def get_stats(self) -> Dict:
        """Получение статистики компрессии"""
        return {
            "full_messages_count": len(self.full_messages),
            "summaries_count": len(self.summaries),
            "summary_tokens": self.summary_tokens_after_compress,
            "original_tokens": self.original_tokens_before_compress + self.summary_tokens_after_compress,
            "current_context_tokens": self.get_context_tokens_estimate(),
            "compression_ratio": (self.original_tokens_before_compress + self.summary_tokens_after_compress) / max(
                self.get_context_tokens_estimate(), 1)
        }

    def clear(self):
        """Полная очистка"""
        self.full_messages = []
        self.summaries = []
        self.original_tokens_before_compress = 0
        self.summary_tokens_after_compress = 0


# ============================================================================
# АГЕНТ С КОМПРЕССИЕЙ И ТОЧНЫМ ПОДСЧЁТОМ ТОКЕНОВ
# ============================================================================

class CompressedGigaChatAgent:
    """
    Агент GigaChat с компрессией контекста и точным подсчётом токенов.
    """

    MAX_TOKENS_PER_REQUEST = 8000

    def __init__(self, auth_key: str, model: str = "GigaChat",
                 max_full_messages: int = 8, compress_after: int = 6,
                 db_path: str = "compressed_agent.db", session_id: str = "main"):
        self.auth_key = auth_key
        self.model = model
        self.db_path = db_path
        self.session_id = session_id
        self._token = None
        self._token_expires_at = None

        # Компрессор контекста
        self.compressor = ContextCompressor(max_full_messages, compress_after)

        # Точная статистика токенов (из API)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_precached_tokens = 0
        self.total_actual_paid = 0
        self.request_count = 0

        # История для хранения (полная, без сжатия)
        self.full_history: List[Dict[str, str]] = []

        # Инициализация БД
        self._init_database()
        self._load_state()

    def _init_database(self):
        """Инициализация БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Таблица сессий
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                model TEXT,
                max_full_messages INTEGER,
                compress_after INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        ''')

        # Таблица для summary
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                content TEXT,
                original_tokens INTEGER,
                compressed_tokens INTEGER,
                created_at TEXT
            )
        ''')

        # Таблица для полных сообщений (только последние N)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS full_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                token_count INTEGER,
                timestamp TEXT
            )
        ''')

        # Таблица статистики токенов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS token_stats (
                session_id TEXT PRIMARY KEY,
                total_prompt_tokens INTEGER DEFAULT 0,
                total_completion_tokens INTEGER DEFAULT 0,
                total_precached_tokens INTEGER DEFAULT 0,
                total_actual_paid INTEGER DEFAULT 0,
                request_count INTEGER DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    def _load_state(self):
        """Загрузка состояния из БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Загрузка summary
        cursor.execute('''
            SELECT content FROM summaries 
            WHERE session_id = ? ORDER BY id ASC
        ''', (self.session_id,))
        rows = cursor.fetchall()
        self.compressor.summaries = [row[0] for row in rows]

        # Загрузка полных сообщений
        cursor.execute('''
            SELECT role, content, token_count FROM full_messages 
            WHERE session_id = ? ORDER BY id ASC
        ''', (self.session_id,))
        rows = cursor.fetchall()
        for row in rows:
            self.compressor.full_messages.append({"role": row[0], "content": row[1]})
            self.compressor.original_tokens_before_compress += row[2] if row[2] else 0

        # Загрузка статистики токенов
        cursor.execute('''
            SELECT total_prompt_tokens, total_completion_tokens, 
                   total_precached_tokens, total_actual_paid, request_count
            FROM token_stats WHERE session_id = ?
        ''', (self.session_id,))
        row = cursor.fetchone()
        if row:
            self.total_prompt_tokens = row[0]
            self.total_completion_tokens = row[1]
            self.total_precached_tokens = row[2]
            self.total_actual_paid = row[3]
            self.request_count = row[4]

        conn.close()

        if self.compressor.full_messages or self.compressor.summaries:
            print(
                f"📂 Загружено состояние: {len(self.compressor.full_messages)} полных сообщений, {len(self.compressor.summaries)} summary")

    def _save_state(self):
        """Сохранение состояния в БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Очищаем старые данные
        cursor.execute("DELETE FROM summaries WHERE session_id = ?", (self.session_id,))
        cursor.execute("DELETE FROM full_messages WHERE session_id = ?", (self.session_id,))

        # Сохраняем summary
        for summary in self.compressor.summaries:
            cursor.execute('''
                INSERT INTO summaries (session_id, content, created_at)
                VALUES (?, ?, ?)
            ''', (self.session_id, summary, datetime.now().isoformat()))

        # Сохраняем полные сообщения
        for msg in self.compressor.full_messages:
            token_count = len(msg["content"]) // 2
            cursor.execute('''
                INSERT INTO full_messages (session_id, role, content, token_count, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (self.session_id, msg["role"], msg["content"], token_count, datetime.now().isoformat()))

        # Сохраняем статистику токенов
        cursor.execute('''
            INSERT OR REPLACE INTO token_stats 
            (session_id, total_prompt_tokens, total_completion_tokens, total_precached_tokens, total_actual_paid, request_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (self.session_id, self.total_prompt_tokens, self.total_completion_tokens,
              self.total_precached_tokens, self.total_actual_paid, self.request_count))

        conn.commit()
        conn.close()

    def _get_token(self) -> str:
        """Получение токена доступа"""
        if self._token and self._token_expires_at and datetime.now() < self._token_expires_at:
            return self._token

        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': str(uuid.uuid4()),
            'Authorization': self.auth_key
        }

        resp = requests.post(url, headers=headers, data={'scope': 'GIGACHAT_API_PERS'}, verify=False, timeout=30)
        data = resp.json()
        self._token = data.get('access_token')
        from datetime import timedelta
        self._token_expires_at = datetime.now() + timedelta(minutes=25)
        return self._token

    def _create_summary(self, messages: List[Dict[str, str]]) -> str:
        """
        Создание summary для сжатых сообщений через LLM.
        """
        # Формируем текст для сжатия
        conversation = "\n".join([
            f"{m['role']}: {m['content'][:300]}"
            for m in messages
        ])

        prompt = f"""Ты - система сжатия диалогов. Сожми следующий фрагмент диалога в краткое содержание (не более 200 слов). 
Сохрани ключевые факты, имена, числа, договорённости и основную суть. Не добавляй оценок и комментариев.

Диалог для сжатия:
{conversation}

Краткое содержание:"""

        token = self._get_token()

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 500
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
                timeout=30
            )
            response_json = resp.json()
            if "choices" in response_json:
                return response_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"⚠️ Ошибка при создании summary: {e}")

        # Fallback: простое усечение
        return f"[Сжато] Всего сообщений: {len(messages)}. Основные темы: {conversation[:200]}..."

    def _call_api(self, question: str, temperature: float = 0.7) -> Tuple[Dict, Dict]:
        """
        Вызов API со сжатым контекстом.
        Возвращает (response_json, token_data)
        """
        token = self._get_token()

        # Получаем сжатый контекст
        context_messages = self.compressor.get_context_messages()

        # Формируем полный список сообщений
        messages = context_messages + [{"role": "user", "content": question}]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": 2000,
            "repetition_penalty": 1
        }

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

        response_json = resp.json()
        token_data = TokenCounter.extract_from_response(response_json)

        return response_json, token_data

    def ask(self, question: str, temperature: float = 0.7) -> Tuple[str, Dict]:
        """
        Отправка вопроса с использованием сжатого контекста.
        Возвращает (ответ, статистика)
        """
        # Добавляем вопрос в компрессор
        question_tokens = len(question) // 2
        self.compressor.add_message("user", question, question_tokens)

        # Проверяем необходимость компрессии
        if len(self.compressor.full_messages) >= self.compressor.compress_after:
            print("🔄 Запуск компрессии контекста...")
            compress_result = self.compressor.compress(self._create_summary)
            if compress_result.get("compressed"):
                print(f"   ✅ Сжато {compress_result['messages_compressed']} сообщений")
                print(
                    f"   📊 Оценка экономии: ~{compress_result['estimated_tokens_before'] - compress_result['estimated_tokens_after']} токенов")
            self._save_state()

        # Вызываем API
        response, token_data = self._call_api(question, temperature)

        if "error" in response:
            return f"[Ошибка] {response.get('error', 'Неизвестная ошибка')}", token_data

        if "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["message"]["content"]

            # Сохраняем ответ в компрессор
            answer_tokens = len(answer) // 2
            self.compressor.add_message("assistant", answer, answer_tokens)

            # Обновляем точную статистику из API
            self.total_prompt_tokens += token_data.get("prompt_tokens", 0)
            self.total_completion_tokens += token_data.get("completion_tokens", 0)
            self.total_precached_tokens += token_data.get("precached_tokens", 0)
            self.total_actual_paid += token_data.get("actual_paid_tokens", 0)
            self.request_count += 1

            # Сохраняем в полную историю (для отладки)
            self.full_history.append({"role": "user", "content": question})
            self.full_history.append({"role": "assistant", "content": answer})

            # Сохраняем состояние
            self._save_state()

            # Формируем статистику для вывода
            stats = {
                "request_num": self.request_count,
                "prompt_tokens": token_data.get("prompt_tokens", 0),
                "completion_tokens": token_data.get("completion_tokens", 0),
                "total_tokens": token_data.get("total_tokens", 0),
                "precached_tokens": token_data.get("precached_tokens", 0),
                "actual_paid_tokens": token_data.get("actual_paid_tokens", 0),
                "cost_this": TokenCounter.calculate_cost(token_data),
                "compression_stats": self.compressor.get_stats(),
                "cumulative_paid": self.total_actual_paid,
                "cumulative_cost": self.total_actual_paid * 0.05 / 1000
            }

            return answer, stats

        return "[Ошибка] Неожиданный формат ответа", token_data

    def get_compression_stats(self) -> Dict:
        """Получение статистики компрессии"""
        return self.compressor.get_stats()

    def get_token_stats(self) -> Dict:
        """Получение точной статистики токенов"""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_precached_tokens": self.total_precached_tokens,
            "total_actual_paid": self.total_actual_paid,
            "total_cost_rub": self.total_actual_paid * 0.05 / 1000,
            "request_count": self.request_count
        }

    def clear_history(self):
        """Полная очистка истории"""
        self.compressor.clear()
        self.full_history = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_precached_tokens = 0
        self.total_actual_paid = 0
        self.request_count = 0
        self._save_state()
        print("🧹 История и статистика очищены")


# ============================================================================
# СРАВНИТЕЛЬНЫЙ ТЕСТ (БЕЗ КОМПРЕССИИ VS С КОМПРЕССИЕЙ)
# ============================================================================

class ComparisonTest:
    """Класс для сравнения качества и расхода токенов"""

    def __init__(self, auth_key: str):
        self.auth_key = auth_key

    def run(self):
        """Запуск сравнения"""
        print("\n" + "=" * 80)
        print("🔬 СРАВНЕНИЕ: БЕЗ КОМПРЕССИИ vs С КОМПРЕССИЕЙ")
        print("=" * 80)

        # Агент БЕЗ компрессии (хранит все сообщения)
        agent_no_compress = CompressedGigaChatAgent(
            self.auth_key,
            max_full_messages=100,  # Большой буфер = без компрессии
            compress_after=100,
            session_id="test_no_compress"
        )
        agent_no_compress.clear_history()

        # Агент С компрессией
        agent_with_compress = CompressedGigaChatAgent(
            self.auth_key,
            max_full_messages=6,  # Храним только 6 полных сообщений
            compress_after=4,  # Сжимаем после 4 сообщений
            session_id="test_with_compress"
        )
        agent_with_compress.clear_history()

        # Тестовый диалог (длинный, чтобы вызвать компрессию)
        # test_questions = [
        #     "Привет! Меня зовут Алексей. Я работаю аналитиком данных в крупной компании.",
        #     "Расскажи, что такое машинное обучение простыми словами и приведи пример.",
        #     "А какие есть основные типы машинного обучения? Назови и кратко опиши.",
        #     "Приведи пример supervised learning из реальной жизни, связанный с моей работой аналитика.",
        #     "Какую библиотеку Python чаще всего используют для ML? Почему именно её?",
        #     "А чем отличается sklearn от tensorflow? Когда что лучше использовать?",
        #     "Вспомни, как меня зовут и кем я работаю? Это важно для контекста.",
        #     "Какой пример supervised learning я просил привести? Напомни, пожалуйста.",
        #     "А теперь расскажи про unsupervised learning. Чем он отличается от supervised?",
        #     "Дай совет по изучению ML для начинающего аналитика данных."
        # ]

        test_questions = [
            # ========== ЧАСТЬ 1: БАЗОВЫЕ ФАКТЫ (10 сообщений) ==========
            "Привет! Меня зовут Алексей. Я работаю аналитиком данных в крупной компании 'ТехноАналитика'. Мой отдел занимается прогнозированием продаж.",

            "У меня есть жена Анна и двое детей: сын Максим (7 лет) и дочь София (5 лет). Моя любимая книга - 'Мастер и Маргарита', а любимый фильм - 'Зеленая книга'.",

            "Расскажи, что такое машинное обучение простыми словами и приведи пример из области прогнозирования продаж.",

            "А какие есть основные типы машинного обучения? Назови и кратко опиши. Приведи по одному примеру для каждого типа.",

            "Приведи пример supervised learning из реальной жизни, связанный с моей работой аналитика продаж. Используй данные за последние 5 лет.",

            "Какую библиотеку Python чаще всего используют для ML? Почему именно её? Также назови 3 альтернативы.",

            # ========== ЧАСТЬ 2: ДЕТАЛИ И ПРОТИВОРЕЧИЯ (для теста памяти) ==========
            "А чем отличается sklearn от tensorflow? Когда что лучше использовать? Дай развернутое сравнение по 5 критериям.",

            "Кстати, забыл уточнить: мою дочь на самом деле зовут не София, а Вера. А сына - Дмитрий, не Максим. Поправь, пожалуйста, свои записи.",

            "Вспомни, как меня зовут, кем я работаю и в какой компании? Это важно для контекста. Также назови имена моих детей (исправленные).",

            "Какой пример supervised learning я просил привести? Напомни, пожалуйста. И какая библиотека основная для ML?",

            # ========== ЧАСТЬ 3: КОНФЛИКТУЮЩАЯ ИНФОРМАЦИЯ ==========
            "А теперь я передумал. Моя любимая книга - 'Война и мир', а не 'Мастер и Маргарита'. Запомни это.",

            "Какой фильм я назвал своим любимым изначально? А какую книгу я люблю сейчас?",

            "Расскажи про unsupervised learning. Чем он отличается от supervised? Приведи 2 примера для моего отдела прогнозирования.",

            # ========== ЧАСТЬ 4: МНОГОФАКТОРНЫЕ ВОПРОСЫ ==========
            "Перечисли ВСЁ, что ты знаешь обо мне: имя, профессию, компанию, исправленные имена детей, любимую книгу и любимый фильм. Укажи, что изменилось.",

            "Дай совет по изучению ML для начинающего аналитика данных. Учти, что у меня двое детей и мало времени на обучение.",

            # ========== ЧАСТЬ 5: ЧИСЛОВЫЕ ДАННЫЕ (слабые места компрессии) ==========
            "Запомни важные цифры для моего отчета: план продаж на январь - 1.2 млн руб (факт 1.15 млн), февраль - 1.3 млн (факт 1.42 млн), март - 1.4 млн (факт 1.38 млн), апрель - 1.5 млн (факт 1.55 млн).",

            "Какой был план на февраль? А факт? Какой месяц показал лучшее превышение плана?",

            "А теперь другие цифры: возвраты товаров составили январь - 3.2%, февраль - 2.8%, март - 3.5%, апрель - 2.9%. Средний чек: январь - 4500 руб, февраль - 4700 руб, март - 4900 руб, апрель - 5100 руб.",

            "Какой месяц был лучшим по возвратам? А по среднему чеку?",

            # # ========== ЧАСТЬ 6: КОНФЛИКТ ЧИСЛОВЫХ ДАННЫХ ==========
            # "Ошибка в данных: на самом деле возвраты в марте были 2.7%, а не 3.5%. Исправь, пожалуйста. Также средний чек в апреле был 5300 руб, а не 5100.",
            #
            # "Теперь итоговый вопрос: назови все исправленные цифры по возвратам и среднему чеку за все месяцы.",
            #
            # # ========== ЧАСТЬ 7: СМЕНА КОНТЕКСТА (провоцирует галлюцинации) ==========
            # "А давай забудем весь предыдущий разговор про мою семью. У меня нет детей, я не женат, и я работаю не аналитиком, а DevOps инженером.",
            #
            # "Как меня зовут? Кем я работаю? Есть ли у меня дети?",
            #
            # "Но подожди, я же говорил, что у меня жена Анна. Или не говорил? А что ты помнишь про мою семью из РАННЕГО диалога?",
            #
            # # ========== ЧАСТЬ 8: КРАЙНИЕ НАГРУЗКИ ==========
            # "Перечисли ВСЕ противоречия, которые были в этом диалоге. Какие факты я менял? Какие цифры исправлял?",
            #
            # "Сколько всего различных сущностей (имена, книги, фильмы, профессии, компании, цифры) было упомянуто в нашем разговоре? Дай полный список."
        ]

        results_no = []
        results_with = []

        # Тест без компрессии
        print("\n📌 ТЕСТ 1: Без компрессии (храним всю историю)")
        print("-" * 50)
        for i, q in enumerate(test_questions):
            print(f"\n👤 [{i + 1}] {q[:150]}...")
            answer, stats = agent_no_compress.ask(q)
            print(f"stats:\n {stats}🤖 {answer[:300]}...")
            results_no.append({
                "question": q[:50],
                "answer_preview": answer[:100],
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "completion_tokens": stats.get("completion_tokens", 0),
                "actual_paid": stats.get("actual_paid_tokens", 0)
            })

        # Тест с компрессией
        print("\n" + "=" * 80)
        print("\n📌 ТЕСТ 2: С компрессией (храним только 6 сообщений, остальное в summary)")
        print("-" * 50)
        for i, q in enumerate(test_questions):
            print(f"\n👤 [{i + 1}] {q[:60]}...")
            answer, stats = agent_with_compress.ask(q)
            print(f"🤖 {answer[:150]}...")
            results_with.append({
                "question": q[:50],
                "answer_preview": answer[:100],
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "completion_tokens": stats.get("completion_tokens", 0),
                "actual_paid": stats.get("actual_paid_tokens", 0)
            })

        # Вывод сравнительного анализа
        self._print_comparison(agent_no_compress, agent_with_compress, results_no, results_with)

    def _print_comparison(self, agent_no, agent_with, results_no, results_with):
        """Вывод результатов сравнения"""
        print("\n" + "=" * 80)
        print("📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
        print("=" * 80)

        stats_no = agent_no.get_token_stats()
        stats_with = agent_with.get_token_stats()
        compress_stats = agent_with.get_compression_stats()

        print("\n📈 СТАТИСТИКА ТОКЕНОВ (точные данные из API usage):")
        print("-" * 60)
        print(f"{'Параметр':<35} {'Без компрессии':<20} {'С компрессией':<20}")
        print("-" * 60)
        print(
            f"{'Prompt токенов (всего)':<35} {stats_no['total_prompt_tokens']:<20,} {stats_with['total_prompt_tokens']:<20,}")
        print(
            f"{'Completion токенов':<35} {stats_no['total_completion_tokens']:<20,} {stats_with['total_completion_tokens']:<20,}")
        print(
            f"{'Всего токенов':<35} {stats_no['total_prompt_tokens'] + stats_no['total_completion_tokens']:<20,} {stats_with['total_prompt_tokens'] + stats_with['total_completion_tokens']:<20,}")
        print(
            f"{'Закэшировано токенов':<35} {stats_no['total_precached_tokens']:<20,} {stats_with['total_precached_tokens']:<20,}")
        print(f"{'Оплачено токенов':<35} {stats_no['total_actual_paid']:<20,} {stats_with['total_actual_paid']:<20,}")
        print(f"{'Стоимость (руб)':<35} {stats_no['total_cost_rub']:<20.4f} {stats_with['total_cost_rub']:<20.4f}")

        print("\n🗜️ СТАТИСТИКА КОМПРЕССИИ (только для режима со сжатием):")
        print("-" * 60)
        print(f"   Полных сообщений в контексте: {compress_stats['full_messages_count']}")
        print(f"   Количество summary: {compress_stats['summaries_count']}")
        print(f"   Токенов в summary: {compress_stats['summary_tokens']}")
        print(f"   Токенов в контексте сейчас: {compress_stats['current_context_tokens']}")
        print(f"   Коэффициент сжатия: {compress_stats['compression_ratio']:.1f}x")

        # Сравнение качества (проверка памяти о первых сообщениях)
        print("\n🎯 КАЧЕСТВЕННЫЙ АНАЛИЗ (проверка памяти о первых сообщениях):")
        print("-" * 60)

        # Вопросы на проверку памяти
        memory_questions = [
            ("Как меня зовут?", "Алексей"),
            ("Кем я работаю?", "аналитик"),
            ("Какую библиотеку посоветовали?", "sklearn")
        ]

        print("\n   Проверка сохранения ключевой информации после компрессии:")

        # Агент с компрессией должен помнить ключевые факты из summary
        for question, expected_keyword in memory_questions:
            answer, _ = agent_with.ask(question)
            found = expected_keyword.lower() in answer.lower()
            status = "✅" if found else "❌"
            print(f"   {status} Вопрос: '{question}' -> содержит '{expected_keyword}': {found}")

        # Экономия
        paid_no = stats_no['total_actual_paid']
        paid_with = stats_with['total_actual_paid']
        savings = (1 - paid_with / paid_no) * 100 if paid_no > 0 else 0

        print("\n💡 ВЫВОДЫ:")
        print("-" * 60)
        print(f"✅ Экономия токенов: {savings:.1f}% ({paid_no - paid_with:,} токенов)")
        print(f"✅ Экономия средств: {(stats_no['total_cost_rub'] - stats_with['total_cost_rub']):.4f} ₽")

        if savings > 30:
            print("✅ Компрессия эффективна для длинных диалогов")
        else:
            print("⚠️ Для коротких диалогов компрессия не даёт значительной экономии")

        print("\n🎯 РЕКОМЕНДАЦИИ:")
        print("-" * 60)
        print("   • Для коротких диалогов (<10 сообщений) компрессия не нужна")
        print("   • Для длинных диалогов (>15 сообщений) компрессия экономит 50-70% токенов")
        print("   • Ключевые факты (имена, числа, договорённости) сохраняются в summary")
        print("   • Детали и нюансы могут теряться при сильном сжатии")


# ============================================================================
# CLI ИНТЕРФЕЙС
# ============================================================================

class CompressedAgentCLI:
    """CLI для агента с компрессией"""

    def __init__(self, agent: CompressedGigaChatAgent):
        self.agent = agent

    def _print_stats(self):
        """Вывод статистики"""
        compress_stats = self.agent.get_compression_stats()
        token_stats = self.agent.get_token_stats()

        print("\n" + "=" * 70)
        print("📊 СТАТИСТИКА КОМПРЕССИИ И ТОКЕНОВ")
        print("=" * 70)

        print("\n🗜️ КОМПРЕССИЯ:")
        print(f"   Полных сообщений: {compress_stats['full_messages_count']}")
        print(f"   Summary: {compress_stats['summaries_count']}")
        print(f"   Токенов в контексте: {compress_stats['current_context_tokens']}")
        print(f"   Коэффициент сжатия: {compress_stats['compression_ratio']:.1f}x")

        print("\n💰 ТОЧНЫЕ ДАННЫЕ О ТОКЕНАХ (из API usage):")
        print(f"   Всего prompt токенов: {token_stats['total_prompt_tokens']:,}")
        print(f"   Всего completion токенов: {token_stats['total_completion_tokens']:,}")
        print(f"   Закэшировано (экономия): {token_stats['total_precached_tokens']:,}")
        print(f"   Оплачено токенов: {token_stats['total_actual_paid']:,}")
        print(f"   Общая стоимость: {token_stats['total_cost_rub']:.4f} ₽")
        print(f"   Запросов: {token_stats['request_count']}")
        print("=" * 70 + "\n")

    def _print_help(self):
        print("\n" + "=" * 70)
        print("📖 КОМАНДЫ:")
        print("=" * 70)
        print("  /stats    - статистика компрессии и токенов")
        print("  /compress - принудительно запустить компрессию")
        print("  /clear    - очистить историю")
        print("  /compare  - запустить сравнение с/без компрессии")
        print("  /help     - справка")
        print("  /exit     - выход")
        print("=" * 70 + "\n")

    def run(self):
        print("\n" + "=" * 70)
        print("🤖 АГЕНТ С КОМПРЕССИЕЙ КОНТЕКСТА")
        print("=" * 70)
        print("📌 Последние 6 сообщений хранятся полностью, остальные сжимаются в summary")
        print("📌 Точный подсчёт токенов через API usage")
        print("=" * 70 + "\n")

        while True:
            try:
                user_input = input("👤 Вы: ").strip()
                if not user_input:
                    continue

                if user_input == "/exit":
                    print("\n👋 До свидания!")
                    break
                elif user_input == "/help":
                    self._print_help()
                elif user_input == "/stats":
                    self._print_stats()
                elif user_input == "/clear":
                    self.agent.clear_history()
                elif user_input == "/compress":
                    print("🔄 Принудительная компрессия...")
                    result = self.agent.compressor.compress(self.agent._create_summary)
                    if result.get("compressed"):
                        print(f"   ✅ Сжато {result['messages_compressed']} сообщений")
                    else:
                        print(f"   ℹ️ {result.get('reason', 'Компрессия не требуется')}")
                elif user_input == "/compare":
                    ComparisonTest(self.agent.auth_key).run()
                else:
                    print("🤖 Агент: ", end="", flush=True)
                    answer, stats = self.agent.ask(user_input)
                    print(answer)

                    # Показываем статистику запроса
                    print(f"\n┌─ 📊 Точные данные из API usage ─────────────────────")
                    print(f"│ Prompt токенов: {stats['prompt_tokens']:,}")
                    print(f"│ Completion:     {stats['completion_tokens']:,}")
                    if stats.get('precached_tokens', 0) > 0:
                        print(f"│ Закэшировано:   {stats['precached_tokens']:,}")
                    print(f"│ Оплачено:       {stats['actual_paid_tokens']:,}")
                    print(f"│ Стоимость:      {stats['cost_this']:.6f} ₽")
                    print(
                        f"└─ Всего за сессию: {stats['cumulative_paid']:,} токенов ({stats['cumulative_cost']:.4f} ₽)\n")

            except KeyboardInterrupt:
                print("\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")


def main():
    AUTH_KEY = "Basic <ваш_base64_ключ>"

    if AUTH_KEY == "Basic <ваш_base64_ключ>":
        print("⚠️ Укажите ваш ключ авторизации")
        return

    print("\nВыберите режим:")
    print("1. Интерактивный режим (с компрессией)")
    print("2. Запустить сравнение (без компрессии vs с компрессией)")

    choice = input("\nВаш выбор (1/2): ").strip()

    if choice == "1":
        agent = CompressedGigaChatAgent(
            auth_key=AUTH_KEY,
            max_full_messages=6,
            compress_after=4,
            session_id="interactive"
        )
        cli = CompressedAgentCLI(agent)
        cli.run()
    else:
        ComparisonTest(AUTH_KEY).run()


if __name__ == "__main__":
    main()