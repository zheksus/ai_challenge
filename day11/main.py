import requests
import uuid
import json
import sqlite3
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from copy import deepcopy


# ============================================================================
# ТОЧНЫЙ ПОДСЧЁТ ТОКЕНОВ
# ============================================================================

class LLMMemoryClassifier:
    """Классификация сообщений с помощью LLM для определения типа памяти"""

    CLASSIFICATION_PROMPT = """
    Ты - система классификации сообщений для ассистента. Проанализируй сообщение пользователя и определи, какую информацию нужно сохранить.

    Сообщение: {message}

    Ответь строго в формате JSON. Не добавляй никаких других пояснений, только JSON.

    Пример правильного ответа:
    {{"long_term": {{"profile": {{"user_name": null, "user_profession": null}}, "preferences": {{"likes": null, "dislikes": null}}}}, "working": {{"constraints": {{"budget": null, "deadline": null}}, "requirements": [], "decisions": []}}}}

    Теперь ответь для сообщения выше:
    """

    def __init__(self, agent):
        self.agent = agent

    def _clean_json_response(self, text: str) -> str:
        """Очистка JSON ответа от лишних символов"""
        if not text:
            return "{}"

        # Удаляем markdown-блоки
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # Находим JSON объект
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()

        # Заменяем переносы строк и табуляции на пробелы
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)

        # Исправляем типичные ошибки
        text = re.sub(r',\s*}', '}', text)  # Удаляем запятые перед закрывающей скобкой
        text = re.sub(r',\s*]', ']', text)  # Удаляем запятые перед закрывающим массивом

        return text.strip()

    def classify(self, message: str, assistant_response: str = "") -> Dict:
        """Классификация сообщения с помощью LLM"""

        prompt = self.CLASSIFICATION_PROMPT.format(message=message)

        # Формируем сообщения для API
        messages = [
            {"role": "system",
             "content": "Ты - система извлечения информации в формате JSON. Отвечай ТОЛЬКО валидным JSON без пояснений."},
            {"role": "user", "content": prompt}
        ]

        try:
            response, _ = self.agent._call_api(messages, temperature=0.2)

            if "choices" in response:
                result_text = response["choices"][0]["message"]["content"]
                print(f"🔍 [DEBUG] Raw response: {result_text[:200]}...")  # Отладка

                # Очищаем ответ
                cleaned_text = self._clean_json_response(result_text)
                print(f"🔍 [DEBUG] Cleaned: {cleaned_text[:200]}...")  # Отладка

                if cleaned_text and cleaned_text != "{}":
                    result = json.loads(cleaned_text)
                    return result

        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка парсинга JSON: {e}")
            print(f"   Текст: {result_text[:200] if 'result_text' in locals() else 'None'}")
        except Exception as e:
            print(f"⚠️ Ошибка LLM классификации: {e}")

        # Возвращаем пустой результат в случае ошибки
        return {
            "long_term": {
                "profile": {},
                "preferences": {}
            },
            "working": {
                "constraints": {},
                "requirements": [],
                "decisions": []
            }
        }


class TokenCounter:
    @staticmethod
    def extract_from_response(response: Dict) -> Dict:
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
        paid_tokens = token_data.get("actual_paid_tokens", token_data.get("total_tokens", 0))
        return paid_tokens * price_per_1k / 1000


# ============================================================================
# ТИПЫ ДАННЫХ ДЛЯ ПАМЯТИ
# ============================================================================

@dataclass
class ShortTermMemoryItem:
    """Элемент краткосрочной памяти (текущий диалог)"""
    role: str  # 'user' или 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tokens: int = 0

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tokens": self.tokens
        }


@dataclass
class WorkingMemoryItem:
    """Элемент рабочей памяти (данные текущей задачи)"""
    key: str
    value: Any
    type: str  # 'requirement', 'constraint', 'decision', 'question', 'answer'
    confidence: float = 1.0  # 0-1, насколько уверены
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "value": self.value,
            "type": self.type,
            "confidence": self.confidence,
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class LongTermMemoryItem:
    """Элемент долговременной памяти (профиль, решения, знания)"""
    category: str  # 'profile', 'preference', 'decision', 'knowledge', 'agreement'
    key: str
    value: Any
    importance: float = 0.5  # 0-1, насколько важно
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }


# ============================================================================
# МЕНЕДЖЕР ПАМЯТИ
# ============================================================================

class MemoryManager:
    """
    Менеджер трёхуровневой памяти:
    - Краткосрочная: текущий диалог (последние N сообщений)
    - Рабочая: данные текущей задачи (требования, ограничения, решения)
    - Долговременная: профиль пользователя, предпочтения, важные решения
    """

    def __init__(self, db_path: str = "agent_memory.db", session_id: str = "default"):
        self.db_path = db_path
        self.session_id = session_id

        # Три уровня памяти
        self.short_term: List[ShortTermMemoryItem] = []  # текущий диалог
        self.working: Dict[str, WorkingMemoryItem] = {}  # ключ -> значение
        self.long_term: Dict[str, LongTermMemoryItem] = {}  # category:key -> значение

        # Настройки
        self.max_short_term_items = 20  # последние 20 сообщений
        self.auto_sync = True

        # Инициализация БД
        self._init_database()
        self._load_from_db()

    def _init_database(self):
        """Инициализация БД для всех трёх уровней памяти"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Краткосрочная память (сообщения диалога)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS short_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                tokens INTEGER
            )
        ''')

        # Рабочая память (текущая задача)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS working_memory (
                session_id TEXT,
                key TEXT,
                value TEXT,
                type TEXT,
                confidence REAL,
                updated_at TEXT,
                PRIMARY KEY (session_id, key)
            )
        ''')

        # Долговременная память (профиль, предпочтения)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS long_term_memory (
                session_id TEXT,
                category TEXT,
                key TEXT,
                value TEXT,
                importance REAL,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER,
                PRIMARY KEY (session_id, category, key)
            )
        ''')

        conn.commit()
        conn.close()

    def _load_from_db(self):
        """Загрузка данных из БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Загрузка краткосрочной памяти
        cursor.execute('''
            SELECT role, content, timestamp, tokens FROM short_term_memory 
            WHERE session_id = ? ORDER BY id ASC
        ''', (self.session_id,))
        for row in cursor.fetchall():
            self.short_term.append(ShortTermMemoryItem(
                role=row[0],
                content=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                tokens=row[3] or 0
            ))

        # Загрузка рабочей памяти
        cursor.execute('''
            SELECT key, value, type, confidence, updated_at FROM working_memory 
            WHERE session_id = ?
        ''', (self.session_id,))
        for row in cursor.fetchall():
            try:
                value = json.loads(row[1])
            except:
                value = row[1]
            self.working[row[0]] = WorkingMemoryItem(
                key=row[0],
                value=value,
                type=row[2],
                confidence=row[3],
                updated_at=datetime.fromisoformat(row[4])
            )

        # Загрузка долговременной памяти
        cursor.execute('''
            SELECT category, key, value, importance, created_at, last_accessed, access_count 
            FROM long_term_memory WHERE session_id = ?
        ''', (self.session_id,))
        for row in cursor.fetchall():
            try:
                value = json.loads(row[2])
            except:
                value = row[2]
            memory_key = f"{row[0]}:{row[1]}"
            self.long_term[memory_key] = LongTermMemoryItem(
                category=row[0],
                key=row[1],
                value=value,
                importance=row[3],
                created_at=datetime.fromisoformat(row[4]),
                last_accessed=datetime.fromisoformat(row[5]),
                access_count=row[6]
            )

        conn.close()

        # Ограничиваем краткосрочную память
        while len(self.short_term) > self.max_short_term_items:
            self.short_term.pop(0)

    def _save_to_db(self):
        """Сохранение всех уровней памяти в БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Очищаем старые данные для этой сессии
        cursor.execute("DELETE FROM short_term_memory WHERE session_id = ?", (self.session_id,))
        cursor.execute("DELETE FROM working_memory WHERE session_id = ?", (self.session_id,))
        cursor.execute("DELETE FROM long_term_memory WHERE session_id = ?", (self.session_id,))

        # Сохраняем краткосрочную память
        for item in self.short_term:
            cursor.execute('''
                INSERT INTO short_term_memory (session_id, role, content, timestamp, tokens)
                VALUES (?, ?, ?, ?, ?)
            ''', (self.session_id, item.role, item.content, item.timestamp.isoformat(), item.tokens))

        # Сохраняем рабочую память
        for key, item in self.working.items():
            cursor.execute('''
                INSERT INTO working_memory (session_id, key, value, type, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (self.session_id, key, json.dumps(item.value, ensure_ascii=False),
                  item.type, item.confidence, item.updated_at.isoformat()))

        # Сохраняем долговременную память
        for memory_key, item in self.long_term.items():
            cursor.execute('''
                INSERT INTO long_term_memory (session_id, category, key, value, importance, 
                                              created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (self.session_id, item.category, item.key, json.dumps(item.value, ensure_ascii=False),
                  item.importance, item.created_at.isoformat(), item.last_accessed.isoformat(),
                  item.access_count))

        conn.commit()
        conn.close()

    # ========== КРАТКОСРОЧНАЯ ПАМЯТЬ ==========

    def add_to_short_term(self, role: str, content: str, tokens: int = 0):
        """Добавление сообщения в краткосрочную память"""
        item = ShortTermMemoryItem(role=role, content=content, tokens=tokens)
        self.short_term.append(item)

        # Ограничиваем размер
        while len(self.short_term) > self.max_short_term_items:
            removed = self.short_term.pop(0)
            # При необходимости можно архивировать удалённые сообщения
            self._archive_short_term_item(removed)

        if self.auto_sync:
            self._save_to_db()

    def _archive_short_term_item(self, item: ShortTermMemoryItem):
        """Архивация вытесненного из краткосрочной памяти сообщения"""
        # Можем извлечь важные факты в рабочую память
        self._extract_facts_to_working(item.content)

    def _extract_facts_to_working(self, content: str):
        """Извлечение фактов из сообщения в рабочую память"""
        content_lower = content.lower()

        # Извлечение имени
        if "меня зовут" in content_lower:
            match = re.search(r'зовут\s+([А-Я][а-я]+(?:\s+[А-Я][а-я]+)?)', content)
            if match:
                self.add_to_working("user_name", match.group(1), "profile", confidence=0.9)

        # Извлечение профессии
        if "работаю" in content_lower:
            match = re.search(r'работаю\s+([А-Яа-я]+\s*[А-Яа-я]*)', content)
            if match:
                self.add_to_working("user_profession", match.group(1), "profile", confidence=0.8)

        # Извлечение бюджета
        budget_match = re.search(r'бюджет\D*(\d+(?:\.?\d+)?)\s*(?:млн|тыс|руб|миллион|тысяч)', content_lower)
        if budget_match:
            self.add_to_working("budget", float(budget_match.group(1)), "constraint", confidence=0.85)

        # Извлечение срока
        deadline_match = re.search(r'срок\D*(\d+)\s*(?:месяц|недель|дней)', content_lower)
        if deadline_match:
            self.add_to_working("deadline", int(deadline_match.group(1)), "constraint", confidence=0.8)

    def get_short_term_context(self, last_n: int = None) -> List[Dict]:
        """Получение контекста из краткосрочной памяти"""
        n = last_n if last_n else self.max_short_term_items
        recent = self.short_term[-n:]
        return [{"role": item.role, "content": item.content} for item in recent]

    # ========== РАБОЧАЯ ПАМЯТЬ ==========

    def add_to_working(self, key: str, value: Any, type: str, confidence: float = 1.0):
        """Добавление/обновление рабочей памяти"""
        self.working[key] = WorkingMemoryItem(
            key=key,
            value=value,
            type=type,
            confidence=confidence,
            updated_at=datetime.now()
        )

        # Важные данные с высокой уверенностью могут попасть в долговременную память
        if confidence >= 0.8 and type in ["profile", "decision", "agreement"]:
            self._promote_to_long_term(key, value, type)

        if self.auto_sync:
            self._save_to_db()

    def get_from_working(self, key: str) -> Optional[Any]:
        """Получение значения из рабочей памяти"""
        item = self.working.get(key)
        return item.value if item else None

    def get_all_working(self) -> Dict[str, Any]:
        """Получение всей рабочей памяти"""
        return {key: item.value for key, item in self.working.items()}

    def _promote_to_long_term(self, key: str, value: Any, category: str):
        """Продвижение важных данных в долговременную память"""
        memory_key = f"{category}:{key}"
        if memory_key not in self.long_term:
            self.long_term[memory_key] = LongTermMemoryItem(
                category=category,
                key=key,
                value=value,
                importance=0.7
            )
        else:
            self.long_term[memory_key].value = value
            self.long_term[memory_key].last_accessed = datetime.now()
            self.long_term[memory_key].access_count += 1

    # ========== ДОЛГОВРЕМЕННАЯ ПАМЯТЬ ==========

    def add_to_long_term(self, category: str, key: str, value: Any, importance: float = 0.5):
        """Добавление в долговременную память"""
        memory_key = f"{category}:{key}"
        if memory_key in self.long_term:
            self.long_term[memory_key].value = value
            self.long_term[memory_key].last_accessed = datetime.now()
            self.long_term[memory_key].access_count += 1
            self.long_term[memory_key].importance = max(self.long_term[memory_key].importance, importance)
        else:
            self.long_term[memory_key] = LongTermMemoryItem(
                category=category,
                key=key,
                value=value,
                importance=importance
            )

        if self.auto_sync:
            self._save_to_db()

    def get_from_long_term(self, category: str, key: str) -> Optional[Any]:
        """Получение из долговременной памяти"""
        memory_key = f"{category}:{key}"
        item = self.long_term.get(memory_key)
        if item:
            item.last_accessed = datetime.now()
            item.access_count += 1
            if self.auto_sync:
                self._save_to_db()
            return item.value
        return None

    def get_all_long_term(self, category: str = None) -> Dict:
        """Получение всей долговременной памяти (или по категории)"""
        result = {}
        for memory_key, item in self.long_term.items():
            if category is None or item.category == category:
                result[f"{item.category}:{item.key}"] = item.value
        return result

    # ========== ФОРМИРОВАНИЕ КОНТЕКСТА ДЛЯ LLM ==========

    def get_full_context(self, include_short_term: bool = True,
                         include_working: bool = True,
                         include_long_term: bool = True,
                         last_n_messages: int = 10) -> List[Dict[str, str]]:
        """
        Формирование полного контекста для отправки в LLM.
        """
        context = []

        # Объединяем ВСЕ системные сообщения в ОДНО
        system_parts = []

        # 1. Долговременная память
        if include_long_term and self.long_term:
            profile = self.get_all_long_term("profile")
            preferences = self.get_all_long_term("preference")
            agreements = self.get_all_long_term("agreement")

            if profile or preferences or agreements:
                long_term_text = "### ДОЛГОВРЕМЕННАЯ ПАМЯТЬ (профиль и предпочтения):\n"
                if profile:
                    long_term_text += f"- Профиль: {json.dumps(profile, ensure_ascii=False)}\n"
                if preferences:
                    long_term_text += f"- Предпочтения: {json.dumps(preferences, ensure_ascii=False)}\n"
                if agreements:
                    long_term_text += f"- Договорённости: {json.dumps(agreements, ensure_ascii=False)}\n"
                system_parts.append(long_term_text)

        # 2. Рабочая память
        if include_working and self.working:
            working_data = self.get_all_working()
            working_text = "### РАБОЧАЯ ПАМЯТЬ (текущая задача):\n"
            for key, value in working_data.items():
                working_text += f"- {key}: {value}\n"
            system_parts.append(working_text)

        # 3. Добавляем ОДНО system сообщение со всем содержимым
        if system_parts:
            combined_system = "\n".join(system_parts)
            context.append({"role": "system", "content": combined_system})

        # 4. Краткосрочная память (последние сообщения)
        if include_short_term:
            short_term_context = self.get_short_term_context(last_n_messages)
            context.extend(short_term_context)

        return context

    # ========== УПРАВЛЕНИЕ ==========

    def clear_short_term(self):
        """Очистка краткосрочной памяти"""
        self.short_term = []
        if self.auto_sync:
            self._save_to_db()

    def clear_working(self):
        """Очистка рабочей памяти"""
        self.working = {}
        if self.auto_sync:
            self._save_to_db()

    def get_stats(self) -> Dict:
        """Получение статистики памяти"""
        return {
            "short_term_count": len(self.short_term),
            "working_count": len(self.working),
            "long_term_count": len(self.long_term),
            "short_term_items": [{"role": i.role, "preview": i.content[:50]} for i in self.short_term[-5:]],
            "working_items": {k: {"type": v.type, "value": v.value} for k, v in self.working.items()},
            "long_term_by_category": {
                cat: len([i for i in self.long_term.values() if i.category == cat])
                for cat in set(i.category for i in self.long_term.values())
            }
        }


# ============================================================================
# АГЕНТ С МНОГОУРОВНЕВОЙ ПАМЯТЬЮ
# ============================================================================

class MemoryAwareAgent:
    """Агент с трёхуровневой моделью памяти"""

    def __init__(self, auth_key: str, model: str = "GigaChat", session_id: str = "default"):
        self.auth_key = auth_key
        self.model = model
        self.session_id = session_id
        self._token = None
        self._token_expires_at = None

        # Менеджер памяти
        self.memory = MemoryManager(session_id=session_id)

        # Статистика
        self.total_actual_paid = 0
        self.request_count = 0

    def _get_token(self) -> str:
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

    def _call_api(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Tuple[Dict, Dict]:
        token = self._get_token()

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

    def _extract_and_store_memory(self, user_message: str, assistant_response: str):
        """
        Извлечение информации из диалога с использованием LLM.
        """
        # Используем LLM для классификации
        if not hasattr(self, 'llm_classifier'):
            self.llm_classifier = LLMMemoryClassifier(self)

        classification = self.llm_classifier.classify(user_message, assistant_response)

        if not classification:
            print("⚠️ LLM классификация не дала результата")
            return

        # ===== ДОЛГОВРЕМЕННАЯ ПАМЯТЬ =====
        long_term = classification.get("long_term", {})

        # Профиль
        profile = long_term.get("profile", {})
        if profile.get("user_name"):
            self.memory.add_to_long_term("profile", "user_name", profile["user_name"], importance=0.9)
            print(f"📝 [LLM] Сохранено имя: {profile['user_name']}")

        if profile.get("user_profession"):
            self.memory.add_to_long_term("profile", "user_profession", profile["user_profession"], importance=0.8)
            print(f"📝 [LLM] Сохранена профессия: {profile['user_profession']}")

        if profile.get("company"):
            self.memory.add_to_long_term("profile", "company", profile["company"], importance=0.7)
            print(f"📝 [LLM] Сохранена компания: {profile['company']}")

        # Предпочтения
        preferences = long_term.get("preferences", {})
        if preferences.get("likes"):
            self.memory.add_to_long_term("preference", "likes", preferences["likes"], importance=0.6)
            print(f"📝 [LLM] Сохранено предпочтение: {preferences['likes']}")

        if preferences.get("dislikes"):
            self.memory.add_to_long_term("preference", "dislikes", preferences["dislikes"], importance=0.6)
            print(f"📝 [LLM] Сохранён антипатия: {preferences['dislikes']}")

        # Знания
        if long_term.get("knowledge"):
            self.memory.add_to_long_term("knowledge", "fact", long_term["knowledge"], importance=0.5)
            print(f"📝 [LLM] Сохранён факт: {long_term['knowledge']}")

        # ===== РАБОЧАЯ ПАМЯТЬ =====
        working = classification.get("working", {})

        # Ограничения
        constraints = working.get("constraints", {})
        if constraints.get("budget"):
            self.memory.add_to_working("budget", constraints["budget"], "constraint", confidence=0.85)
            print(f"📝 [LLM] Сохранён бюджет: {constraints['budget']}")

        if constraints.get("deadline"):
            self.memory.add_to_working("deadline", constraints["deadline"], "constraint", confidence=0.8)
            print(f"📝 [LLM] Сохранён срок: {constraints['deadline']}")

        # Требования
        requirements = working.get("requirements", [])
        if requirements:
            existing = self.memory.get_from_working("requirements") or []
            if isinstance(existing, list):
                existing.extend(requirements)
                self.memory.add_to_working("requirements", existing, "requirement", confidence=0.7)
            print(f"📝 [LLM] Добавлены требования: {requirements}")

        # Решения
        decisions = working.get("decisions", [])
        for decision in decisions:
            self.memory.add_to_working("last_agreement", decision, "decision", confidence=0.8)
            print(f"📝 [LLM] Сохранено решение: {decision}")


    def _extract_and_store_memory_old(self, user_message: str, assistant_response: str):
        """
        Извлечение информации из диалога и сохранение в соответствующие слои памяти.
        Явно выбираем, что и куда сохраняется.
        """
        content = user_message.lower()

        # ===== В РАБОЧУЮ ПАМЯТЬ (текущая задача) =====

        # Бюджет
        budget_match = re.search(r'бюджет\D*(\d+(?:\.?\d+)?)\s*(?:млн|тыс|руб|миллион|тысяч)', content)
        if budget_match:
            self.memory.add_to_working("budget", float(budget_match.group(1)), "constraint", confidence=0.85)

        # Срок
        deadline_match = re.search(r'срок\D*(\d+)\s*(?:месяц|недель|дней|месяца|недели|дня)', content)
        if deadline_match:
            self.memory.add_to_working("deadline", int(deadline_match.group(1)), "constraint", confidence=0.8)

        # Требования
        if "нужно" in content or "требует" in content or "хочу" in content:
            requirement_match = re.search(r'(?:нужно|хочу|требуется)\s+([^.!?]+)', content)
            if requirement_match:
                reqs = self.memory.get_from_working("requirements") or []
                if isinstance(reqs, list):
                    reqs.append(requirement_match.group(1).strip())
                    self.memory.add_to_working("requirements", reqs, "requirement", confidence=0.7)

        # ===== В ДОЛГОВРЕМЕННУЮ ПАМЯТЬ (профиль) =====

        # Имя
        name_match = re.search(r'меня зовут\s+([А-Я][а-я]+(?:\s+[А-Я][а-я]+)?)', content)
        if name_match:
            self.memory.add_to_long_term("profile", "user_name", name_match.group(1), importance=0.9)

        # Профессия
        profession_match = re.search(r'работаю\s+([А-Яа-я]+\s*[А-Яа-я]*)', content)
        if profession_match:
            self.memory.add_to_long_term("profile", "user_profession", profession_match.group(1), importance=0.8)

        # Предпочтения
        if "люблю" in content or "нравит" in content or "предпочитаю" in content:
            pref_match = re.search(r'(?:люблю|нравится|предпочитаю)\s+([^.!?]+)', content)
            if pref_match:
                self.memory.add_to_long_term("preference", "likes", pref_match.group(1).strip(), importance=0.6)

        # ===== В РАБОЧУЮ ПАМЯТЬ (решения) =====

        # Согласие/отказ
        if "да" in content[:20] or "согласен" in content:
            self.memory.add_to_working("last_agreement", True, "decision", confidence=0.9)
        elif "нет" in content[:20] or "не согласен" in content:
            self.memory.add_to_working("last_agreement", False, "decision", confidence=0.9)

    def ask(self, question: str, temperature: float = 0.7) -> Tuple[str, Dict]:
        """Отправка вопроса с использованием всех уровней памяти"""

        # Сохраняем вопрос пользователя в краткосрочную память
        question_tokens = len(question) // 2
        self.memory.add_to_short_term("user", question, question_tokens)

        # Извлекаем информацию из вопроса в рабочую/долговременную память
        self._extract_and_store_memory(question, "")

        # Формируем контекст из всех трёх уровней памяти
        context = self.memory.get_full_context(
            include_short_term=True,
            include_working=True,
            include_long_term=True,
            last_n_messages=10
        )

        # Вызываем API
        response, token_data = self._call_api(context, temperature)

        if "error" in response:
            return f"[Ошибка] {response.get('error', 'Неизвестная ошибка')}", token_data

        if "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["message"]["content"]

            # Сохраняем ответ в краткосрочную память
            answer_tokens = token_data.get("completion_tokens", len(answer) // 2)
            self.memory.add_to_short_term("assistant", answer, answer_tokens)

            # Обновляем статистику
            self.total_actual_paid += token_data.get("actual_paid_tokens", 0)
            self.request_count += 1

            stats = {
                "request_num": self.request_count,
                "tokens_used": token_data.get("actual_paid_tokens", 0),
                "cost_this": TokenCounter.calculate_cost(token_data),
                "cumulative_cost": self.total_actual_paid * 0.05 / 1000,
                "memory_stats": self.memory.get_stats()
            }

            return answer, stats

        return "[Ошибка] Неожиданный формат ответа", token_data

    def get_memory_stats(self) -> Dict:
        """Получение статистики памяти"""
        return self.memory.get_stats()

    def clear_memory(self, levels: List[str] = None):
        """Очистка выбранных уровней памяти"""
        if levels is None:
            levels = ["short_term", "working", "long_term"]

        if "short_term" in levels:
            self.memory.clear_short_term()
        if "working" in levels:
            self.memory.clear_working()
        if "long_term" in levels:
            # Очистка долговременной памяти (осторожно!)
            self.memory.long_term = {}
            self.memory._save_to_db()

        print(f"🧹 Очищены уровни: {', '.join(levels)}")


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

class MemoryTestCLI:
    """CLI для тестирования многоуровневой памяти"""

    def __init__(self, auth_key: str):
        self.agent = MemoryAwareAgent(auth_key, session_id="test_session")

    def _print_help(self):
        print("\n" + "=" * 70)
        print("📖 КОМАНДЫ:")
        print("=" * 70)
        print("  /stats      - показать статистику всех уровней памяти")
        print("  /working    - показать рабочую память")
        print("  /longterm   - показать долговременную память")
        print("  /clear      - очистить краткосрочную память")
        print("  /clear all  - очистить ВСЮ память")
        print("  /help       - справка")
        print("  /exit       - выход")
        print("=" * 70)

    def _print_stats(self):
        stats = self.agent.get_memory_stats()
        print("\n" + "=" * 70)
        print("📊 СТАТИСТИКА ПАМЯТИ")
        print("=" * 70)
        print(f"📝 Краткосрочная память: {stats['short_term_count']} сообщений")
        print(f"🔧 Рабочая память: {stats['working_count']} элементов")
        print(f"💾 Долговременная память: {stats['long_term_count']} элементов")

        print("\n📌 Последние сообщения (краткосрочная):")
        for item in stats['short_term_items']:
            print(f"   {item['role']}: {item['preview']}...")

        print("\n📌 Рабочая память:")
        for key, data in stats['working_items'].items():
            print(f"   {key} ({data['type']}): {data['value']}")

        print("\n📌 Долговременная память (по категориям):")
        for cat, count in stats['long_term_by_category'].items():
            print(f"   {cat}: {count} элементов")
        print("=" * 70)

    def run(self):
        print("\n" + "=" * 70)
        print("🧠 АГЕНТ С МНОГОУРОВНЕВОЙ ПАМЯТЬЮ")
        print("=" * 70)
        print("Уровни памяти:")
        print("  📝 Краткосрочная - текущий диалог (последние 20 сообщений)")
        print("  🔧 Рабочая - данные текущей задачи (бюджет, срок, требования)")
        print("  💾 Долговременная - профиль, предпочтения, важные решения")
        print("=" * 70)
        self._print_help()

        while True:
            try:
                user_input = input("\n👤 Вы: ").strip()
                if not user_input:
                    continue

                if user_input == "/exit":
                    print("👋 До свидания!")
                    break
                elif user_input == "/help":
                    self._print_help()
                elif user_input == "/debug":
                    context = self.agent.memory.get_full_context()
                    print("\n🔍 КОНТЕКСТ, ОТПРАВЛЯЕМЫЙ В API:")
                    for i, msg in enumerate(context):
                        print(f"\n[{i}] {msg['role']}:")
                        print(f"   {msg['content'][:1100]}...")
                elif user_input == "/stats":
                    self._print_stats()
                elif user_input == "/working":
                    working = self.agent.memory.get_all_working()
                    print("\n🔧 РАБОЧАЯ ПАМЯТЬ:")
                    for k, v in working.items():
                        print(f"   {k}: {v}")
                elif user_input == "/longterm":
                    longterm = self.agent.memory.get_all_long_term()
                    print("\n💾 ДОЛГОВРЕМЕННАЯ ПАМЯТЬ:")
                    for k, v in longterm.items():
                        print(f"   {k}: {v}")
                elif user_input == "/test_classify":
                    print("\n🧪 ТЕСТ КЛАССИФИКАЦИИ LLM")
                    test_message = input("Введите тестовое сообщение: ")
                    classifier = LLMMemoryClassifier(self.agent)
                    result = classifier.classify(test_message)
                    print(f"\nРезультат классификации:")
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                elif user_input == "/clear":
                    self.agent.clear_memory(["short_term"])
                    print("🧹 Краткосрочная память очищена")
                elif user_input == "/clear all":
                    self.agent.clear_memory(["short_term", "working", "long_term"])
                    print("🧹 ВСЯ память очищена")
                else:
                    print("🤖 Агент: ", end="", flush=True)
                    answer, stats = self.agent.ask(user_input)
                    print(answer)

                    # Краткая статистика запроса
                    print(
                        f"\n└─ 💰 {stats.get('cost_this', 0):.6f}₽ | 📊 память: {stats.get('memory_stats', {}).get('short_term_count', 0)} сообщений")

            except KeyboardInterrupt:
                print("\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")


def main():
    AUTH_KEY = "Basic <ваш_base64_ключ>"

    if AUTH_KEY == "Basic <ваш_base64_ключ>":
        print("⚠️ Укажите ваш ключ авторизации в AUTH_KEY")
        return

    cli = MemoryTestCLI(AUTH_KEY)
    cli.run()


if __name__ == "__main__":
    main()