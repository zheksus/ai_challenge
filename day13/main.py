import os

import requests
import uuid
import json
import sqlite3
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
# from enum import Enum
from dataclasses import dataclass, field, asdict
# from copy import deepcopy

from day13.profile import ProfileManager
from day13.task_manager import TaskManager
# from day13.task import IntelligentTaskManager


# ============================================================================
# ТОЧНЫЙ ПОДСЧЁТ ТОКЕНОВ
# ============================================================================

class TaskStep:
    """Шаг задачи с критериями завершения"""

    def __init__(self, name: str, description: str = "",
                 completion_criteria: List[str] = None,
                 required_artifacts: List[str] = None):
        self.name = name
        self.description = description or name
        self.completion_criteria = completion_criteria or []
        self.required_artifacts = required_artifacts or []
        self.status = "pending"  # pending, in_progress, completed, blocked
        self.artifacts = {}  # key: artifact_name, value: content

    def is_completed(self, context: Dict) -> Tuple[bool, str]:
        """
        Проверка, завершён ли шаг.
        Возвращает (завершён, причина_если_не_завершён)
        """
        # Проверяем наличие всех required_artifacts в контексте
        missing_artifacts = []
        for artifact in self.required_artifacts:
            if artifact not in context:
                missing_artifacts.append(artifact)

        if missing_artifacts:
            return False, f"Отсутствуют: {', '.join(missing_artifacts)}"

        # Проверяем критерии завершения
        # Они могут быть проверены через LLM или правилами
        return True, "Все критерии выполнены"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "criteria": self.completion_criteria,
            "required_artifacts": self.required_artifacts,
            "artifacts": self.artifacts
        }

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
                         last_n_messages: int = 10,
                         profile_prompt: str = "",
                         task_context: str = "") -> List[Dict[str, str]]:
        """
        Формирование полного контекста для отправки в LLM.
        """
        context = []

        # Объединяем ВСЕ системные сообщения в ОДНО
        system_parts = []

        # 👇 ДОБАВЬТЕ ТАСК-КОНТЕКСТ
        if task_context:
            system_parts.append(task_context)

        # 👇 ДОБАВЬТЕ ПРОФИЛЬ ПЕРВЫМ
        if profile_prompt:
            system_parts.append(profile_prompt)

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
    """Агент с трёхуровневой моделью памяти и умным управлением задачами"""

    def __init__(self, auth_key: str, model: str = "GigaChat", session_id: str = "default"):
        self.auth_key = auth_key
        self.model = model
        self.session_id = session_id
        self._token = None
        self._token_expires_at = None

        # Менеджер памяти
        self.memory = MemoryManager(session_id=session_id)

        # Менеджер профиля
        self.profile_manager = ProfileManager(self)

        # 👇 Менеджер задач (передаём self для вызовов LLM)
        self.task_manager = TaskManager(self)

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

    def ask(self, question: str, temperature: float = 0.7) -> tuple:
        """Отправка вопроса с поддержкой задач"""

        # Проверяем, является ли сообщение командой
        if question.startswith("/"):
            parts = question.split(maxsplit=1)
            command = parts[0]
            args = parts[1] if len(parts) > 1 else ""

            # Обрабатываем команды через TaskManager
            if command in ["/task", "/confirm", "/status", "/reset"]:
                response = self.task_manager.handle_command(command, args)

                # Сохраняем в память
                self.memory.add_to_short_term("user", question, len(question) // 2)
                self.memory.add_to_short_term("assistant", response, len(response) // 2)

                return response, {"task_handled": True}

        # ===== ОБЫЧНОЕ СООБЩЕНИЕ =====

        # Если есть активная задача — обрабатываем через TaskManager
        if not self.task_manager.is_idle():
            response = self.task_manager.handle_message(question)
            if response and not response.startswith("📭"):
                # Сохраняем в память
                self.memory.add_to_short_term("user", question, len(question) // 2)
                self.memory.add_to_short_term("assistant", response, len(response) // 2)
                return response, {"task_handled": True}

        # ===== ОБЫЧНЫЙ ЗАПРОС К LLM =====

        # Сохраняем вопрос в память
        self.memory.add_to_short_term("user", question, len(question) // 2)

        # Получаем профиль и контекст задачи
        profile_prompt = self.profile_manager.get_profile_prompt()
        task_context = self.task_manager.get_context_prompt()

        # Формируем контекст
        context = self.memory.get_full_context(
            include_short_term=True,
            include_working=True,
            include_long_term=True,
            last_n_messages=10,
            profile_prompt=profile_prompt,
            task_context=task_context
        )

        # Вызываем LLM
        response, token_data = self._call_api(context, temperature)

        if "error" in response:
            return f"[Ошибка] {response.get('error', 'Неизвестная ошибка')}", token_data

        if "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["message"]["content"]

            # Сохраняем ответ
            answer_tokens = token_data.get("completion_tokens", len(answer) // 2)
            self.memory.add_to_short_term("assistant", answer, answer_tokens)

            self.total_actual_paid += token_data.get("actual_paid_tokens", 0)
            self.request_count += 1

            stats = {
                "request_num": self.request_count,
                "tokens_used": token_data.get("actual_paid_tokens", 0),
                "cost_this": token_data.get("actual_paid_tokens", 0) * 0.05 / 1000,
                "task_state": self.task_manager.machine.current_state.value
            }

            return answer, stats

        return "[Ошибка] Неожиданный формат ответа", token_data

    def get_memory_stats(self) -> Dict:
        """Получение статистики памяти"""
        return self.memory.get_stats()

    def clear_memory(self, levels: List[str] = None):
        """Очистка выбранных уровней памяти"""
        # ... существующий код ...
        pass



# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

class MemoryTestCLI:
    """CLI для работы с памятью и задачами"""

    def __init__(self, auth_key: str):
        self.agent = MemoryAwareAgent(auth_key)

    def _print_help(self):
        print("\n" + "=" * 70)
        print("📖 ДОСТУПНЫЕ КОМАНДЫ:")
        print("=" * 70)

        print("\n📌 УПРАВЛЕНИЕ ЗАДАЧАМИ:")
        print("  /task <описание>   - начать новую задачу")
        print("  /confirm           - подтвердить решение и перейти к следующей фазе")
        print("  /status            - показать статус задачи")
        print("  /reset             - сбросить текущую задачу")

        print("\n📌 УПРАВЛЕНИЕ ПАМЯТЬЮ:")
        print("  /stats             - показать статистику всех уровней памяти")
        print("  /working           - показать рабочую память (текущая задача)")
        print("  /longterm          - показать долговременную память (профиль, история)")
        print("  /clear             - очистить краткосрочную память")
        print("  /clear working     - очистить рабочую память")
        print("  /clear longterm    - очистить долговременную память")
        print("  /clear all         - очистить ВСЮ память")

        print("\n📌 УПРАВЛЕНИЕ ПРОФИЛЕМ:")
        print("  /profile           - показать текущий профиль")
        print("  /profile edit      - редактировать профиль")
        print("  /profile reset     - сбросить профиль (опросник заново)")

        print("\n📌 ОТЛАДКА:")
        print("  /debug             - показать контекст, отправляемый в API")
        print("  /help              - показать эту справку")
        print("  /exit              - выход")

        print("\n💡 На любой фазе задачи вы можете:")
        print("  - Сказать 'сгенерируй план/код' или 'проверь решение'")
        print("  - Сказать 'план: ...' или 'код: ...'")
        print("  - Задать уточняющий вопрос")
        print("=" * 70 + "\n")

    def _print_stats(self):
        """Показать статистику памяти"""
        stats = self.agent.get_memory_stats()
        print("\n" + "=" * 70)
        print("📊 СТАТИСТИКА ПАМЯТИ")
        print("=" * 70)
        print(f"📝 Краткосрочная память: {stats['short_term_count']} сообщений")
        print(f"🔧 Рабочая память: {stats['working_count']} элементов")
        print(f"💾 Долговременная память: {stats['long_term_count']} элементов")

        print("\n📌 Последние сообщения (краткосрочная):")
        for item in stats['short_term_items'][-5:]:
            print(f"   {item['role']}: {item['preview']}...")

        print("\n📌 Рабочая память:")
        for key, data in stats['working_items'].items():
            print(f"   {key} ({data['type']}): {data['value']}")

        print("\n📌 Долговременная память (по категориям):")
        for cat, count in stats['long_term_by_category'].items():
            print(f"   {cat}: {count} элементов")
        print("=" * 70 + "\n")

    def _print_working(self):
        """Показать рабочую память"""
        working = self.agent.memory.get_all_working()
        print("\n" + "=" * 70)
        print("🔧 РАБОЧАЯ ПАМЯТЬ (данные текущей задачи)")
        print("=" * 70)
        if not working:
            print("📭 Рабочая память пуста")
        else:
            for key, value in working.items():
                if isinstance(value, list):
                    print(f"   {key}: {', '.join(str(v) for v in value)}")
                else:
                    print(f"   {key}: {value}")
        print("=" * 70 + "\n")

    def _print_longterm(self):
        """Показать долговременную память"""
        longterm = self.agent.memory.get_all_long_term()
        print("\n" + "=" * 70)
        print("💾 ДОЛГОВРЕМЕННАЯ ПАМЯТЬ")
        print("=" * 70)
        if not longterm:
            print("📭 Долговременная память пуста")
        else:
            # Группируем по категориям
            categories = {}
            for key, value in longterm.items():
                if ":" in key:
                    category, name = key.split(":", 1)
                    if category not in categories:
                        categories[category] = {}
                    categories[category][name] = value

            for category, items in categories.items():
                print(f"\n📂 {category.upper()}:")
                for name, value in items.items():
                    if isinstance(value, dict) or isinstance(value, list):
                        print(f"   {name}: {json.dumps(value, ensure_ascii=False)[:100]}...")
                    else:
                        print(f"   {name}: {value}")
        print("=" * 70 + "\n")

    def _print_profile(self):
        """Показать профиль пользователя"""
        profile = self.agent.profile_manager.profile
        print("\n" + "=" * 70)
        print("👤 ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ")
        print("=" * 70)
        print(self.agent.profile_manager.show_profile())
        print("=" * 70 + "\n")

    def _edit_profile(self):
        """Редактирование профиля"""
        print("\n📝 Редактирование профиля (Enter для пропуска):")
        profile = self.agent.profile_manager.profile

        name = input(f"👤 Имя [{profile.name}]: ").strip()
        if name:
            profile.name = name

        profession = input(f"💼 Профессия [{profile.profession}]: ").strip()
        if profession:
            profile.profession = profession

        company = input(f"🏢 Компания [{profile.company}]: ").strip()
        if company:
            profile.company = company

        style = input(f"🎨 Стиль общения [{profile.communication_style}]: ").strip()
        if style:
            profile.communication_style = style

        fmt = input(f"📝 Формат ответов [{profile.response_format}]: ").strip()
        if fmt:
            profile.response_format = fmt

        tone = input(f"🎭 Тон [{profile.tone}]: ").strip()
        if tone:
            profile.tone = tone

        detail = input(f"📊 Уровень детализации [{profile.detail_level}]: ").strip()
        if detail:
            profile.detail_level = detail

        self.agent.profile_manager._save_profile(profile)
        print("✅ Профиль обновлён!")

    def _reset_profile(self):
        """Сброс профиля"""
        confirm = input("⚠️ Удалить текущий профиль и создать новый? (да/нет): ").strip().lower()
        if confirm in ["да", "yes", "y"]:
            if os.path.exists("PROFILE.md"):
                os.remove("PROFILE.md")
            self.agent.profile_manager = ProfileManager(self.agent)
            print("✅ Профиль сброшен и создан заново")

    def _clear_memory(self, target: str):
        """Очистка памяти"""
        if target == "short_term" or target == "":
            self.agent.clear_memory(["short_term"])
            print("🧹 Краткосрочная память очищена")
        elif target == "working":
            self.agent.clear_memory(["working"])
            print("🧹 Рабочая память очищена")
        elif target == "longterm":
            self.agent.clear_memory(["long_term"])
            print("🧹 Долговременная память очищена")
        elif target == "all":
            self.agent.clear_memory(["short_term", "working", "long_term"])
            print("🧹 ВСЯ память очищена")
        else:
            print(f"❌ Неизвестная цель: {target}. Используйте: short_term, working, longterm, all")

    def _debug_context(self):
        """Показать контекст, отправляемый в API"""
        task_context = self.agent.task_manager.get_context_prompt()
        context = self.agent.memory.get_full_context(
            include_short_term=True,
            include_working=True,
            include_long_term=True,
            last_n_messages=10,
            profile_prompt=self.agent.profile_manager.get_profile_prompt(),
            task_context=task_context
        )

        print("\n" + "=" * 70)
        print("🔍 КОНТЕКСТ, ОТПРАВЛЯЕМЫЙ В API")
        print("=" * 70)
        print(f"📊 Всего сообщений: {len(context)}")
        print("-" * 70)

        for i, msg in enumerate(context):
            role = msg["role"].upper()
            content_preview = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
            print(f"\n[{i}] {role}:")
            print(f"   {content_preview}")

        print("=" * 70 + "\n")

    def run(self):
        """Запуск CLI"""
        print("\n" + "=" * 70)
        print("🤖 АГЕНТ С ПАМЯТЬЮ И УПРАВЛЕНИЕМ ЗАДАЧАМИ")
        print("=" * 70)

        # Показываем статус профиля
        if self.agent.profile_manager.profile.is_initialized:
            print(f"👤 Профиль: {self.agent.profile_manager.profile.name or 'Не указан'}")
        else:
            print("⚠️ Профиль не заполнен. Используйте /profile для настройки.")

        print("📌 Введите /help для списка команд")
        print("=" * 70 + "\n")

        while True:
            try:
                user_input = input("👤 Вы: ").strip()
                if not user_input:
                    continue

                # ===== ВЫХОД =====
                if user_input == "/exit":
                    print("👋 До свидания!")
                    break

                # ===== ПОМОЩЬ =====
                if user_input == "/help":
                    self._print_help()
                    continue

                # ===== УПРАВЛЕНИЕ ЗАДАЧАМИ =====
                if user_input.startswith("/task"):
                    parts = user_input.split(maxsplit=1)
                    description = parts[1] if len(parts) > 1 else ""
                    response = self.agent.task_manager.handle_command("/task", description)
                    print(f"\n🤖 Агент: {response}")
                    continue

                if user_input == "/confirm":
                    response = self.agent.task_manager.handle_command("/confirm", "")
                    print(f"\n🤖 Агент: {response}")
                    continue

                if user_input == "/status":
                    response = self.agent.task_manager.handle_command("/status", "")
                    print(f"\n{response}")
                    continue

                if user_input == "/reset":
                    response = self.agent.task_manager.handle_command("/reset", "")
                    print(f"\n🤖 Агент: {response}")
                    continue

                # ===== УПРАВЛЕНИЕ ПАМЯТЬЮ =====
                if user_input == "/stats":
                    self._print_stats()
                    continue

                if user_input == "/working":
                    self._print_working()
                    continue

                if user_input == "/longterm":
                    self._print_longterm()
                    continue

                if user_input.startswith("/clear"):
                    parts = user_input.split()
                    target = parts[1] if len(parts) > 1 else ""
                    self._clear_memory(target)
                    continue

                # ===== УПРАВЛЕНИЕ ПРОФИЛЕМ =====
                if user_input == "/profile":
                    self._print_profile()
                    continue

                if user_input == "/profile edit":
                    self._edit_profile()
                    continue

                if user_input == "/profile reset":
                    self._reset_profile()
                    continue

                # ===== ОТЛАДКА =====
                if user_input == "/debug":
                    self._debug_context()
                    continue

                # ===== ОБЫЧНОЕ СООБЩЕНИЕ =====
                response, stats = self.agent.ask(user_input)

                # Выводим ответ
                print(f"\n🤖 Агент: {response}")

                # Показываем статус задачи если есть
                task_state = stats.get("task_state", "idle")
                if task_state != "idle":
                    print(f"\n└─ 📊 Этап: {task_state.upper()}")

                # Показываем статистику токенов если есть
                if stats.get("tokens_used", 0) > 0:
                    print(f"└─ 💰 {stats.get('cost_this', 0):.6f}₽ | {stats.get('tokens_used', 0)} токенов")

            except KeyboardInterrupt:
                print("\n👋 До свидания!")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")


def main():
    AUTH_KEY = "Basic <ваш_base64_ключ>"

    if AUTH_KEY == "Basic <ваш_base64_ключ>":
        print("⚠️ Укажите ваш ключ авторизации в AUTH_KEY")
        return

    cli = MemoryTestCLI(AUTH_KEY)
    cli.run()


if __name__ == "__main__":
    main()