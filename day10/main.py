import requests
import uuid
import json
import copy
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


# ============================================================================
# ТОЧНЫЙ ПОДСЧЁТ ТОКЕНОВ
# ============================================================================

class TokenCounter:
    """Извлечение точных данных о токенах из ответа API"""

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
# СТРАТЕГИИ УПРАВЛЕНИЯ КОНТЕКСТОМ
# ============================================================================

class ContextStrategy(Enum):
    SLIDING_WINDOW = "sliding_window"
    STICKY_FACTS = "sticky_facts"
    BRANCHING = "branching"


class SlidingWindowStrategy:
    """
    Стратегия 1: Sliding Window
    Хранит только последние N сообщений, остальное отбрасывает.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages: List[Dict[str, str]] = []
        self.discarded_count = 0

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

        # Если превысили размер окна - удаляем самое старое
        if len(self.messages) > self.window_size:
            removed = self.messages.pop(0)
            self.discarded_count += 1

    def get_context(self) -> List[Dict[str, str]]:
        """Возвращает контекст для запроса"""
        return self.messages.copy()

    def clear(self):
        self.messages = []
        self.discarded_count = 0

    def get_stats(self) -> Dict:
        return {
            "strategy": "Sliding Window",
            "window_size": self.window_size,
            "current_messages": len(self.messages),
            "discarded_messages": self.discarded_count
        }


class StickyFactsStrategy:
    """
    Стратегия 2: Sticky Facts / Key-Value Memory
    Хранит отдельный блок фактов (ключ-значение), который обновляется после каждого сообщения.
    В запрос отправляет: facts + последние N сообщений.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.messages: List[Dict[str, str]] = []
        self.facts: Dict[str, Any] = {}
        self.fact_updates_log: List[str] = []

    def _extract_facts_from_message(self, content: str) -> Dict[str, Any]:
        """
        Извлекает ключевые факты из сообщения пользователя.
        В реальности здесь мог бы быть LLM-парсер, но для демо используем правила.
        """
        facts = {}
        content_lower = content.lower()

        # Имя
        if "меня зовут" in content_lower:
            import re
            match = re.search(r'зовут\s+([А-Я][а-я]+(?:\s+[А-Я][а-я]+)?)', content)
            if match:
                facts["name"] = match.group(1)

        # Профессия
        if "работаю" in content_lower:
            import re
            match = re.search(r'работаю\s+([А-Яа-я]+\s*[А-Яа-я]*)', content)
            if match:
                facts["profession"] = match.group(1)

        # Компания
        if "компани" in content_lower:
            import re
            match = re.search(r'компани(?:и|я|ей)?\s+["\']?([А-Яа-я0-9\s]+?)[;"\']?', content)
            if match:
                facts["company"] = match.group(1).strip()

        # Бюджет
        if "бюджет" in content_lower or "руб" in content_lower:
            import re
            match = re.search(r'(\d+(?:\.?\d+)?)\s*(?:млн|тыс|руб)', content)
            if match:
                facts["budget"] = match.group(1)

        # Срок
        if "срок" in content_lower or "месяц" in content_lower or "недель" in content_lower:
            import re
            match = re.search(r'(\d+)\s*(?:месяц|недель|дней)', content)
            if match:
                facts["deadline"] = match.group(1)

        # Желания/предпочтения
        if "хочу" in content_lower or "нужно" in content_lower:
            for keyword in ["интеграция", "api", "чат-бот", "нейросеть", "аналитика"]:
                if keyword in content_lower:
                    if "requirements" not in facts:
                        facts["requirements"] = []
                    facts["requirements"].append(keyword)

        return facts

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

        # Если сообщение от пользователя - извлекаем и обновляем факты
        if role == "user":
            new_facts = self._extract_facts_from_message(content)
            for key, value in new_facts.items():
                old_value = self.facts.get(key)
                self.facts[key] = value
                if old_value and old_value != value:
                    self.fact_updates_log.append(f"{key}: {old_value} -> {value}")
                else:
                    self.fact_updates_log.append(f"{key}: {value}")

        # Ограничиваем окно сообщений
        if len(self.messages) > self.window_size:
            self.messages.pop(0)

    def get_context(self) -> List[Dict[str, str]]:
        """Возвращает контекст: факты (как системное сообщение) + последние сообщения"""
        context = []

        # Добавляем факты как системное сообщение
        if self.facts:
            facts_text = "Извлечённые факты из диалога:\n" + "\n".join([f"- {k}: {v}" for k, v in self.facts.items()])
            context.append({"role": "system", "content": facts_text})

        # Добавляем последние сообщения
        context.extend(self.messages)

        return context

    def clear(self):
        self.messages = []
        self.facts = {}
        self.fact_updates_log = []

    def get_stats(self) -> Dict:
        return {
            "strategy": "Sticky Facts",
            "window_size": self.window_size,
            "current_messages": len(self.messages),
            "facts_count": len(self.facts),
            "facts": self.facts.copy(),
            "updates_count": len(self.fact_updates_log)
        }


class BranchPoint:
    """Точка ветвления диалога"""

    def __init__(self, name: str, messages: List[Dict[str, str]], facts: Dict = None):
        self.name = name
        self.messages = copy.deepcopy(messages)
        self.facts = copy.deepcopy(facts) if facts else {}


class BranchingStrategy:
    """
    Стратегия 3: Branching (ветки диалога)
    Позволяет сохранять checkpoint, создавать ветки и переключаться между ними.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.current_branch: str = "main"
        self.branches: Dict[str, Dict] = {
            "main": {
                "messages": [],
                "facts": {},
                "created_at": datetime.now().isoformat()
            }
        }
        self.checkpoints: Dict[str, BranchPoint] = {}
        self.branch_history: List[str] = []

    def add_message(self, role: str, content: str):
        """Добавляет сообщение в текущую ветку"""
        branch = self.branches[self.current_branch]
        branch["messages"].append({"role": role, "content": content})

        # Извлекаем факты (упрощённо)
        if role == "user":
            new_facts = self._extract_facts(content)
            for k, v in new_facts.items():
                branch["facts"][k] = v

        # Ограничиваем размер
        if len(branch["messages"]) > self.window_size:
            branch["messages"].pop(0)

    def _extract_facts(self, content: str) -> Dict:
        """Извлечение фактов (упрощённо)"""
        facts = {}
        content_lower = content.lower()

        if "меня зовут" in content_lower:
            import re
            match = re.search(r'зовут\s+([А-Я][а-я]+)', content)
            if match:
                facts["name"] = match.group(1)
        if "бюджет" in content_lower:
            import re
            match = re.search(r'(\d+(?:\.?\d+)?)\s*(?:млн|тыс)', content)
            if match:
                facts["budget"] = match.group(1)

        return facts

    def create_checkpoint(self, name: str):
        """Создаёт checkpoint текущего состояния"""
        branch = self.branches[self.current_branch]
        self.checkpoints[name] = BranchPoint(
            name=name,
            messages=branch["messages"].copy(),
            facts=branch["facts"].copy()
        )
        return f"✅ Checkpoint '{name}' создан"

    def create_branch(self, from_checkpoint: str, branch_name: str):
        """Создаёт новую ветку от указанного checkpoint"""
        if from_checkpoint not in self.checkpoints:
            return f"❌ Checkpoint '{from_checkpoint}' не найден"

        checkpoint = self.checkpoints[from_checkpoint]
        self.branches[branch_name] = {
            "messages": checkpoint.messages.copy(),
            "facts": checkpoint.facts.copy(),
            "created_at": datetime.now().isoformat(),
            "parent_checkpoint": from_checkpoint
        }
        self.branch_history.append(f"Создана ветка '{branch_name}' от '{from_checkpoint}'")
        return f"✅ Ветка '{branch_name}' создана от checkpoint '{from_checkpoint}'"

    def switch_branch(self, branch_name: str):
        """Переключается на другую ветку"""
        if branch_name not in self.branches:
            return f"❌ Ветка '{branch_name}' не найдена. Доступны: {list(self.branches.keys())}"

        self.current_branch = branch_name
        return f"🔄 Переключено на ветку '{branch_name}'"

    def get_context(self) -> List[Dict[str, str]]:
        """Возвращает контекст текущей ветки"""
        branch = self.branches[self.current_branch]

        context = []

        # Добавляем факты как системное сообщение
        if branch["facts"]:
            facts_text = "Известные факты:\n" + "\n".join([f"- {k}: {v}" for k, v in branch["facts"].items()])
            context.append({"role": "system", "content": facts_text})

        # Добавляем сообщения
        context.extend(branch["messages"])

        return context

    def get_available_branches(self) -> List[str]:
        return list(self.branches.keys())

    def clear(self):
        self.branches = {"main": {"messages": [], "facts": {}, "created_at": datetime.now().isoformat()}}
        self.current_branch = "main"
        self.checkpoints = {}
        self.branch_history = []

    def get_stats(self) -> Dict:
        branch = self.branches[self.current_branch]
        return {
            "strategy": "Branching",
            "current_branch": self.current_branch,
            "available_branches": list(self.branches.keys()),
            "checkpoints_count": len(self.checkpoints),
            "current_messages": len(branch["messages"]),
            "facts_count": len(branch["facts"]),
            "branch_history": self.branch_history[-5:]
        }


# ============================================================================
# АГЕНТ С ПЕРЕКЛЮЧАЕМЫМИ СТРАТЕГИЯМИ
# ============================================================================

class GigaChatAgentWithStrategies:
    """Агент GigaChat с поддержкой трёх стратегий управления контекстом"""

    def __init__(self, auth_key: str, model: str = "GigaChat"):
        self.auth_key = auth_key
        self.model = model
        self._token = None
        self._token_expires_at = None

        # Текущая стратегия
        self.current_strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW
        self.sliding_window = SlidingWindowStrategy(window_size=8)
        self.sticky_facts = StickyFactsStrategy(window_size=5)
        self.branching = BranchingStrategy(window_size=20)

        # Статистика токенов
        self.total_actual_paid = 0
        self.request_count = 0
        self.strategy_stats: Dict[str, List] = {"sliding_window": [], "sticky_facts": [], "branching": []}

        # Полная история (для отладки)
        self.full_history: List[Dict[str, str]] = []

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
        """Вызов API с получением статистики токенов"""
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

    def switch_strategy(self, strategy: ContextStrategy):
        """Переключение стратегии"""
        old = self.current_strategy
        self.current_strategy = strategy
        return f"🔄 Стратегия изменена: {old.value} → {strategy.value}"

    def ask(self, question: str, temperature: float = 0.7) -> Tuple[str, Dict]:
        """Отправка вопроса с текущей стратегией"""

        # Получаем контекст в зависимости от стратегии
        if self.current_strategy == ContextStrategy.SLIDING_WINDOW:
            self.sliding_window.add_message("user", question)
            context = self.sliding_window.get_context()
        elif self.current_strategy == ContextStrategy.STICKY_FACTS:
            self.sticky_facts.add_message("user", question)
            context = self.sticky_facts.get_context()
        else:  # BRANCHING
            self.branching.add_message("user", question)
            context = self.branching.get_context()

        # Вызываем API
        response, token_data = self._call_api(context, temperature)

        if "error" in response:
            return f"[Ошибка] {response.get('error', 'Неизвестная ошибка')}", token_data

        if "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["message"]["content"]

            # Сохраняем ответ в стратегию
            if self.current_strategy == ContextStrategy.SLIDING_WINDOW:
                self.sliding_window.add_message("assistant", answer)
            elif self.current_strategy == ContextStrategy.STICKY_FACTS:
                self.sticky_facts.add_message("assistant", answer)
            else:
                self.branching.add_message("assistant", answer)

            # Обновляем статистику
            self.full_history.append({"role": "user", "content": question})
            self.full_history.append({"role": "assistant", "content": answer})
            self.total_actual_paid += token_data.get("actual_paid_tokens", 0)
            self.request_count += 1

            # Сохраняем статистику по стратегии
            if self.current_strategy == ContextStrategy.SLIDING_WINDOW:
                strategy_stats = self.sliding_window.get_stats()
            elif self.current_strategy == ContextStrategy.STICKY_FACTS:
                strategy_stats = self.sticky_facts.get_stats()
            else:
                strategy_stats = self.branching.get_stats()

            stats = {
                "request_num": self.request_count,
                "prompt_tokens": token_data.get("prompt_tokens", 0),
                "completion_tokens": token_data.get("completion_tokens", 0),
                "total_tokens": token_data.get("total_tokens", 0),
                "precached_tokens": token_data.get("precached_tokens", 0),
                "actual_paid_tokens": token_data.get("actual_paid_tokens", 0),
                "cost_this": TokenCounter.calculate_cost(token_data),
                "cumulative_paid": self.total_actual_paid,
                "cumulative_cost": self.total_actual_paid * 0.05 / 1000,
                "strategy": self.current_strategy.value,
                "strategy_stats": strategy_stats
            }

            return answer, stats

        return "[Ошибка] Неожиданный формат ответа", token_data

    def create_checkpoint(self, name: str):
        """Создание checkpoint (только для стратегии Branching)"""
        if self.current_strategy == ContextStrategy.BRANCHING:
            return self.branching.create_checkpoint(name)
        return "⚠️ Checkpoint доступны только в стратегии Branching"

    def create_branch(self, checkpoint: str, branch_name: str):
        """Создание ветки (только для стратегии Branching)"""
        if self.current_strategy == ContextStrategy.BRANCHING:
            return self.branching.create_branch(checkpoint, branch_name)
        return "⚠️ Ветвление доступно только в стратегии Branching"

    def switch_branch(self, branch_name: str):
        """Переключение ветки (только для стратегии Branching)"""
        if self.current_strategy == ContextStrategy.BRANCHING:
            return self.branching.switch_branch(branch_name)
        return "⚠️ Ветвление доступно только в стратегии Branching"

    def get_strategy_stats(self) -> Dict:
        if self.current_strategy == ContextStrategy.SLIDING_WINDOW:
            return self.sliding_window.get_stats()
        elif self.current_strategy == ContextStrategy.STICKY_FACTS:
            return self.sticky_facts.get_stats()
        else:
            return self.branching.get_stats()

    def clear(self):
        self.sliding_window.clear()
        self.sticky_facts.clear()
        self.branching.clear()
        self.full_history = []
        self.total_actual_paid = 0
        self.request_count = 0


# ============================================================================
# ТЕСТИРОВАНИЕ: СЦЕНАРИЙ СБОРА ТЗ (10-15 сообщений)
# ============================================================================

class StrategyTester:
    """Тестировщик стратегий на одном сценарии"""

    def __init__(self, auth_key: str):
        self.auth_key = auth_key

    def get_tz_scenario(self) -> List[str]:
        """Сценарий сбора технического задания (15 сообщений)"""
        return [
            "Здравствуйте! Мы хотим разработать чат-бота для службы поддержки интернет-магазина.",
            "Какой у вас бюджет? Ориентировочно 500 тысяч рублей.",
            "Какие основные функции должны быть у бота? Нужна интеграция с CRM, ответы на частые вопросы, возможность передать оператору.",
            "Какой срок реализации? Нужно уложиться в 2 месяца.",
            "А какая технологическая платформа предпочтительна? Мы используем Bitrix24.",
            "Какое количество пользователей планируется? Около 1000 в день.",
            "Нужна ли аналитика по диалогам? Да, очень важна аналитика: количество обращений, темы, оценка качества.",
            "А какой язык программирования предпочтителен? У нас Python-команда.",
            "Поддержка каких языков нужна? Только русский, но в будущем возможно английский.",
            "Какой канал связи? Telegram, сайт и ВКонтакте.",
            "Есть ли требования к безопасности? Да, нужна авторизация пользователей и шифрование данных.",
            "Какой формат ответов? Бот должен отвечать текстом, кнопками и иногда картинками.",
            "Нужна ли возможность обучения бота? Да, чтобы операторы могли пополнять базу ответов.",
            "Какой у вас контакт для связи? Меня зовут Дмитрий, почта dmitry@example.com",
            "Когда планируете старт? Хотели бы начать через 2 недели."
        ]

    def evaluate_response_quality(self, answer: str, question_index: int) -> Dict:
        """Оценка качества ответа"""
        quality = {
            "answers_specific_numbers": 0,
            "remembers_name": 0,
            "remembers_platform": 0,
            "remembers_budget": 0,
            "remembers_deadline": 0,
            "mentions_crm": 0,
            "mentions_security": 0,
            "total_score": 0
        }

        if "500" in answer or "пятьсот" in answer:
            quality["remembers_budget"] = 1
        if "2 месяца" in answer or "два месяца" in answer:
            quality["remembers_deadline"] = 1
        if "Bitrix" in answer:
            quality["remembers_platform"] = 1
        if "Дмитрий" in answer:
            quality["remembers_name"] = 1
        if "CRM" in answer or "интеграц" in answer:
            quality["mentions_crm"] = 1
        if "шифров" in answer or "безопасн" in answer:
            quality["mentions_security"] = 1

        quality["total_score"] = sum([v for k, v in quality.items() if k != "total_score"])
        return quality

    def run_test(self, strategy: ContextStrategy) -> Dict:
        """Запуск теста для одной стратегии"""
        print(f"\n{'=' * 70}")
        print(f"🧪 ТЕСТИРОВАНИЕ: {strategy.value.upper()}")
        print(f"{'=' * 70}")

        agent = GigaChatAgentWithStrategies(self.auth_key)
        agent.switch_strategy(strategy)

        scenario = self.get_tz_scenario()
        results = []
        total_tokens = 0
        total_cost = 0

        for i, question in enumerate(scenario):
            print(f"\n📝 Шаг {i + 1}: {question[:80]}...")
            answer, stats = agent.ask(question)
            print(f"🤖 Ответ: {answer[:150]}...")
            print(f"💰 Токенов: {stats.get('actual_paid_tokens', 0)} | Стоимость: {stats.get('cost_this', 0):.6f}₽")

            quality = self.evaluate_response_quality(answer, i)

            results.append({
                "step": i + 1,
                "question": question,
                "answer_preview": answer[:200],
                "tokens": stats.get("actual_paid_tokens", 0),
                "cost": stats.get("cost_this", 0),
                "quality": quality
            })

            total_tokens += stats.get("actual_paid_tokens", 0)
            total_cost += stats.get("cost_this", 0)

        # Финальная проверка памяти (на 15-м шаге)
        print(f"\n{'=' * 50}")
        print("🔍 ФИНАЛЬНАЯ ПРОВЕРКА ПАМЯТИ")
        test_questions = [
            "Напомните, как меня зовут?",
            "Какой у нас бюджет на проект?",
            "Какой срок реализации мы обсуждали?",
            "Какая CRM используется?",
            "Какие требования к безопасности были указаны?"
        ]

        memory_results = []
        for q in test_questions:
            answer, stats = agent.ask(q)
            quality = self.evaluate_response_quality(answer, -1)
            memory_results.append({
                "question": q,
                "answer": answer[:100],
                "remembered": quality["total_score"] > 0
            })
            print(f"📌 {q}")
            print(f"   {'✅' if quality['total_score'] > 0 else '❌'} {answer[:80]}...")

        final_stats = agent.get_strategy_stats()

        return {
            "strategy": strategy.value,
            "steps": results,
            "memory_test": memory_results,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "strategy_details": final_stats,
            "overall_quality": sum(r["quality"]["total_score"] for r in results) / len(results)
        }

    def run_full_comparison(self):
        """Запуск сравнения всех трёх стратегий"""
        print("\n" + "=" * 80)
        print("🚀 ЗАПУСК СРАВНЕНИЯ СТРАТЕГИЙ УПРАВЛЕНИЯ КОНТЕКСТОМ")
        print("=" * 80)
        print("Сценарий: Сбор технического задания (15 сообщений)")
        print("=" * 80)

        results = {}

        # Тест Sliding Window
        results["sliding_window"] = self.run_test(ContextStrategy.SLIDING_WINDOW)

        # Тест Sticky Facts
        results["sticky_facts"] = self.run_test(ContextStrategy.STICKY_FACTS)

        # Тест Branching
        # results["branching"] = self.run_test(ContextStrategy.BRANCHING)
        results["branching"] = self.run_branching_test()  # <-- ИЗМЕНЕНИЕ

        # Вывод сравнительного анализа
        self.print_comparison(results)

        return results

    def print_comparison(self, results: Dict):
        """Вывод сравнительного анализа"""
        print("\n" + "=" * 80)
        print("📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ СТРАТЕГИЙ")
        print("=" * 80)

        print("\n📈 КОЛИЧЕСТВЕННЫЕ ПОКАЗАТЕЛИ:")
        print("-" * 60)
        print(f"{'Стратегия':<20} {'Токенов':<15} {'Стоимость ₽':<15} {'Качество':<15}")
        print("-" * 60)

        for name, data in results.items():
            print(
                f"{name:<20} {data['total_tokens']:<15,} {data['total_cost']:<15.4f} {data['overall_quality']:<15.1f}")

        print("\n🎯 КАЧЕСТВО ПАМЯТИ (после 15 сообщений):")
        print("-" * 60)

        memory_questions = [
            "Как зовут заказчика?",
            "Бюджет проекта",
            "Срок реализации",
            "CRM система",
            "Требования безопасности"
        ]

        print(f"{'Стратегия':<20}", end="")
        for q in memory_questions[:3]:
            print(f"{q[:12]:<15}", end="")
        print()
        print("-" * 60)

        for name, data in results.items():
            print(f"{name:<20}", end="")
            remembered = [mr["remembered"] for mr in data["memory_test"]]
            for r in remembered[:3]:
                print(f"{'✅' if r else '❌':<15}", end="")
            print()

        print("\n💡 ВЫВОДЫ И РЕКОМЕНДАЦИИ:")
        print("-" * 60)

        # Сравнение
        best_quality = max(results.items(), key=lambda x: x[1]["overall_quality"])
        best_tokens = min(results.items(), key=lambda x: x[1]["total_tokens"])

        print(f"\n🏆 Лучшее качество: {best_quality[0]} (оценка: {best_quality[1]['overall_quality']:.1f})")
        print(f"💰 Самая низкая стоимость: {best_tokens[0]} ({best_tokens[1]['total_cost']:.4f} ₽)")



        print(f"\n📊 ИТОГОВАЯ ДИАГРАММА РАСХОДА ТОКЕНОВ:")
        max_tokens = max([d['total_tokens'] for d in results.values()])
        for name, data in results.items():
            bar_len = int(data['total_tokens'] / max_tokens * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"   {name:<15}: {bar} {data['total_tokens']:,} токенов")


    def run_branching_test(self) -> Dict:
        """Тест стратегии ветвления - создаём две ветки от checkpoint после 4 сообщений"""
        print(f"\n{'=' * 70}")
        print(f"🧪 ТЕСТИРОВАНИЕ: BRANCHING (реальное ветвление)")
        print(f"{'=' * 70}")

        agent = GigaChatAgentWithStrategies(self.auth_key)
        agent.switch_strategy(ContextStrategy.BRANCHING)

        scenario = self.get_tz_scenario()
        results = {
            "branch_main": [],
            "branch_a": [],
            "branch_b": [],
            "checkpoint_name": "decision_point"
        }

        # ========== ЧАСТЬ 1: ОБЩИЙ ДИАЛОГ (первые 4 сообщения) ==========
        print("\n📌 ОБЩИЙ ДИАЛОГ (первые 4 сообщения):")
        print("-" * 50)

        for i in range(4):
            print(f"\n📝 Шаг {i + 1}: {scenario[i]}")
            answer, stats = agent.ask(scenario[i])
            print(f"🤖 {answer[:200]}...")
            results["branch_main"].append({
                "step": i + 1,
                "question": scenario[i],
                "answer": answer[:200]
            })

        # Создаём checkpoint после 4 сообщений
        print("\n" + "=" * 50)
        checkpoint_result = agent.create_checkpoint("decision_point")
        print(f"📌 {checkpoint_result}")

        # Сохраняем текущие факты/состояние для отладки
        current_stats = agent.get_strategy_stats()
        print(
            f"📊 Текущее состояние: {current_stats.get('facts', current_stats.get('current_messages', 0))} фактов/сообщений")

        # ========== ЧАСТЬ 2: ВЕТКА A - ПРЕМИУМ ВАРИАНТ ==========
        print("\n" + "=" * 50)
        print("🌿 ВЕТКА A: Премиум решение (большой бюджет, быстрый срок)")
        print("=" * 50)

        # Создаём и переключаемся на ветку A
        agent.switch_branch("main")
        agent.create_branch("decision_point", "premium_branch")
        agent.switch_branch("premium_branch")

        premium_questions = [
            "А что если мы увеличим бюджет до 1.2 миллионов рублей? Сможете ускорить разработку до 1 месяца?",
            "Отлично! Тогда добавьте ещё интеграцию с 1С и систему аналитики PowerBI.",
            "Какую скидку вы можете предложить при бюджете 1.5 миллиона?",
            "И последнее: нужен ли будет дополнительный техподдержка после запуска?"
        ]

        for i, q in enumerate(premium_questions):
            print(f"\n📝 Ветка A (премиум) шаг {i + 1}: {q[:80]}...")
            answer, stats = agent.ask(q)
            print(f"🤖 {answer[:200]}...")
            results["branch_a"].append({
                "step": i + 1,
                "question": q,
                "answer": answer[:200],
                "tokens": stats.get("actual_paid_tokens", 0)
            })

        # ========== ЧАСТЬ 3: ВЕТКА B - ЭКОНОМ ВАРИАНТ ==========
        print("\n" + "=" * 50)
        print("🌿 ВЕТКА B: Эконом решение (малый бюджет, длительный срок)")
        print("=" * 50)

        # Создаём и переключаемся на ветку B
        agent.switch_branch("main")
        agent.create_branch("decision_point", "economy_branch")
        agent.switch_branch("economy_branch")

        economy_questions = [
            "А если уменьшить бюджет до 300 тысяч рублей? Какие функции придётся урезать?",
            "Понял. Сколько времени займёт разработка при таком бюджете?",
            "Можем ли мы сделать MVP сначала, а остальное добавить позже?",
            "Какой минимальный бюджет для базовой версии без интеграций?"
        ]

        for i, q in enumerate(economy_questions):
            print(f"\n📝 Ветка B (эконом) шаг {i + 1}: {q[:80]}...")
            answer, stats = agent.ask(q)
            print(f"🤖 {answer[:200]}...")
            results["branch_b"].append({
                "step": i + 1,
                "question": q,
                "answer": answer[:200],
                "tokens": stats.get("actual_paid_tokens", 0)
            })

        # ========== ЧАСТЬ 4: ПРОВЕРКА НЕЗАВИСИМОСТИ ВЕТОК ==========
        print("\n" + "=" * 70)
        print("🔄 ПРОВЕРКА НЕЗАВИСИМОСТИ ВЕТОК (переключение и проверка памяти)")
        print("=" * 70)

        memory_checks = []

        # Проверка ветки A
        print("\n📌 Переключаемся на ВЕТКУ A (premium_branch)")
        agent.switch_branch("premium_branch")

        test_questions_a = [
            "Какой у нас бюджет в этой ветке?",
            "Какой срок разработки мы обсуждали?",
            "Какие дополнительные интеграции я просил добавить?"
        ]

        branch_a_memory = []
        for q in test_questions_a:
            answer, stats = agent.ask(q)
            remembered = any([
                "1.2" in answer or "1200000" in answer or "миллион" in answer,
                "месяц" in answer,
                "1С" in answer or "PowerBI" in answer
            ])
            branch_a_memory.append({
                "question": q,
                "answer": answer[:100],
                "remembered": remembered
            })
            print(f"   {'✅' if remembered else '❌'} {q}")
            print(f"      {answer[:100]}...")

        # Проверка ветки B
        print("\n📌 Переключаемся на ВЕТКУ B (economy_branch)")
        agent.switch_branch("economy_branch")

        test_questions_b = [
            "Какой у нас бюджет в этой ветке?",
            "Какие функции мы урезали?",
            "Что такое MVP и когда мы его получим?"
        ]

        branch_b_memory = []
        for q in test_questions_b:
            answer, stats = agent.ask(q)
            remembered = any([
                "300" in answer or "300000" in answer,
                "урез" in answer or "меньше" in answer,
                "MVP" in answer
            ])
            branch_b_memory.append({
                "question": q,
                "answer": answer[:100],
                "remembered": remembered
            })
            print(f"   {'✅' if remembered else '❌'} {q}")
            print(f"      {answer[:100]}...")

        # Проверка, что ветки НЕ путаются
        print("\n📌 Переключаемся обратно на ВЕТКУ A для проверки независимости")
        agent.switch_branch("premium_branch")
        answer, _ = agent.ask("У нас бюджет 300 тысяч или 1.2 миллиона?")

        is_independent = "1.2" in answer or "миллион" in answer
        print(f"   {'✅' if is_independent else '❌'} Ветка A помнит свой бюджет (1.2 млн): {answer[:100]}...")

        agent.switch_branch("economy_branch")
        answer, _ = agent.ask("У нас бюджет 1.2 миллиона или 300 тысяч?")

        is_independent_b = "300" in answer or "триста" in answer
        print(f"   {'✅' if is_independent_b else '❌'} Ветка B помнит свой бюджет (300 тыс): {answer[:100]}...")

        # ========== ЧАСТЬ 5: ФИНАЛЬНАЯ СТАТИСТИКА ==========
        print("\n" + "=" * 70)
        print("📊 ФИНАЛЬНАЯ СТАТИСТИКА ВЕТВЛЕНИЯ")
        print("=" * 70)

        final_stats = agent.branching.get_stats()

        comparison = {
            "branching": {
                "branch_a_messages": len(results["branch_a"]),
                "branch_b_messages": len(results["branch_b"]),
                "branch_a_remembered": sum(1 for m in branch_a_memory if m["remembered"]),
                "branch_b_remembered": sum(1 for m in branch_b_memory if m["remembered"]),
                "is_independent": is_independent and is_independent_b,
                "available_branches": final_stats.get("available_branches", []),
                "checkpoint_used": "decision_point"
            }
        }

        print(f"\n🌿 Результаты ветвления:")
        print(f"   - Ветка A (premium_branch): {len(results['branch_a'])} сообщений")
        print(f"   - Ветка B (economy_branch): {len(results['branch_b'])} сообщений")
        print(f"   - Память ветки A: {comparison['branching']['branch_a_remembered']}/3")
        print(f"   - Память ветки B: {comparison['branching']['branch_b_remembered']}/3")
        print(f"   - Ветки независимы: {'✅ Да' if is_independent and is_independent_b else '❌ Нет'}")

        results["branching_stats"] = comparison["branching"]
        results["branch_a_memory"] = branch_a_memory
        results["branch_b_memory"] = branch_b_memory

        return results

# ============================================================================
# CLI ИНТЕРФЕЙС
# ============================================================================

class InteractiveCLI:
    """Интерактивный CLI для работы с агентом"""

    def __init__(self, auth_key: str):
        self.agent = GigaChatAgentWithStrategies(auth_key)

    def _print_help(self):
        print("\n" + "=" * 70)
        print("📖 КОМАНДЫ:")
        print("=" * 70)
        print("  /strategy [sw|facts|branch]  - переключить стратегию")
        print("  /status                      - показать текущую стратегию и статистику")
        print("  /checkpoint <name>           - создать checkpoint (branching)")
        print("  /branch <from> <to>          - создать ветку (branching)")
        print("  /switch <branch>             - переключить ветку (branching)")
        print("  /clear                       - очистить историю")
        print("  /compare                     - запустить сравнение стратегий")
        print("  /help                        - справка")
        print("  /exit                        - выход")
        print("=" * 70)

    def _print_status(self):
        stats = self.agent.get_strategy_stats()
        print(f"\n📊 Текущая стратегия: {self.agent.current_strategy.value}")
        for k, v in stats.items():
            if k not in ["strategy"]:
                print(f"   {k}: {v}")
        print(f"   Всего запросов: {self.agent.request_count}")
        print(f"   Оплачено токенов: {self.agent.total_actual_paid}")
        print(f"   Общая стоимость: {self.agent.total_actual_paid * 0.05 / 1000:.6f} ₽")

    def run(self):
        print("\n" + "=" * 70)
        print("🤖 АГЕНТ С 3 СТРАТЕГИЯМИ УПРАВЛЕНИЯ КОНТЕКСТОМ")
        print("=" * 70)
        print("Доступные стратегии:")
        print("   1. Sliding Window (sw)  - только последние N сообщений")
        print("   2. Sticky Facts (facts) - факты + последние сообщения")
        print("   3. Branching (branch)   - ветки диалога")
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

                elif user_input == "/status":
                    self._print_status()

                elif user_input.startswith("/strategy"):
                    parts = user_input.split()
                    if len(parts) > 1:
                        strat = parts[1]
                        if strat in ["sw", "sliding_window"]:
                            self.agent.switch_strategy(ContextStrategy.SLIDING_WINDOW)
                        elif strat in ["facts", "sticky_facts"]:
                            self.agent.switch_strategy(ContextStrategy.STICKY_FACTS)
                        elif strat in ["branch", "branching"]:
                            self.agent.switch_strategy(ContextStrategy.BRANCHING)
                        else:
                            print(f"❌ Неизвестная стратегия: {strat}")
                    else:
                        print("Используйте: /strategy sw | facts | branch")

                elif user_input.startswith("/checkpoint"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        result = self.agent.create_checkpoint(parts[1])
                        print(result)
                    else:
                        print("Укажите имя checkpoint: /checkpoint my_point")

                elif user_input.startswith("/branch"):
                    parts = user_input.split()
                    if len(parts) > 2:
                        result = self.agent.create_branch(parts[1], parts[2])
                        print(result)
                    else:
                        print("Используйте: /branch <from_checkpoint> <new_branch_name>")

                elif user_input.startswith("/switch"):
                    parts = user_input.split()
                    if len(parts) > 1:
                        result = self.agent.switch_branch(parts[1])
                        print(result)
                    else:
                        print("Используйте: /switch <branch_name>")

                elif user_input == "/clear":
                    self.agent.clear()
                    print("🧹 История и статистика очищены")

                elif user_input == "/compare":
                    tester = StrategyTester(self.agent.auth_key)
                    tester.run_full_comparison()

                else:
                    print("🤖 Агент: ", end="", flush=True)
                    answer, stats = self.agent.ask(user_input)
                    print(answer)
                    print(
                        f"\n└─ 📊 {stats.get('strategy', 'unknown')} | Токенов: {stats.get('actual_paid_tokens', 0)} | Стоимость: {stats.get('cost_this', 0):.6f}₽ | Всего: {stats.get('cumulative_paid', 0)} токенов")

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

    print("\nВыберите режим:")
    print("1. Интерактивный режим (работа с агентом)")
    print("2. Запустить сравнение стратегий (авто-тест на сценарии ТЗ)")

    choice = input("\nВаш выбор (1/2): ").strip()

    if choice == "1":
        cli = InteractiveCLI(AUTH_KEY)
        cli.run()
    else:
        tester = StrategyTester(AUTH_KEY)
        tester.run_full_comparison()


if __name__ == "__main__":
    main()