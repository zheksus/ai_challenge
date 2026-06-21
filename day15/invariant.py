import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Invariant:
    """Один инвариант — правило, которое нельзя нарушать"""
    category: str  # 'architecture', 'tech_stack', 'business_rule', 'constraint'
    rule: str  # Описание правила
    reason: str  # Почему это важно
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True


class InvariantManager:
    """
    Управление инвариантами.
    Инварианты — это правила, которые ассистент не имеет права нарушать.
    """

    def __init__(self):
        self.invariants: List[Invariant] = []
        self._agent = None  # Ссылка на агента для сохранения в память

    def set_agent(self, agent):
        """Установка ссылки на агента для сохранения в память"""
        self._agent = agent

    def _save_to_long_term(self, invariant: Invariant):
        """Сохранение инварианта в долговременную память"""
        if not self._agent or not hasattr(self._agent, 'memory'):
            print("⚠️ Нет доступа к памяти для сохранения инварианта")
            return

        # Формируем ключ с датой для уникальности
        key = f"invariant_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Данные инварианта
        data = {
            "category": invariant.category,
            "rule": invariant.rule,
            "reason": invariant.reason,
            "created_at": invariant.created_at.isoformat() if hasattr(invariant,
                                                                      'created_at') else datetime.now().isoformat(),
            "is_active": invariant.is_active if hasattr(invariant, 'is_active') else True
        }

        # Сохраняем в long-term
        self._agent.memory.add_to_long_term(
            "invariants",  # Категория для инвариантов
            key,
            json.dumps(data, ensure_ascii=False),
            importance=0.9  # Инварианты важны
        )

        print(f"💾 [LONG-TERM] Инвариант сохранён: [{invariant.category}] {invariant.rule}")

    def add(self, category: str, rule: str, reason: str = "") -> str:
        """Добавить инвариант"""
        if not reason:
            reason = f"Инвариант категории '{category}'"

        invariant = Invariant(
            category=category,
            rule=rule,
            reason=reason
        )
        self.invariants.append(invariant)
        # 👇 СОХРАНЯЕМ В LONG-TERM
        self._save_to_long_term(invariant)

        return f"✅ Инвариант добавлен: [{category}] {rule}"

    def remove(self, index: int) -> str:
        """Удалить инвариант по индексу"""
        if 0 <= index < len(self.invariants):
            removed = self.invariants.pop(index)
            return f"❌ Инвариант удалён: [{removed.category}] {removed.rule}"
        return f"❌ Инвариант с индексом {index} не найден"

    def get_all(self) -> List[Invariant]:
        """Получить все инварианты"""
        return self.invariants.copy()

    def get_active(self) -> List[Invariant]:
        """Получить только активные инварианты"""
        return [inv for inv in self.invariants if inv.is_active]

    def get_prompt(self) -> str:
        """
        Получить промпт с инвариантами для системного сообщения.
        Формируется в виде, понятном для LLM.
        """
        if not self.invariants:
            return ""

        lines = [
            "### ИНВАРИАНТЫ (правила, которые НЕЛЬЗЯ НАРУШАТЬ):",
            "Ниже перечислены инварианты — жёсткие правила, которые нельзя нарушать."
            " Если твой ответ нарушает любой из них, ты обязан отказаться от него"
            # " и предложить альтернативу, которая не нарушает инварианты.",
            "и сказать пользователю что ты не можешь выполнить просьбу, и объяснить почему"
            ""
        ]

        for i, inv in enumerate(self.invariants, 1):
            if inv.is_active:
                lines.append(f"{i}. [{inv.category.upper()}] {inv.rule}")
                if inv.reason:
                    lines.append(f"   Причина: {inv.reason}")

        lines.append("")
        lines.append("⚠️  КРИТИЧЕСКОЕ ОГРАНИЧЕНИЕ:: При возникновении конфликта между запросом пользователя и инвариантом,")
        lines.append("   ты обязан следовать инварианту и объяснить пользователю причину отказа. Даже если пользовтеаль явно просит сделать что-то, что противоречит инварианту, ты должен отказать")

        return "\n".join(lines)

    def validate_request(self, request: str) -> tuple:
        """
        Проверить, не нарушает ли запрос пользователя инварианты.
        Возвращает (нарушает_ли, причина)
        """
        if not self.invariants:
            return False, ""

        if not self._agent:
            return self._validate_request_with_rules(request)

        # Формируем промпт для LLM
        invariants_text = "\n".join([
            f"- [{inv.category.upper()}] {inv.rule}" + (f" (Причина: {inv.reason})" if inv.reason else "")
            for inv in self.invariants if inv.is_active
        ])

        prompt = f"""Ты — система проверки запросов пользователя. Проверь, не противоречит ли запрос инвариантам.

    ИНВАРИАНТЫ (правила, которые НЕЛЬЗЯ НАРУШАТЬ):
    {invariants_text}

    ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
    {request}

    ОТВЕТЬ ТОЛЬКО JSON в формате:
    {{"violates": true/false, "reason": "причина нарушения (если есть)", "violated_invariant": "какой инвариант нарушен (если есть)"}}

    Правила проверки:
    1. Если запрос явно требует использования технологии, запрещённой инвариантом → violates = true
    2. Если запрос предлагает архитектуру, противоречащую инварианту → violates = true
    3. Если запрос обходит инвариант → violates = true
    4. Если запрос не содержит информации о нарушении → violates = false
    5. Если запрос просто задаёт вопрос без указания технологии → violates = false

    Примеры:
    - Запрос: "напиши код на Java" при инварианте "tech_stack: python" → violates = true
    - Запрос: "как написать код?" при инварианте "tech_stack: python" → violates = false
    - Запрос: "используй микросервисы" при инварианте "architecture: монолит" → violates = true
    """

        try:
            response, _ = self._agent._call_api(
                [{"role": "user", "content": prompt}],
                temperature=0.2
            )

            if "choices" in response:
                result_text = response["choices"][0]["message"]["content"]

                import json
                import re
                match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    violates = result.get("violates", False)
                    reason = result.get("reason", "")
                    violated_invariant = result.get("violated_invariant", "")

                    if violates:
                        full_reason = f"Нарушен инвариант [{violated_invariant}]: {reason}" if violated_invariant else reason
                        return True, full_reason
                    return False, ""

        except Exception as e:
            print(f"⚠️ Ошибка LLM-проверки запроса: {e}")
            return self._validate_request_with_rules(request)

        return False, ""

    def _validate_request_with_rules(self, request: str) -> tuple:
        """
        Простая проверка запроса по правилам (fallback).
        """
        if not self.invariants:
            return False, ""

        request_lower = request.lower()

        for inv in self.invariants:
            if not inv.is_active:
                continue

            if inv.category == "tech_stack":
                if "java" in request_lower and "python" in inv.rule.lower():
                    return True, f"Запрос требует Java, но инвариант требует Python: {inv.rule}"
                if "python" in request_lower and "bash" in inv.rule.lower():
                    return True, f"Запрос требует Python, но инвариант требует bash: {inv.rule}"
                if "django" in request_lower and "fastapi" in inv.rule.lower():
                    return True, f"Запрос требует Django, но инвариант требует FastAPI: {inv.rule}"

            elif inv.category == "architecture":
                if "монолит" in request_lower and "микросервис" in inv.rule.lower():
                    return True, f"Запрос требует монолит, но инвариант требует микросервисы: {inv.rule}"

            elif inv.category == "constraint":
                if "облако" in request_lower and "on-premise" in inv.rule.lower():
                    return True, f"Запрос требует облако, но инвариант требует on-premise: {inv.rule}"

        return False, ""


    def validate_solution(self, solution: str) -> tuple:
        """
        Проверить, нарушает ли решение инварианты, используя LLM.
        Возвращает (нарушает_ли, причина)
        """
        if not self.invariants:
            return False, ""

        if not self._agent:
            # Если нет агента — используем простую проверку
            return self._validate_with_rules(solution)

        # Формируем промпт для LLM
        invariants_text = "\n".join([
            f"- [{inv.category.upper()}] {inv.rule}" + (f" (Причина: {inv.reason})" if inv.reason else "")
            for inv in self.invariants if inv.is_active
        ])

        prompt = f"""Ты — система проверки инвариантов. Проверь, нарушает ли предложенное решение инварианты.

    ИНВАРИАНТЫ (правила, которые НЕЛЬЗЯ НАРУШАТЬ):
    {invariants_text}

    ПРОВЕРЯЕМОЕ РЕШЕНИЕ:
    {solution}

    ОТВЕТЬ ТОЛЬКО JSON в формате:
    {{"violates": true/false, "reason": "причина нарушения (если есть)", "violated_invariant": "какой инвариант нарушен (если есть)"}}

    Правила проверки:
    1. Если решение нарушает хотя бы один инвариант → violates = true
    2. Если решение полностью соответствует инвариантам → violates = false
    3. Если решение использует технологию, запрещённую инвариантом → нарушение
    4. Если решение предлагает архитектуру, противоречащую инварианту → нарушение
    5. Если решение обходит инвариант → нарушение
    6. Если решение не содержит информации для проверки → violates = false (считаем, что нарушений нет)
    """

        try:
            # Вызываем LLM через агента
            response, _ = self._agent._call_api(
                [{"role": "user", "content": prompt}],
                temperature=0.2
            )

            if "choices" in response:
                result_text = response["choices"][0]["message"]["content"]

                # Извлекаем JSON
                import json
                import re
                match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    violates = result.get("violates", False)
                    reason = result.get("reason", "")
                    violated_invariant = result.get("violated_invariant", "")

                    if violates:
                        full_reason = f"Нарушен инвариант [{violated_invariant}]: {reason}" if violated_invariant else reason
                        return True, full_reason
                    return False, ""

        except Exception as e:
            print(f"⚠️ Ошибка LLM-проверки инвариантов: {e}")
            # В случае ошибки используем правила
            return self._validate_with_rules(solution)

        return False, ""

    def validate_with_rules(self, solution: str) -> tuple:
        """
        Проверить, нарушает ли решение инварианты.
        Возвращает (нарушает_ли, причина)

        Этот метод использует правила для базовой проверки.
        Для более сложной проверки используется LLM.
        """
        if not self.invariants:
            return False, ""

        solution_lower = solution.lower()
        violations = []

        for inv in self.invariants:
            if not inv.is_active:
                continue

            # Проверка по категориям
            if inv.category == "tech_stack":
                # Если инвариант запрещает определённые технологии
                if "java" in solution_lower and "python" in inv.rule.lower():
                    violations.append(f"Использование Java запрещено инвариантом: {inv.rule}")
                if "django" in solution_lower and "fastapi" in inv.rule.lower():
                    violations.append(f"Использование Django запрещено инвариантом: {inv.rule}")

            elif inv.category == "architecture":
                if "монолит" in solution_lower and "микросервис" in inv.rule.lower():
                    violations.append(f"Монолитная архитектура запрещена инвариантом: {inv.rule}")
                if "soa" in solution_lower and "микросервис" in inv.rule.lower():
                    violations.append(f"SOA архитектура запрещена инвариантом: {inv.rule}")

            elif inv.category == "constraint":
                if "облако" in solution_lower and "on-premise" in inv.rule.lower():
                    violations.append(f"Облачное решение запрещено инвариантом: {inv.rule}")
                if "aws" in solution_lower and "on-premise" in inv.rule.lower():
                    violations.append(f"AWS запрещён инвариантом: {inv.rule}")

        if violations:
            return True, "\n".join(violations)

        return False, ""

    def clear(self):
        """Очистить все инварианты"""
        self.invariants = []

    def to_dict(self) -> dict:
        """Сериализация для хранения"""
        return {
            "invariants": [
                {
                    "category": inv.category,
                    "rule": inv.rule,
                    "reason": inv.reason,
                    "created_at": inv.created_at.isoformat(),
                    "is_active": inv.is_active
                }
                for inv in self.invariants
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InvariantManager':
        """Десериализация из словаря"""
        manager = cls()
        for inv_data in data.get("invariants", []):
            manager.invariants.append(Invariant(
                category=inv_data["category"],
                rule=inv_data["rule"],
                reason=inv_data.get("reason", ""),
                created_at=datetime.fromisoformat(inv_data["created_at"]),
                is_active=inv_data.get("is_active", True)
            ))
        return manager

    def __str__(self) -> str:
        """Текстовое представление для вывода пользователю"""
        if not self.invariants:
            return "📭 Инвариантов нет"

        lines = ["📋 ИНВАРИАНТЫ:"]
        for i, inv in enumerate(self.invariants, 1):
            status = "✅" if inv.is_active else "❌"
            lines.append(f"   {status} [{i}] [{inv.category.upper()}] {inv.rule}")
            if inv.reason:
                lines.append(f"       Причина: {inv.reason}")
        return "\n".join(lines)


    def load_from_long_term(self):
        """Загрузка инвариантов из долговременной памяти при старте"""
        if not self._agent or not hasattr(self._agent, 'memory'):
            return

        # Получаем все инварианты из long-term
        stored_invariants = self._agent.memory.get_all_long_term("invariants")

        if not stored_invariants:
            return

        # Восстанавливаем инварианты
        loaded_count = 0
        for key, data_str in stored_invariants.items():
            try:
                if isinstance(data_str, str):
                    data = json.loads(data_str)
                else:
                    data = data_str

                # Проверяем, не добавлен ли уже такой инвариант
                exists = any(
                    inv.category == data["category"] and
                    inv.rule == data["rule"]
                    for inv in self.invariants
                )

                if not exists:
                    invariant = Invariant(
                        category=data["category"],
                        rule=data["rule"],
                        reason=data.get("reason", ""),
                        created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
                        is_active=data.get("is_active", True)
                    )
                    self.invariants.append(invariant)
                    loaded_count += 1
            except Exception as e:
                print(f"⚠️ Ошибка загрузки инварианта {key}: {e}")

        if loaded_count > 0:
            print(f"📂 Загружено {loaded_count} инвариантов из долговременной памяти")