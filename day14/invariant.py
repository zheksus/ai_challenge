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
            " и предложить альтернативу, которая не нарушает инварианты.",
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

    def validate_solution(self, solution: str) -> tuple:
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