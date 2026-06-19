import json
import re
from datetime import datetime
from typing import Optional

from day13.task_state_machine import TaskState, TaskEvent, TaskStateMachine


class TaskManager:
    """
    Управление задачами с жёсткими переходами между фазами.
    """

    def __init__(self, agent):
        self.agent = agent
        self.machine = TaskStateMachine()

        # Данные текущей задачи
        self.task_description = ""
        self.plan = None  # План на фазе PLANNING
        self.code = None  # Код на фазе EXECUTION
        self.review = None  # Проверка на фазе VALIDATION

        # Статистика
        self.completed_tasks = []  # Архив завершённых задач

        # Флаг: было ли сгенерировано решение в текущей фазе
        self.has_solution = False

        # Счётчик запросов к LLM для отладки
        self._llm_call_count = 0

    # ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========

    def _call_llm(self, prompt: str, purpose: str = "") -> str:
        """
        Вызов LLM через агента с отладочной печатью.
        purpose - описание цели запроса (для логов)
        """
        self._llm_call_count += 1
        call_id = self._llm_call_count

        print("\n" + "=" * 70)
        print(f"📤 [LLM #{call_id}] ЗАПРОС К LLM")
        print("=" * 70)
        print(f"🎯 Цель: {purpose or 'не указана'}")
        print(f"📝 Текущий этап: {self.machine.current_state.value}")
        print(
            f"📋 Описание задачи: {self.task_description[:100]}..." if self.task_description else "📋 Описание задачи: не указано")
        print("-" * 70)
        print("📄 ПРОМПТ:")
        print("-" * 70)
        # Показываем промпт полностью, но обрезаем если слишком длинный
        if len(prompt) > 2000:
            print(prompt[:2000])
            print(f"... (обрезано, всего {len(prompt)} символов)")
        else:
            print(prompt)
        print("-" * 70)

        try:
            response, token_data = self.agent._call_api(
                [{"role": "user", "content": prompt}],
                temperature=0.7
            )

            # Отладочная печать ответа
            print("=" * 70)
            print(f"📥 [LLM #{call_id}] ОТВЕТ")
            print("=" * 70)

            if "choices" in response:
                result = response["choices"][0]["message"]["content"]
                tokens_used = token_data.get("actual_paid_tokens", 0)
                print(f"💰 Токенов использовано: {tokens_used}")
                print("-" * 70)
                print("📄 ОТВЕТ:")
                print("-" * 70)
                if len(result) > 2000:
                    print(result[:2000])
                    print(f"... (обрезано, всего {len(result)} символов)")
                else:
                    print(result)
                print("=" * 70 + "\n")
                return result
            else:
                print(f"⚠️ Ошибка: {response}")
                print("=" * 70 + "\n")
                return f"⚠️ Ошибка: {response}"

        except Exception as e:
            print(f"❌ Ошибка LLM: {e}")
            print("=" * 70 + "\n")
            return f"⚠️ Ошибка LLM: {e}"

    def _get_solution_hint(self) -> str:
        """Подсказка о том, что нужно предоставить"""
        hints = {
            TaskState.PLANNING: "Скажите 'план: ...' или 'сгенерируй план'",
            TaskState.EXECUTION: "Скажите 'код: ...' или 'сгенерируй код'",
            TaskState.VALIDATION: "Скажите 'проверь решение'",
        }
        return hints.get(self.machine.current_state, "Предоставьте решение для текущей фазы")

    def _generate_plan(self) -> str:
        """Генерация плана через LLM"""
        prompt = f"""Составь подробный план по разработке:

Задача: {self.task_description}

План должен быть структурированным, пошаговым, с указанием основных этапов и подзадач.

Ответь в формате:
1. [Название этапа]
   - [Подзадача 1]
   - [Подзадача 2]
...
"""
        response = self._call_llm(prompt, purpose="Генерация плана")
        self.plan = response
        self.has_solution = True
        return f"📋 Сгенерирован план:\n{response}\n\n✅ Используйте /confirm для перехода к выполнению."

    def _generate_code(self) -> str:
        """Генерация кода через LLM"""
        prompt = f"""Напиши код для следующей задачи:

Задача: {self.task_description}

План: {self.plan or 'План не составлен'}

Напиши реализацию. Используй Python (если не указано иное). Добавь комментарии.
"""
        response = self._call_llm(prompt, purpose="Генерация кода")
        self.code = response
        self.has_solution = True
        return f"💻 Сгенерирован код:\n{response}\n\n✅ Используйте /confirm для перехода к проверке."

    def _review_solution(self) -> str:
        """Проверка решения через LLM"""
        prompt = f"""Проверь следующее решение:

Задача: {self.task_description}
План: {self.plan or 'Нет плана'}
Код: {self.code or 'Нет кода'}

Найди:
1. Ошибки и проблемы
2. Несоответствия плану
3. Предложения по улучшению

Ответь в формате:
🔴 Ошибки:
...
🟡 Предложения:
...
✅ Что работает хорошо:
...
"""
        response = self._call_llm(prompt, purpose="Проверка решения")
        self.review = response
        self.has_solution = True
        return f"🔍 Проверка завершена:\n{response}\n\n✅ Используйте /confirm для завершения задачи."

    def _refine_plan(self, message: str) -> str:
        """Корректировка плана через LLM"""
        prompt = f"""Текущий план:
{self.plan}

Пользователь просит изменить план:
{message}

Обнови план с учётом пожеланий пользователя. Выведи обновлённый план в том же формате.
"""
        response = self._call_llm(prompt, purpose=f"Корректировка плана: {message[:50]}...")
        self.plan = response
        self.has_solution = True
        return f"📋 План обновлён:\n{response}\n\n✅ Используйте /confirm для перехода к выполнению."

    def _refine_code(self, message: str) -> str:
        """Корректировка кода через LLM"""
        prompt = f"""Текущий код:
{self.code}

Пользователь просит изменить код:
{message}

Обнови код с учётом пожеланий пользователя. Выведи обновлённый код.
"""
        response = self._call_llm(prompt, purpose=f"Корректировка кода: {message[:50]}...")
        self.code = response
        self.has_solution = True
        return f"💻 Код обновлён:\n{response}\n\n✅ Используйте /confirm для перехода к проверке."

    def _refine_review(self, message: str) -> str:
        """Корректировка проверки через LLM"""
        prompt = f"""Текущая проверка:
{self.review}

Пользователь просит дополнить проверку:
{message}

Обнови проверку с учётом пожеланий пользователя. Выведи обновлённую проверку в том же формате.
"""
        response = self._call_llm(prompt, purpose=f"Корректировка проверки: {message[:50]}...")
        self.review = response
        self.has_solution = True
        return f"🔍 Проверка обновлена:\n{response}\n\n✅ Используйте /confirm для завершения задачи."

    # ========== ОСТАЛЬНЫЕ МЕТОДЫ (без изменений) ==========

    def handle_command(self, command: str, args: str = "") -> str:
        """Обработка команд: /task, /confirm, /status, /reset"""
        if command == "/task":
            return self._start_task(args)
        elif command == "/confirm":
            return self._confirm_phase()
        elif command == "/status":
            return self._get_status()
        elif command == "/reset":
            return self._reset_task()
        else:
            return f"❌ Неизвестная команда: {command}"

    def handle_message(self, message: str) -> str:
        """Обработка обычного сообщения (корректировка или вопрос внутри фазы)"""
        if self.machine.current_state == TaskState.IDLE:
            return "📭 Нет активной задачи. Используйте /task <описание> для начала."

        if self.machine.current_state == TaskState.DONE:
            return "✅ Задача уже завершена. Используйте /reset для новой задачи."

        if self.machine.current_state == TaskState.PLANNING:
            return self._handle_planning_message(message)

        if self.machine.current_state == TaskState.EXECUTION:
            return self._handle_execution_message(message)

        if self.machine.current_state == TaskState.VALIDATION:
            return self._handle_validation_message(message)

        return f"❌ Неизвестное состояние: {self.machine.current_state.value}"

    def _start_task(self, description: str) -> str:
        """Начать новую задачу"""
        if not description or not description.strip():
            return "❌ Укажите описание задачи: /task <описание>"

        if self.machine.current_state not in [TaskState.IDLE, TaskState.DONE]:
            return f"⚠️ Сначала завершите текущую задачу. Текущее состояние: {self.machine.current_state.value}"

        self.task_description = description.strip()
        self.plan = None
        self.code = None
        self.review = None
        self.has_solution = False

        success, msg = self.machine.transition(TaskEvent.START)

        return f"✅ Задача начата!\n📝 Описание: {self.task_description}\n\n{msg}"

    def _confirm_phase(self) -> str:
        """Подтверждение текущей фазы и переход к следующей"""
        current = self.machine.current_state

        if current == TaskState.IDLE:
            return "❌ Нет активной задачи. Используйте /task для начала."

        if current == TaskState.DONE:
            return "✅ Задача уже завершена! Используйте /reset для новой задачи."

        if not self.has_solution:
            return f"❌ Сначала предоставьте решение.\n{self._get_solution_hint()}"

        success, msg = self.machine.transition(TaskEvent.CONFIRM)

        if success:
            if self.machine.current_state == TaskState.DONE:
                self._archive_task()
                self.has_solution = False
            else:
                self.has_solution = False
            return msg
        else:
            return msg

    def _get_status(self) -> str:
        """Получение статуса задачи"""
        if self.machine.current_state == TaskState.IDLE:
            return "📭 Нет активной задачи. Используйте /task <описание>"

        progress = self.machine.get_progress()
        lines = [
            "=" * 50,
            "📊 СТАТУС ЗАДАЧИ",
            "=" * 50,
            f"📝 Описание: {self.task_description}",
            f"📌 Этап: {progress['state'].upper()}",
            f"📈 Прогресс: {progress['progress']}%",
            f"⏱️ Время: {progress['elapsed'] or 'не начато'}",
            f"💬 {progress['description']}",
        ]

        if self.plan:
            preview = self.plan[:150] + "..." if len(self.plan) > 150 else self.plan
            lines.append(f"\n📋 План:\n{preview}")

        if self.code:
            preview = self.code[:150] + "..." if len(self.code) > 150 else self.code
            lines.append(f"\n💻 Код:\n{preview}")

        if self.review:
            preview = self.review[:150] + "..." if len(self.review) > 150 else self.review
            lines.append(f"\n🔍 Проверка:\n{preview}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def _reset_task(self) -> str:
        """Сброс задачи"""
        if self.machine.current_state == TaskState.DONE:
            self._archive_task()

        self.task_description = ""
        self.plan = None
        self.code = None
        self.review = None
        self.has_solution = False

        return self.machine.reset()

    def _handle_planning_message(self, message: str) -> str:
        """Обработка сообщения на фазе PLANNING"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["сгенерируй план", "generate plan", "составь план", "предложи план"]):
            return self._generate_plan()

        if "план:" in message_lower or "plan:" in message_lower:
            match = re.search(r'(?:план|plan):\s*([\s\S]+)', message, re.IGNORECASE)
            if match:
                plan_content = match.group(1).strip()
                self.plan = plan_content
                self.has_solution = True
                return f"📋 План сохранён:\n{plan_content}\n\n✅ Используйте /confirm для перехода к выполнению."

        if self.plan:
            return self._refine_plan(message)

        return "📋 Я не вижу плана. Скажите 'план: ...' или 'сгенерируй план'."

    def _handle_execution_message(self, message: str) -> str:
        """Обработка сообщения на фазе EXECUTION"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["сгенерируй код", "generate code", "напиши код"]):
            return self._generate_code()

        if "код:" in message_lower or "code:" in message_lower:
            match = re.search(r'(?:код|code):\s*([\s\S]+)', message, re.IGNORECASE)
            if match:
                code_content = match.group(1).strip()
                self.code = code_content
                self.has_solution = True
                return f"💻 Код сохранён:\n{code_content}\n\n✅ Используйте /confirm для перехода к проверке."

        if self.code:
            return self._refine_code(message)

        return "💻 Я не вижу кода. Скажите 'код: ...' или 'сгенерируй код'."

    def _handle_validation_message(self, message: str) -> str:
        """Обработка сообщения на фазе VALIDATION"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["проверь решение", "проверь код", "review", "найди ошибки"]):
            return self._review_solution()

        if self.review:
            return self._refine_review(message)

        return "🔍 Скажите 'проверь решение' для автоматической проверки."

    def _archive_task(self):
        """Архивирование завершённой задачи в long_term"""
        if hasattr(self.agent, 'memory'):
            task_data = {
                "description": self.task_description,
                "plan": self.plan,
                "code": self.code,
                "review": self.review,
                "completed_at": datetime.now().isoformat()
            }
            self.agent.memory.add_to_long_term(
                "completed_tasks",
                f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                json.dumps(task_data, ensure_ascii=False),
                importance=0.8
            )
            self.completed_tasks.append(task_data)

    def get_context_prompt(self) -> str:
        """Получение промпта для системного сообщения"""
        if self.machine.current_state == TaskState.IDLE:
            return ""

        lines = [
            "### СОСТОЯНИЕ ЗАДАЧИ:",
            f"📊 Этап: {self.machine.current_state.value.upper()}",
            f"📝 Описание: {self.task_description}",
            f"💬 {self.machine.get_state_description()}",
        ]

        if self.plan:
            preview = self.plan[:300] + "..." if len(self.plan) > 300 else self.plan
            lines.append(f"\n📋 План:\n{preview}")

        if self.code:
            preview = self.code[:300] + "..." if len(self.code) > 300 else self.code
            lines.append(f"\n💻 Код:\n{preview}")

        if self.review:
            preview = self.review[:300] + "..." if len(self.review) > 300 else self.review
            lines.append(f"\n🔍 Проверка:\n{preview}")

        if self.has_solution:
            lines.append("\n✅ Решение предоставлено. Используйте /confirm для перехода к следующей фазе.")
        else:
            lines.append(f"\n❌ Решение не предоставлено. {self._get_solution_hint()}")

        state_prompt = self.machine.get_state_prompt()
        if state_prompt:
            lines.append(f"\n### ИНСТРУКЦИЯ ДЛЯ АССИСТЕНТА:\n{state_prompt}")

        return "\n".join(lines)

    def is_idle(self) -> bool:
        """Проверка, есть ли активная задача"""
        return self.machine.current_state == TaskState.IDLE