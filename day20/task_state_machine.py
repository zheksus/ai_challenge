from enum import Enum
from datetime import datetime
from typing import Optional


class TaskState(Enum):
    """Состояния задачи"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"


class TaskEvent(Enum):
    """События перехода"""
    START = "start"
    CONFIRM = "confirm"
    RESET = "reset"


class TaskStateMachine:
    """
    Конечный автомат для управления задачей.
    Переходы жёстко контролируются: только вперёд.
    """

    # Допустимые переходы: (текущее_состояние, событие) -> новое_состояние
    TRANSITIONS = {
        (TaskState.IDLE, TaskEvent.START): TaskState.PLANNING,
        (TaskState.PLANNING, TaskEvent.CONFIRM): TaskState.EXECUTION,
        (TaskState.EXECUTION, TaskEvent.CONFIRM): TaskState.VALIDATION,
        (TaskState.VALIDATION, TaskEvent.CONFIRM): TaskState.DONE,
        (TaskState.IDLE, TaskEvent.RESET): TaskState.IDLE,
        (TaskState.PLANNING, TaskEvent.RESET): TaskState.IDLE,
        (TaskState.EXECUTION, TaskEvent.RESET): TaskState.IDLE,
        (TaskState.VALIDATION, TaskEvent.RESET): TaskState.IDLE,
        (TaskState.DONE, TaskEvent.RESET): TaskState.IDLE,
    }

    # Описания состояний
    STATE_DESCRIPTIONS = {
        TaskState.IDLE: "Нет активной задачи. Используйте /task <описание> для начала.",
        TaskState.PLANNING: "Планирование. Составьте план или скажите 'сгенерируй план'.",
        TaskState.EXECUTION: "Выполнение. Напишите код или скажите 'сгенерируй код'.",
        TaskState.VALIDATION: "Проверка. Проверьте решение или скажите 'проверь решение'.",
        TaskState.DONE: "Задача завершена! Используйте /reset для новой задачи.",
    }

    # Действия при входе в состояние
    STATE_ENTRY_MESSAGES = {
        TaskState.PLANNING: "📋 Составьте план. Скажите 'план: ...' или 'сгенерируй план'.",
        TaskState.EXECUTION: "💻 Напишите код. Скажите 'код: ...' или 'сгенерируй код'.",
        TaskState.VALIDATION: "🔍 Проверьте решение. Скажите 'проверь решение'.",
        TaskState.DONE: "✅ Задача завершена! Используйте /reset для новой задачи.",
    }

    def __init__(self):
        self.current_state = TaskState.IDLE
        self.start_time = None
        self.history = []  # История переходов

    def transition(self, event: TaskEvent) -> tuple:
        """
        Переход в новое состояние по событию.
        Возвращает (успех, сообщение)
        """
        key = (self.current_state, event)

        if key not in self.TRANSITIONS:
            return False, f"❌ Невозможно перейти из '{self.current_state.value}' по событию '{event.value}'"

        old_state = self.current_state
        new_state = self.TRANSITIONS[key]

        # Запоминаем время старта
        if event == TaskEvent.START:
            self.start_time = datetime.now()

        # Выполняем переход
        self.current_state = new_state

        # Сохраняем историю
        self.history.append({
            "from": old_state.value,
            "to": new_state.value,
            "event": event.value,
            "timestamp": datetime.now().isoformat()
        })

        # Формируем сообщение о переходе
        msg = f"✅ Переход: {old_state.value} → {new_state.value}"

        # Добавляем подсказку для нового состояния
        if new_state in self.STATE_ENTRY_MESSAGES:
            msg += f"\n{self.STATE_ENTRY_MESSAGES[new_state]}"

        return True, msg

    def reset(self):
        """Сброс состояния"""
        old = self.current_state
        self.current_state = TaskState.IDLE
        self.start_time = None
        self.history.append({
            "from": old.value,
            "to": "idle",
            "event": "reset",
            "timestamp": datetime.now().isoformat()
        })
        return "🔄 Задача сброшена. Используйте /task для новой задачи."

    def get_state_description(self) -> str:
        """Описание текущего состояния"""
        return self.STATE_DESCRIPTIONS.get(self.current_state, "Неизвестное состояние")

    def get_state_prompt(self) -> str:
        """
        Получение промпта для системного сообщения на основе текущего состояния.
        Используется для формирования контекста LLM.
        """
        prompts = {
            TaskState.PLANNING: "Ты помогаешь пользователю составить план. Если пользователь просит сгенерировать план - создай структурированный план. Если пользователь даёт свой план - сохрани его в контексте.",
            TaskState.EXECUTION: "Ты помогаешь пользователю с реализацией. Если пользователь просит сгенерировать код - напиши код. Если пользователь даёт свой код - сохрани его в контексте.",
            TaskState.VALIDATION: "Ты помогаешь пользователю проверить решение. Найди ошибки, предложи улучшения.",
            TaskState.DONE: "Задача завершена. Можешь предложить следующую задачу или подвести итоги.",
            TaskState.IDLE: "Нет активной задачи. Предложи пользователю начать новую задачу командой /task."
        }
        return prompts.get(self.current_state, "")

    def is_actionable(self) -> bool:
        """Можно ли выполнять действия в текущем состоянии"""
        return self.current_state in [TaskState.PLANNING, TaskState.EXECUTION, TaskState.VALIDATION]

    def get_progress(self) -> dict:
        """Прогресс выполнения (для статуса)"""
        states_order = [TaskState.IDLE, TaskState.PLANNING, TaskState.EXECUTION, TaskState.VALIDATION, TaskState.DONE]
        try:
            current_idx = states_order.index(self.current_state)
            total = len(states_order) - 1  # IDLE не считается
            progress = (current_idx / total) * 100 if current_idx > 0 else 0
        except ValueError:
            progress = 0

        elapsed = None
        if self.start_time:
            delta = datetime.now() - self.start_time
            elapsed = f"{delta.total_seconds() // 60:.0f} мин"

        return {
            "state": self.current_state.value,
            "progress": round(progress, 1),
            "elapsed": elapsed,
            "description": self.get_state_description()
        }