"""
CLI-агент с оркестрацией по этапам задачи (d14).

Архитектура:
  OrchestratorAgent  — маршрутизирует к нужному агенту, валидирует переходы
  PlanningAgent      — разбивает задачу на шаги, создаёт план
  ExecutionAgent     — помогает выполнить текущий шаг
  ValidationAgent    — проверяет полноту выполнения

Пользователь подтверждает каждый переход командой /confirm.
Пауза в любой момент — возобновление без повторных объяснений.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic

from memory import (
    ShortTermMemory, WorkingMemory, LongTermMemory, InvariantMemory,
    PROFILE_FIELDS, PREF_FIELDS, INVARIANT_CATEGORIES,
    STAGES, STAGE_TRANSITIONS, WAITING, WORKING,
)

# ── Init ───────────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("ERROR: CLAUDE_API_KEY not found in .env")

client  = anthropic.Anthropic(api_key=API_KEY)
MODEL   = "claude-opus-4-8"

short     = ShortTermMemory()
working   = WorkingMemory()
long_     = LongTermMemory()
invariants = InvariantMemory()


# ══════════════════════════════════════════════════════════════════════════════
#  SUB-AGENTS
#  Each agent: builds its system prompt, has its own tool set, calls the LLM.
#  The orchestrator only decides which one runs.
# ══════════════════════════════════════════════════════════════════════════════

def _prefs_block() -> str:
    text = long_.prefs_to_prompt_text()
    if not text:
        return ""
    return f"\n## Правила общения (соблюдай в каждом ответе)\n{text}\n"

def _invariants_block() -> str:
    return invariants.to_prompt_block()



def _run_agent_loop(system: str, tools: list, tool_handler, user_input: str) -> str:
    """Generic tool-use loop used by every sub-agent."""
    short.add("user", user_input)
    # Names of "finish" tools — when model stops without calling one, we nudge it.
    finish_tools = {t["name"] for t in tools if t["name"].startswith("finish_")}
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            messages=short.get_messages(),
            tools=tools,
        )
        if response.stop_reason == "tool_use":
            assistant_blocks, tool_results = [], []
            called_finish = False
            for block in response.content:
                if block.type == "text":
                    assistant_blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_blocks.append({
                        "type": "tool_use", "id": block.id,
                        "name": block.name, "input": block.input,
                    })
                    result = tool_handler(block.name, block.input)
                    print(f"  [АГЕНТ] {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                    if block.name in finish_tools:
                        called_finish = True
            short.add("assistant", assistant_blocks)
            short.add("user", tool_results)
            if called_finish:
                # Return any text that came alongside the finish tool call — stop here.
                inline_text = "\n".join(
                    b["text"] for b in assistant_blocks if b.get("type") == "text"
                ).strip()
                return inline_text
        else:
            reply = "\n".join(b.text for b in response.content if b.type == "text")
            # If model wrote text but didn't call finish tool, nudge it
            if finish_tools and not any(
                b.type == "tool_use" for b in response.content
            ):
                short.add("assistant", reply)
                short.add("user", [{
                    "type": "text",
                    "text": "Пожалуйста, теперь вызови инструмент finish_* чтобы зафиксировать результат.",
                }])
                # Force tool call on next iteration (use tool_choice=any)
                nudge = client.messages.create(
                    model=MODEL,
                    max_tokens=2048,
                    system=system,
                    messages=short.get_messages(),
                    tools=tools,
                    tool_choice={"type": "any"},
                )
                if nudge.stop_reason == "tool_use":
                    nudge_blocks, nudge_results = [], []
                    for block in nudge.content:
                        if block.type == "text":
                            nudge_blocks.append({"type": "text", "text": block.text})
                        elif block.type == "tool_use":
                            nudge_blocks.append({
                                "type": "tool_use", "id": block.id,
                                "name": block.name, "input": block.input,
                            })
                            result = tool_handler(block.name, block.input)
                            print(f"  [АГЕНТ] {result}")
                            nudge_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            })
                    short.add("assistant", nudge_blocks)
                    short.add("user", nudge_results)
                    final = client.messages.create(
                        model=MODEL, max_tokens=512, system=system,
                        messages=short.get_messages(), tools=tools,
                    )
                    final_reply = "\n".join(b.text for b in final.content if b.type == "text")
                    short.add("assistant", final_reply)
                    return reply + "\n" + final_reply
            short.add("assistant", reply)
            return reply


# ── Planning Agent ─────────────────────────────────────────────────────────────

_PLANNING_TOOLS = [
    {
        "name": "finish_planning",
        "description": (
            "Зафиксировать готовый план в памяти. "
            "Вызывай ТОЛЬКО когда план полностью готов и обсуждён с пользователем. "
            "После вызова пользователь введёт /confirm для перехода к выполнению."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Список шагов плана по порядку (конкретные, выполнимые)",
                },
                "summary": {
                    "type": "string",
                    "description": "Краткое резюме плана для передачи следующему агенту",
                },
            },
            "required": ["steps", "summary"],
        },
    },
    {
        "name": "add_note",
        "description": "Сохранить важное решение или ограничение из обсуждения плана.",
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
    },
]

def _planning_tool_handler(name: str, inp: dict) -> str:
    if name == "finish_planning":
        err = working.finish_planning(inp["steps"], inp["summary"])
        if err:
            return f"Ошибка: {err}"
        return f"План сохранён ({len(inp['steps'])} шагов). Ожидаем /confirm от пользователя."
    if name == "add_note":
        err = working.add_note(inp["note"])
        return "Заметка сохранена." if not err else f"Ошибка: {err}"
    return f"Неизвестный инструмент: {name}"


def planning_agent(user_input: str, task: dict) -> str:
    profile  = long_.to_prompt_text()
    prefs    = _prefs_block()
    inv_block = _invariants_block()
    existing = task["stage_results"].get("planning", "")
    resume_hint = (
        f"\nПредыдущий контекст планирования:\n{existing}\n"
        if existing else ""
    )
    system = f"""Ты — агент планирования. Твоя задача: разбить задачу пользователя на конкретные выполнимые шаги.
{inv_block}
{prefs}
## Задача
{task['name']}

## Профиль пользователя
{profile}
{resume_hint}
## Инструкции
- Сначала задай уточняющие вопросы если нужны детали, или сразу предложи план если задача ясна.
- При составлении плана проверяй каждый шаг на соответствие инвариантам (если они заданы).
- Когда план согласован с пользователем — напиши краткое резюме плана, скажи "Введи /confirm чтобы начать выполнение", ЗАТЕМ вызови finish_planning(steps, summary).
- Шаги должны быть конкретными: что именно делать, какой результат.
- НЕ задавай вопросы ПОСЛЕ вызова finish_planning — это финальный шаг в диалоге.
"""
    return _run_agent_loop(system, _PLANNING_TOOLS, _planning_tool_handler, user_input)


# ── Execution Agent ────────────────────────────────────────────────────────────

_EXECUTION_TOOLS = [
    {
        "name": "finish_execution",
        "description": (
            "Зафиксировать завершение ВСЕХ шагов выполнения. "
            "Вызывай когда все шаги плана разобраны и описаны. "
            "После вызова пользователь введёт /confirm для перехода к валидации."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step_results": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Список итогов по каждому шагу плана (в том же порядке)",
                },
                "summary": {
                    "type": "string",
                    "description": "Общий итог выполнения: что сделано, ключевые решения, результат",
                },
            },
            "required": ["step_results", "summary"],
        },
    },
    {
        "name": "add_note",
        "description": "Сохранить важное решение или наблюдение.",
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
    },
]

def _execution_tool_handler(name: str, inp: dict) -> str:
    if name == "finish_execution":
        err = working.finish_execution(inp["step_results"], inp["summary"])
        if err:
            return f"Ошибка: {err}"
        return "Выполнение зафиксировано. Ожидаем /confirm от пользователя."
    if name == "add_note":
        err = working.add_note(inp["note"])
        return "Заметка сохранена." if not err else f"Ошибка: {err}"
    return f"Неизвестный инструмент: {name}"


def execution_agent(user_input: str, task: dict) -> str:
    profile   = long_.to_prompt_text()
    prefs     = _prefs_block()
    inv_block = _invariants_block()
    plan      = task.get("plan", [])
    total     = len(plan)
    plan_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan)) or "(нет шагов)"

    system = f"""Ты — агент выполнения. Твоя задача — провести пользователя через ВСЕ шаги плана за один раз.
{inv_block}
{prefs}
## Задача
{task['name']}

## Профиль пользователя
{profile}

## Контекст планирования
{task['stage_results'].get('planning', '(нет)')}

## План ({total} шагов)
{plan_text}

## Инструкции
- Разбери все {total} шагов по порядку: по каждому дай конкретные рекомендации, детали, примеры.
- Если нужны уточнения от пользователя — задай вопросы по всему плану сразу, а не по одному шагу.
- ОБЯЗАТЕЛЬНО вызови инструмент finish_execution в конце ЭТОГО ЖЕ ответа — сразу после того как описал все шаги.
  - step_results: список кратких итогов для каждого шага (ровно {total} элементов, по порядку плана)
  - summary: общий итог всего выполнения в 2-3 предложениях
- Без вызова finish_execution пользователь не сможет продолжить. Это обязательный шаг.
- Перед вызовом finish_execution напиши: "Выполнение готово ✅. Введи /confirm для перехода к проверке." — затем сразу вызывай инструмент.
- НЕ генерируй текст ПОСЛЕ вызова finish_execution.
"""
    return _run_agent_loop(system, _EXECUTION_TOOLS, _execution_tool_handler, user_input)


# ── Validation Agent ───────────────────────────────────────────────────────────

_VALIDATION_TOOLS = [
    {
        "name": "finish_validation",
        "description": (
            "Зафиксировать результат валидации. "
            "Вызывай когда проверка завершена. "
            "После вызова пользователь введёт /confirm для завершения задачи (или возврата)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "passed": {
                    "type": "boolean",
                    "description": "true — задача выполнена успешно, false — нужно доработать",
                },
                "summary": {
                    "type": "string",
                    "description": "Итоговая оценка: что сделано хорошо, что требует доработки",
                },
            },
            "required": ["passed", "summary"],
        },
    },
    {
        "name": "add_note",
        "description": "Сохранить замечание к валидации.",
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
    },
]

def _validation_tool_handler(name: str, inp: dict) -> str:
    if name == "finish_validation":
        err = working.finish_validation(inp["passed"], inp["summary"])
        if err:
            return f"Ошибка: {err}"
        icon = "✓ Пройдена" if inp["passed"] else "✗ Не пройдена"
        return f"Валидация: {icon}. Ожидаем /confirm от пользователя."
    if name == "add_note":
        err = working.add_note(inp["note"])
        return "Заметка сохранена." if not err else f"Ошибка: {err}"
    return f"Неизвестный инструмент: {name}"


def validation_agent(user_input: str, task: dict) -> str:
    profile   = long_.to_prompt_text()
    prefs     = _prefs_block()
    inv_block = _invariants_block()
    plan      = task.get("plan", [])
    total     = len(plan)

    step_results = []
    for i, step in enumerate(plan):
        note = task["step_notes"][i] if i < len(task["step_notes"]) else ""
        step_results.append(f"  [{i+1}/{total}] {step}\n      → {note or '(нет итога)'}")
    steps_text = "\n".join(step_results) if step_results else "(нет)"

    system = f"""Ты — агент валидации. Ты проверяешь качество и полноту выполненной задачи.
{inv_block}
{prefs}
## Задача
{task['name']}

## Профиль пользователя
{profile}

## План и результаты выполнения
{steps_text}

## Инструкции
- Оцени каждый шаг: выполнен ли он полностью и качественно.
- Если заданы инварианты — проверь, не нарушает ли результат ни один из них. Нарушение = автоматически passed=false.
- Выяви пробелы, риски, незавершённые части.
- Напиши итоговое заключение, скажи "Введи /confirm для завершения задачи", ЗАТЕМ вызови finish_validation(passed, summary).
  - passed=true: все критичные шаги выполнены, задача готова.
  - passed=false: есть существенные пробелы, нужно вернуться.
- НЕ генерируй текст ПОСЛЕ вызова finish_validation.
"""
    return _run_agent_loop(system, _VALIDATION_TOOLS, _validation_tool_handler, user_input)


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def orchestrate(user_input: str) -> str:
    """Route user input to the correct sub-agent based on current task stage."""
    task = working.get_current_task()
    if not task:
        return (
            "Нет активной задачи. Создайте новую: /task new <название>\n"
            "Или выберите существующую: /task list"
        )

    stage = task["stage"]

    if stage == "done":
        return (
            f"Задача '{task['name']}' завершена.\n"
            "Создайте новую (/task new) или переключитесь на другую (/task list)."
        )

    if stage == "planning":
        return planning_agent(user_input, task)
    elif stage == "execution":
        return execution_agent(user_input, task)
    elif stage == "validation":
        return validation_agent(user_input, task)

    return f"Неизвестный этап: {stage}"


def _kickstart_current_stage() -> str:
    """
    Orchestrator-generated prompt to immediately start the current sub-agent.
    Called after task creation and after every /confirm transition — no user
    message needed to get the agent going.
    """
    task = working.get_current_task()
    if not task or task["stage"] == "done":
        return ""

    stage = task["stage"]
    plan  = task.get("plan", [])
    idx   = task.get("step_index", 0)

    if stage == "planning":
        msg = (
            f"Задача: «{task['name']}».\n"
            "Начни планирование: задай пользователю уточняющие вопросы, "
            "чтобы лучше понять задачу, или — если задача уже ясна — "
            "сразу предложи конкретный план шагов."
        )

    elif stage == "execution":
        plan_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan))
        msg = (
            f"Задача: «{task['name']}».\n\n"
            f"План ({len(plan)} шагов):\n{plan_text}\n\n"
            "Разбери все шаги подробно, затем ОБЯЗАТЕЛЬНО вызови инструмент finish_execution."
        )

    elif stage == "validation":
        total = len(plan)
        step_lines = "\n".join(
            f"  [{i+1}/{total}] {plan[i]} → {task['step_notes'][i] or '(нет итога)'}"
            for i in range(total)
        )
        msg = (
            f"Все {total} шагов задачи «{task['name']}» выполнены.\n\n"
            f"Результаты:\n{step_lines}\n\n"
            "Проведи детальную проверку каждого шага. "
            "Оцени полноту, качество и соответствие исходной задаче. "
            "Вынеси итоговый вердикт."
        )

    else:
        return ""

    return orchestrate(msg)


# ── /confirm ───────────────────────────────────────────────────────────────────

def do_confirm() -> str:
    new_stage, message = working.confirm_transition()
    if not new_stage:
        return message  # error or "nothing to confirm"

    short.clear()
    print(f"  ✓ {message}")
    print(f"  → запускаю агент этапа «{working.get_current_task()['stage']}»…\n")

    # Orchestrator immediately kicks off the new sub-agent — no user prompt needed
    return _kickstart_current_stage()


# ── /status ────────────────────────────────────────────────────────────────────

def show_status() -> None:
    task = working.get_current_task()
    if not task:
        print("\n  (нет активной задачи)\n")
        return

    stage = task["stage"]
    action = task["expected_action"]
    plan = task.get("plan", [])
    idx = task.get("step_index", 0)

    print("\n" + "═" * 60)
    print(f"  Задача: {task['name']}")
    print("═" * 60)
    print(f"  Этап    : {stage}")
    if stage == "execution" and plan:
        print(f"  Шаг     : {idx+1}/{len(plan)} — {plan[idx]}")
    print(f"  Действие: {action}", end="")
    if action == WAITING:
        print("  ← введите /confirm для продолжения", end="")
    print()

    if plan:
        print("\n  ПЛАН:")
        for i, step in enumerate(plan):
            if i < idx:
                marker = "✓"
            elif i == idx and stage == "execution":
                marker = "→"
            else:
                marker = " "
            note = task["step_notes"][i] if i < len(task["step_notes"]) else ""
            suffix = f"  ({note[:50]}…)" if note else ""
            print(f"    {marker} [{i+1}] {step}{suffix}")

    if task.get("pending_result"):
        print(f"\n  Ожидает подтверждения:\n  {task['pending_result'][:200]}")

    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  HELP + COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

HELP_TEXT = """
╔══════════════════════════════════════════════════════════════════╗
║                АГЕНТ-ОРКЕСТРАТОР  (d14)                          ║
╠══════════════════════════════════════════════════════════════════╣
║  РАБОЧИЙ ПРОЦЕСС                                                 ║
║  planning → execution → validation → done                        ║
║                                                                  ║
║  /confirm           подтвердить переход на следующий этап        ║
║  /back              откатиться на предыдущий этап                ║
║  /status            текущий этап, ожидаемое действие            ║
╠══════════════════════════════════════════════════════════════════╣
║  ЗАДАЧИ                                                          ║
║  /task new <название>   создать задачу                           ║
║  /task list             список задач                             ║
║  /task use <название>   переключиться на задачу                  ║
║  /note <текст>          добавить заметку к текущей задаче        ║
╠══════════════════════════════════════════════════════════════════╣
║  ИНВАРИАНТЫ                                                      ║
║  /inv list                    показать все инварианты            ║
║  /inv add <кат> <правило>     добавить инвариант                 ║
║  /inv del <id>                удалить инвариант                  ║
║  Категории: arch, tech, stack, business, other                   ║
╠══════════════════════════════════════════════════════════════════╣
║  ПРОФИЛЬ И ПРЕДПОЧТЕНИЯ                                          ║
║  /profile show / /profile <поле> <знач>                         ║
║  /pref show   / /pref <поле> <знач>                             ║
╠══════════════════════════════════════════════════════════════════╣
║  ПРОЧЕЕ                                                          ║
║  /memory            полное состояние памяти                      ║
║  /clear short       сбросить диалог текущего этапа               ║
║  /help              эта справка                                  ║
║  /quit              выход                                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

PROFILE_CMD_MAP = {
    "name": "name", "occupation": "occupation", "grade": "grade", "stack": "stack",
    "имя": "name", "деятельность": "occupation", "грейд": "grade", "стек": "stack",
}
PREF_CMD_MAP = {
    "style": "style", "format": "format", "language": "language", "extras": "extras",
    "стиль": "style", "формат": "format", "язык": "language", "пожелания": "extras",
}


def _section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def show_memory() -> None:
    print("\n" + "═" * 60 + "\n  СОСТОЯНИЕ ПАМЯТИ\n" + "═" * 60)

    _section("SHORT-TERM  (текущий диалог)")
    msgs = short.get_messages()
    if not msgs:
        print("  (пусто)")
    else:
        for i, m in enumerate(msgs, 1):
            role = "Вы   " if m["role"] == "user" else "Агент"
            content = m["content"]
            if isinstance(content, str):
                text = content
            else:
                parts = []
                for b in content:
                    t = b.get("type", "")
                    if t == "text":       parts.append(b["text"])
                    elif t == "tool_use": parts.append(f"[tool:{b['name']}]")
                    elif t == "tool_result": parts.append(f"[result:{b['content']}]")
                text = " | ".join(parts)
            max_w = 54
            idx: int | str = i
            while text:
                print(f"  {idx:>2}. {role}: {text[:max_w]}")
                text = text[max_w:]
                idx, role = "", "     "

    _section("WORKING  (задачи и этапы)")
    tasks = working.get_tasks()
    if not tasks:
        print("  (нет задач)")
    else:
        cur = working.get_current_task()
        cur_id = cur["id"] if cur else ""
        for t in tasks:
            marker = "→" if t["id"] == cur_id else " "
            plan_info = f"  [{t['step_index']+1}/{len(t['plan'])}]" if t["stage"] == "execution" and t["plan"] else ""
            print(f"  {marker} [{t['stage']}{plan_info}] {t['name']}  ({t['expected_action']})")
            if t.get("plan"):
                for i, step in enumerate(t["plan"]):
                    done = i < t["step_index"] or t["stage"] in ("validation", "done")
                    active = (i == t["step_index"] and t["stage"] == "execution")
                    icon = "✓" if done else ("→" if active else " ")
                    note = t["step_notes"][i] if i < len(t["step_notes"]) and t["step_notes"][i] else ""
                    suffix = f"  → {note[:40]}" if note else ""
                    print(f"       {icon} [{i+1}] {step}{suffix}")

    _section("LONG-TERM  (профиль)")
    for key, label in PROFILE_FIELDS.items():
        val = long_.load().get(key) or "(не указано)"
        print(f"  {label}: {val}")

    _section("LONG-TERM  (предпочтения)")
    prefs = long_.load_prefs()
    any_p = False
    for key, label in PREF_FIELDS.items():
        if prefs.get(key):
            print(f"  {label}: {prefs[key]}")
            any_p = True
    if not any_p:
        print("  (не заданы)")

    _section("ИНВАРИАНТЫ")
    inv_list = invariants.get_all()
    if not inv_list:
        print("  (не заданы)")
    else:
        for inv in inv_list:
            label = INVARIANT_CATEGORIES.get(inv["category"], inv["category"])
            print(f"  [{inv['id']}] {label}: {inv['rule']}")

    print("\n" + "═" * 60 + "\n")


def handle_command(raw: str) -> bool:
    cmd = raw.strip()

    if cmd == "/help":
        print(HELP_TEXT)
        return True

    if cmd == "/memory":
        show_memory()
        return True

    if cmd == "/status":
        show_status()
        return True

    if cmd in ("/quit", "/exit"):
        print("До свидания!")
        sys.exit(0)

    if cmd == "/confirm":
        print(do_confirm())
        return True

    if cmd == "/back":
        _, message = working.rollback()
        short.clear()
        print(f"  {message}")
        print("Агент: ", end="", flush=True)
        print(_kickstart_current_stage())
        return True

    if cmd == "/clear short":
        short.clear()
        print("[SHORT-TERM] Диалог этапа сброшен.")
        return True

    if cmd.startswith("/note "):
        err = working.add_note(cmd[6:].strip())
        print("Заметка добавлена." if not err else f"Ошибка: {err}")
        return True

    if cmd.startswith("/task"):
        rest = cmd[5:].strip()
        if rest == "list":
            tasks = working.get_tasks()
            if not tasks:
                print("Задач нет. Создайте: /task new <название>")
            else:
                cur = working.get_current_task()
                cur_id = cur["id"] if cur else ""
                print()
                for t in tasks:
                    marker = "→" if t["id"] == cur_id else " "
                    print(f"  {marker} [{t['stage']}] {t['name']}")
                print()
        elif rest.startswith("new "):
            name = rest[4:].strip()
            err = working.add_task(name)
            if not err:
                short.clear()
                print(f"[WORKING] Задача создана: {name}")
                print(f"  → запускаю агент планирования…\n")
                print("Агент: ", end="", flush=True)
                print(_kickstart_current_stage())
            else:
                print(f"Ошибка: {err}")
        elif rest.startswith("use "):
            name = rest[4:].strip()
            err = working.set_current(name)
            if not err:
                short.clear()
                task = working.get_current_task()
                if task and task["expected_action"] == WAITING:
                    print(f"[WORKING] Текущая задача: {name}")
                    print(f"\nОжидает подтверждения:\n{task['pending_result']}")
                    print("Введите /confirm для продолжения.")
                else:
                    print(f"[WORKING] Текущая задача: {name}")
                    print(f"  → продолжаю агент этапа «{task['stage']}»…\n")
                    print("Агент: ", end="", flush=True)
                    print(_kickstart_current_stage())
            else:
                print(f"Ошибка: {err}")
        else:
            print("Использование: /task new|list|use")
        return True

    if cmd.startswith("/profile"):
        rest = cmd[8:].strip()
        if not rest or rest == "show":
            print()
            for key, label in PROFILE_FIELDS.items():
                print(f"  {label}: {long_.load().get(key) or '(не указано)'}")
            print()
        else:
            parts = rest.split(" ", 1)
            if len(parts) == 2:
                fk = PROFILE_CMD_MAP.get(parts[0].lower())
                if fk:
                    err = long_.set_field(fk, parts[1].strip())
                    print(f"[ПРОФИЛЬ] {PROFILE_FIELDS[fk]}: {parts[1].strip()}" if not err else f"Ошибка: {err}")
                else:
                    print(f"Неизвестное поле: {parts[0]}")
        return True

    if cmd.startswith("/pref"):
        rest = cmd[5:].strip()
        if not rest or rest == "show":
            print()
            for key, label in PREF_FIELDS.items():
                print(f"  {label}: {long_.load_prefs().get(key) or '(не задано)'}")
            print()
        else:
            parts = rest.split(" ", 1)
            if len(parts) == 2:
                fk = PREF_CMD_MAP.get(parts[0].lower())
                if fk:
                    err = long_.set_pref(fk, parts[1].strip())
                    print(f"[ПРЕДПОЧТЕНИЯ] {PREF_FIELDS[fk]}: {parts[1].strip()}" if not err else f"Ошибка: {err}")
                else:
                    print(f"Неизвестное поле: {parts[0]}")
        return True

    if cmd.startswith("/inv"):
        rest = cmd[4:].strip()
        if not rest or rest == "list":
            inv_list = invariants.get_all()
            if not inv_list:
                print("  Инвариантов нет. Добавьте: /inv add <кат> <правило>")
                print(f"  Категории: {', '.join(INVARIANT_CATEGORIES)}")
            else:
                print()
                for inv in inv_list:
                    label = INVARIANT_CATEGORIES.get(inv["category"], inv["category"])
                    print(f"  [{inv['id']}] {label}: {inv['rule']}")
                print()
        elif rest.startswith("add "):
            parts = rest[4:].strip().split(" ", 1)
            if len(parts) < 2:
                print("  Использование: /inv add <категория> <правило>")
                print(f"  Категории: {', '.join(INVARIANT_CATEGORIES)}")
            else:
                cat, rule = parts
                err = invariants.add(cat.strip(), rule.strip())
                if err:
                    print(f"  Ошибка: {err}")
                else:
                    label = INVARIANT_CATEGORIES.get(cat.strip(), cat.strip())
                    print(f"  ✓ Инвариант добавлен [{label}]: {rule.strip()}")
        elif rest.startswith("del "):
            inv_id = rest[4:].strip()
            err = invariants.remove(inv_id)
            print(f"  Ошибка: {err}" if err else f"  ✓ Инвариант {inv_id} удалён")
        else:
            print("  Использование: /inv list | /inv add <кат> <правило> | /inv del <id>")
        return True

    if cmd.startswith("/"):
        print(f"Неизвестная команда: {cmd}  (/help для справки)")
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def startup_flow() -> None:
    # 1. Profile
    if not long_.is_complete():
        data = long_.load()
        missing = [label for key, label in PROFILE_FIELDS.items() if not data[key].strip()]
        print(f"\n⚠  Профиль не заполнен: {', '.join(missing)}.")
        if _ask("Заполнить? (y/n): ").lower() == "y":
            for key, label in PROFILE_FIELDS.items():
                cur = data.get(key, "")
                val = _ask(f"  {label}" + (f" [{cur}]" if cur else "") + ": ")
                if val:
                    long_.set_field(key, val)
            print("[ПРОФИЛЬ] Сохранён.\n")

    # 2. Preferences (first time only)
    if not long_.has_prefs():
        print("\n💬 Предпочтения общения не заданы.")
        if _ask("Настроить сейчас? (y/n): ").lower() == "y":
            hints = {
                "style":    "неформальный / формальный / технический",
                "format":   "кратко / подробно / с примерами кода",
                "language": "русский / английский",
                "extras":   "без эмодзи / не повторяй вопрос / ...",
            }
            for key, label in PREF_FIELDS.items():
                val = _ask(f"  {label} ({hints[key]}): ")
                if val:
                    long_.set_pref(key, val)
            print("[ПРЕДПОЧТЕНИЯ] Сохранены.\n")
        else:
            print("  Ок — скажи в разговоре (напр: 'отвечай покороче').\n")

    if not invariants.get_all():
        print("🔒 Инварианты не заданы (ограничения по стеку, архитектуре, бизнес-правила).")
        if _ask("Добавить сейчас? (y/n): ").lower() == "y":
            cat_list = " / ".join(f"{k} — {v}" for k, v in INVARIANT_CATEGORIES.items())
            print(f"  Категории: {cat_list}")
            print("  Вводи по одному. Пустая строка — завершить.\n")
            count = 0
            while True:
                cat = _ask("  Категория: ").strip()
                if not cat:
                    break
                rule = _ask("  Правило: ").strip()
                if not rule:
                    break
                err = invariants.add(cat, rule)
                if err:
                    print(f"  ⚠ {err}")
                else:
                    count += 1
                    print(f"  ✓ Добавлен [{INVARIANT_CATEGORIES.get(cat, cat)}]: {rule}")
            if count:
                print(f"[ИНВАРИАНТЫ] Сохранено: {count}.\n")
            else:
                print("  Ок — добавить позже: /inv add <кат> <правило>\n")
        else:
            print("  Ок — добавить позже: /inv add <кат> <правило>\n")

    # 3. Active tasks — resume or create
    active = working.get_active_tasks()
    if active:
        print(f"\n📋 Незавершённые задачи ({len(active)}):")
        for i, t in enumerate(active, 1):
            stage = t["stage"]
            if stage == "execution" and t["plan"]:
                stage += f" ({len(t['plan'])} шагов)"
            action_hint = "  ← /confirm" if t["expected_action"] == WAITING else ""
            print(f"  {i}. [{stage}] {t['name']}{action_hint}")
        print("  n. Создать новую\n  (Enter — пропустить)")
        ans = _ask("Выберите (номер / n / Enter): ")
        if ans.isdigit() and 1 <= int(ans) <= len(active):
            chosen = active[int(ans) - 1]
            working.set_current(chosen["name"])
            print(f"\n[ОРКЕСТРАТОР] Возобновляем: {chosen['name']}")
            if chosen["expected_action"] == WAITING and chosen["pending_result"]:
                # Sub-agent finished, waiting for user /confirm
                print(f"\nОжидает подтверждения:\n{chosen['pending_result']}")
                print("Введите /confirm для продолжения.\n")
            else:
                # Sub-agent was mid-work — kick it off immediately
                print(f"  → продолжаю агент этапа «{chosen['stage']}»…\n")
                print("Агент: ", end="", flush=True)
                print(_kickstart_current_stage())
        elif ans.lower() == "n":
            name = _ask("Название новой задачи: ")
            if name:
                err = working.add_task(name)
                if not err:
                    print(f"[РАБОЧАЯ ПАМЯТЬ] Задача создана: {name}")
                    print("  → запускаю агент планирования…\n")
                    print("Агент: ", end="", flush=True)
                    print(_kickstart_current_stage())
    else:
        print("\n📋 Незавершённых задач нет.")
        if _ask("Создать новую задачу? (y/n): ").lower() == "y":
            name = _ask("Название задачи: ")
            if name:
                err = working.add_task(name)
                if not err:
                    print(f"[РАБОЧАЯ ПАМЯТЬ] Задача создана: {name}")
                    print("  → запускаю агент планирования…\n")
                    print("Агент: ", end="", flush=True)
                    print(_kickstart_current_stage())


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 66)
    print("  Агент-оркестратор  |  /help — справка  |  /status — текущий этап")
    print("=" * 66)

    startup_flow()
    show_status()

    while True:
        task = working.get_current_task()
        stage = task["stage"] if task else "—"
        try:
            user_input = input(f"[{stage}] Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_input:
            continue

        if handle_command(user_input):
            continue

        print("Агент: ", end="", flush=True)
        reply = orchestrate(user_input)
        print(reply)
        print()


if __name__ == "__main__":
    main()