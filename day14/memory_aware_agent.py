# В классе MemoryAwareAgent обновить __init__ и ask

class MemoryAwareAgent:
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