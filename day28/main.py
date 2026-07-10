#!/usr/bin/env python3
"""
AI Challenge - Локальный RAG-чат с интеграцией Ollama
Использует локальную модель qwen2.5-coder:1.5b для генерации ответов
и локальный RAG-индексатор для поиска по кодовой базе
"""

import requests
import json
import sys
import os
import re
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

# ============================================================================
# ИМПОРТ RAG-КОМПОНЕНТОВ ИЗ ПРЕДЫДУЩЕГО ЗАДАНИЯ
# ============================================================================

# Здесь должен быть импорт вашего rag_indexer.py
# Если он в той же папке, то:
from rag_indexer import (
    RAGIndexer, IndexData, Chunk, Document,
    FixedSizeChunker, StructureChunker,
    compare_strategies
)

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:1.5b"
HISTORY_FILE = "chat_history.json"
RAG_INDEX_DIR = "rag_index"
FIXED_INDEX_FILE = "fixed_index.json"  # Ваш готовый индекс


# ============================================================================
# ЛОКАЛЬНЫЙ АГЕНТ С RAG
# ============================================================================

class LocalRAGChat:
    """
    Локальный чат-агент с поддержкой RAG.
    Использует Ollama для генерации и локальный индекс для поиска.
    """

    def __init__(self):
        self.history = []
        self.load_history()
        self.check_ollama()

        # Инициализация RAG
        print("\n📚 Инициализация RAG-индексатора...")

        # Создаём RAGIndexer (без auth_key для локальной модели)
        self.rag_indexer = RAGIndexer(
            auth_key="",  # Для локальной модели ключ не нужен
            source_dir="."
        )

        # Загружаем индекс из fixed_index.json
        self._load_fixed_index()

        # Статистика
        self.total_tokens = 0
        self.request_count = 0

    def _load_fixed_index(self):
        """Загрузка индекса из fixed_index.json"""
        fpath = os.path.join(RAG_INDEX_DIR, FIXED_INDEX_FILE)

        if os.path.exists(fpath):
            try:
                index = self.rag_indexer.index_manager.load(fpath)
                self.rag_indexer.indexes["fixed_size"] = index
                print(f"  ✅ Загружен индекс: fixed_size ({len(index.chunks)} чанков)")
                print(f"  📊 Статистика:")
                print(f"     - Всего чанков: {index.stats.get('total_chunks', 0)}")
                print(f"     - Всего символов: {index.stats.get('total_chars', 0):,}")
                print(f"     - Средний размер: {index.stats.get('avg_chunk_size', 0)} символов")
                print(f"     - Документов: {index.stats.get('total_documents', 0)}")
            except Exception as e:
                print(f"  ❌ Ошибка загрузки индекса: {e}")
                print(f"  ℹ️  Используйте /index для создания нового индекса")
        else:
            print(f"  ⚠️ Файл {fpath} не найден")
            print(f"  ℹ️  Используйте /index для создания индекса")

    def check_ollama(self):
        """Проверяет доступность Ollama"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if MODEL_NAME not in model_names:
                    print(f"⚠️ Модель '{MODEL_NAME}' не найдена в Ollama")
                    print(f"📥 Установите её командой: ollama pull {MODEL_NAME}")
                    sys.exit(1)
                print(f"✅ Модель '{MODEL_NAME}' успешно загружена")
                return True
        except requests.exceptions.ConnectionError:
            print("❌ Ошибка: Ollama не запущена!")
            print("🔧 Запустите Ollama и попробуйте снова")
            sys.exit(1)
        return False

    def ask(self, prompt: str, stream: bool = False, context: List[Dict] = None) -> str:
        """
        Отправляет запрос к локальной LLM

        Args:
            prompt (str): Текст запроса
            stream (bool): Включить потоковую передачу
            context (List[Dict]): Дополнительный контекст (для RAG)

        Returns:
            str: Ответ модели
        """
        # Формируем контекст из истории
        history_context = self.build_context()

        # Если есть RAG-контекст, добавляем его
        if context:
            rag_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in context if msg['role'] == 'system'
            ])
            full_prompt = f"{history_context}\n\n{rag_text}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"{history_context}\n\nUser: {prompt}\nAssistant:"

        payload = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }

        try:
            if stream:
                return self.ask_stream(payload)
            else:
                response = requests.post(OLLAMA_URL, json=payload, timeout=90)
                response.raise_for_status()
                result = response.json()
                return result.get('response', '')
        except requests.exceptions.Timeout:
            return "⏱️ Превышено время ожидания ответа от модели"
        except requests.exceptions.RequestException as e:
            return f"❌ Ошибка при запросе к Ollama: {str(e)}"

    def ask_stream(self, payload):
        """Потоковый режим для постепенного вывода ответа"""
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=90)
        response.raise_for_status()

        full_response = ""
        print("\n🤖 ", end="", flush=True)

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        chunk = data['response']
                        print(chunk, end="", flush=True)
                        full_response += chunk
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue

        print()
        return full_response

    def build_context(self) -> str:
        """Строит контекст из последних сообщений"""
        if not self.history:
            return "You are a helpful AI assistant specialized in code analysis. Answer concisely and accurately."

        context = "Previous conversation:\n"
        for msg in self.history[-5:]:
            context += f"{msg['role']}: {msg['content']}\n"
        return context

    def add_to_history(self, role: str, content: str):
        """Добавляет сообщение в историю"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save_history()

    def save_history(self):
        """Сохраняет историю в файл"""
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Не удалось сохранить историю: {e}")

    def load_history(self):
        """Загружает историю из файла"""
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"⚠️ Не удалось загрузить историю: {e}")
            self.history = []

    def clear_history(self):
        """Очищает историю диалога"""
        self.history = []
        self.save_history()
        print("🧹 История диалога очищена")

    def show_history(self):
        """Показывает последние сообщения"""
        if not self.history:
            print("📭 История пуста")
            return

        print("\n📜 Последние сообщения:")
        for msg in self.history[-10:]:
            time = msg.get('timestamp', '')[:16]
            print(f"{time} | {msg['role']}: {msg['content'][:50]}...")

    # ========== RAG-МЕТОДЫ (как в облачном примере) ==========

    def ask_with_rag(self, question: str, strategy: str = "fixed_size", top_k: int = 5) -> str:
        """
        Запрос к LLM с RAG: поиск чанков → контекст → LLM.
        """
        if "fixed_size" not in self.rag_indexer.indexes:
            return "❌ Индекс не загружен. Сначала выполните /index или убедитесь, что fixed_index.json существует."

        # Поиск релевантных чанков
        chunks = self.rag_indexer.search(question, strategy="fixed_size", top_k=top_k)

        if not chunks:
            return "❌ Нет релевантных чанков для вопроса."

        # Формируем контекст из чанков
        context_parts = []
        for chunk in chunks:
            header = f"[{chunk.source}] {chunk.section}"
            context_parts.append(f"{header}\n{chunk.content}")

        context_text = "\n\n---\n\n".join(context_parts)

        # Формируем контекст для LLM (как в облачном примере)
        rag_context = [
            {"role": "system", "content": "Ты — ассистент, отвечающий на вопросы по кодовой базе проекта."},
            {"role": "system",
             "content": f"Используй информацию из фрагментов кода ниже для ответа:\n\n{context_text}"},
            {"role": "system",
             "content": "Отвечай по делу, без лишних пояснений. Если информации недостаточно — так и скажи."}
        ]

        # Добавляем вопрос в историю
        self.add_to_history("user", question)

        # Получаем ответ от модели с RAG-контекстом
        response = self.ask(question, context=rag_context)

        # Добавляем ответ в историю
        self.add_to_history("assistant", response)

        return response

    def ask_without_rag(self, question: str) -> str:
        """Запрос к LLM без RAG (только вопрос)."""
        self.add_to_history("user", question)
        response = self.ask(question)
        self.add_to_history("assistant", response)
        return response

    def search_index(self, query: str, strategy: str = "fixed_size", top_k: int = 5) -> str:
        """Поиск по индексу без LLM."""
        if "fixed_size" not in self.rag_indexer.indexes:
            return "❌ Индекс не загружен. Сначала выполните /index"

        chunks = self.rag_indexer.search(query, strategy="fixed_size", top_k=top_k)

        if not chunks:
            return "🔍 Ничего не найдено."

        lines = [f"\n🔍 ПОИСК (запрос: '{query}')"]
        lines.append("=" * 70)

        # Получаем эмбеддинг запроса
        query_emb = self.rag_indexer.embedding_service.get_embedding(query)

        for i, chunk in enumerate(chunks, 1):
            score = self.rag_indexer._cosine_similarity(query_emb, chunk.embedding or [])
            lines.append(f"\n[{i}] (score: {score:.4f})")
            lines.append(f"    📁 {chunk.source} | {chunk.section}")
            lines.append(f"    📝 {chunk.content[:300]}...")

        return "\n".join(lines)

    def build_indexes(self, strategy: str = "both", file_filter: str = "main.py") -> str:
        """Создание индексов RAG."""
        if strategy not in ("fixed", "structure", "both"):
            return f"❌ Неизвестная стратегия: {strategy}. Используйте: fixed, structure, both"

        print(f"\n📚 Запуск индексации (стратегия: {strategy})...")
        result = self.rag_indexer.build_index(strategy=strategy, file_filter=file_filter)

        lines = ["✅ Индексация завершена!"]

        if "fixed_size" in result:
            idx = result["fixed_size"]
            lines.append(f"\n📊 Фиксированный размер:")
            lines.append(f"   Чанков: {idx.stats['total_chunks']}")
            lines.append(f"   Средний размер: {idx.stats['avg_chunk_size']} символов")

        if "structure" in result:
            idx = result["structure"]
            lines.append(f"\n🏗️ По структуре:")
            lines.append(f"   Чанков: {idx.stats['total_chunks']}")
            lines.append(f"   Средний размер: {idx.stats['avg_chunk_size']} символов")
            dist = idx.stats.get("type_distribution", {})
            for t, count in dist.items():
                lines.append(f"   {t}: {count}")

        return "\n".join(lines)

    def compare_rag(self, question: str) -> str:
        """Сравнение ответов с RAG и без RAG для одного вопроса."""
        print(f"\n📊 Сравнение RAG vs без RAG")
        print("=" * 70)
        print(f"Вопрос: {question}")
        print("=" * 70)

        # Без RAG
        print("\n❌ ОТВЕТ БЕЗ RAG:")
        print("-" * 70)
        without = self.ask_without_rag(question)
        print(f"{without}")

        # С RAG
        print("\n✅ ОТВЕТ С RAG:")
        print("-" * 70)
        with_rag = self.ask_with_rag(question)
        print(f"{with_rag}")

        # Вывод для сравнения
        print("\n" + "=" * 70)
        print("📝 РУЧНОЕ СРАВНЕНИЕ:")
        print("  • Без RAG: модель отвечает только на основе своих знаний")
        print("  • С RAG: модель использует информацию из кодовой базы")
        print("  • Оцените: какой ответ точнее и информативнее?")

        return ""

    def run_rag_comparison(self) -> str:
        """Запуск сравнения RAG vs без RAG по 10 вопросам."""
        questions = [
            "Какие классы памяти реализованы в проекте?",
            "Как называется главный класс агента?",
            "Какие MCP-сервера подключаются в агенте?",
            "Какая модель LLM используется по умолчанию?",
            "Какие стратегии чанкинга реализованы?",
            "Что такое инварианты (InvariantManager)?",
            "Какой протокол используется для MCP-коммуникации?",
            "Какая размерность эмбеддингов в проекте?",
            "Какой класс управляет задачами?",
            "Сколько уровней памяти у MemoryAwareAgent?",
        ]

        if "fixed_size" not in self.rag_indexer.indexes:
            return "❌ Индекс не загружен. Сначала выполните /index"

        lines = []
        lines.append("=" * 100)
        lines.append("📊 СРАВНЕНИЕ: БЕЗ RAG vs С RAG (локальная модель)")
        lines.append("=" * 100)

        for i, q in enumerate(questions, 1):
            lines.append(f"\n{'─' * 100}")
            lines.append(f"[{i}] Вопрос: {q}")

            # Без RAG
            without = self.ask_without_rag(q)
            lines.append(f"\n    ❌ БЕЗ RAG: {without[:300]}...")

            # С RAG
            with_rag = self.ask_with_rag(q)
            lines.append(f"\n    ✅ С RAG:   {with_rag[:300]}...")

        lines.append(f"\n{'=' * 100}")
        lines.append("✅ Сравнение завершено.")
        lines.append("📝 Оцените качество ответов визуально:")
        lines.append("  • Без RAG — общие знания модели")
        lines.append("  • С RAG — информация из кодовой базы")
        lines.append(f"{'=' * 100}")

        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Получение статистики"""
        fixed_index = self.rag_indexer.indexes.get("fixed_size")
        return {
            "model": MODEL_NAME,
            "history_length": len(self.history),
            "request_count": self.request_count,
            "rag_loaded": "fixed_size" in self.rag_indexer.indexes,
            "rag_chunks": len(fixed_index.chunks) if fixed_index else 0,
            "rag_chars": fixed_index.stats.get("total_chars", 0) if fixed_index else 0
        }

    def show_rag_status(self) -> str:
        """Показать статус RAG"""
        stats = self.get_stats()
        lines = ["\n🔎 RAG-СТАТУС"]
        lines.append("=" * 70)

        if stats["rag_loaded"]:
            lines.append(f"  ✅ Индекс загружен: fixed_size")
            lines.append(f"  📊 Чанков: {stats['rag_chunks']}")
            lines.append(f"  📝 Символов: {stats['rag_chars']:,}")
            lines.append(f"  🎯 Модель: {stats['model']}")
        else:
            lines.append("  ❌ Индекс НЕ загружен")
            lines.append("  ℹ️  Используйте /index для создания")

        lines.append(f"  📝 История: {stats['history_length']} сообщений")
        lines.append(f"  🔄 Запросов: {stats['request_count']}")
        lines.append("=" * 70)

        return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def print_help():
    """Выводит справку по командам"""
    print("""
📋 ОСНОВНЫЕ КОМАНДЫ:
  /help          - Показать эту справку
  /clear         - Очистить историю диалога
  /history       - Показать последние сообщения
  /model         - Показать текущую модель
  /stats         - Показать статистику
  /rag           - Показать статус RAG
  /exit          - Выйти из программы

📌 RAG-КОМАНДЫ (как в облачном примере):
  /index [fixed|structure|both] - Создать индексы (по умолчанию: both)
  /indexes                       - Список сохранённых индексов
  /search <запрос>               - Поиск по индексу (без LLM)
  /ask <вопрос>                  - Ответ С RAG (поиск + локальная LLM)
  /ask0 <вопрос>                 - Ответ БЕЗ RAG (только локальная LLM)
  /compare <вопрос>              - Сравнить RAG vs без RAG для одного вопроса
  /compare_all                   - Сравнить RAG vs без RAG по 10 вопросам

💡 Просто введите текст, чтобы задать вопрос модели (без RAG)
""")


def main():
    print("=" * 60)
    print("🤖 Local RAG Chat - Консольная утилита")
    print(f"📦 Модель: {MODEL_NAME}")
    print("🌐 Работает полностью локально (Ollama + RAG)")
    print("=" * 60)

    chat = LocalRAGChat()

    # Показываем статус RAG при старте
    print(chat.show_rag_status())

    print_help()

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()

            if not user_input:
                continue

            # Обработка команд
            if user_input.startswith('/'):
                command = user_input.lower()

                if command == '/exit':
                    print("👋 До свидания!")
                    break

                elif command == '/help':
                    print_help()

                elif command == '/clear':
                    chat.clear_history()

                elif command == '/history':
                    chat.show_history()

                elif command == '/model':
                    print(f"🧠 Текущая модель: {MODEL_NAME}")

                elif command == '/stats':
                    stats = chat.get_stats()
                    print("\n📊 СТАТИСТИКА:")
                    print(f"  Модель: {stats['model']}")
                    print(f"  Сообщений в истории: {stats['history_length']}")
                    print(f"  Запросов: {stats['request_count']}")
                    print(f"  RAG загружен: {'✅' if stats['rag_loaded'] else '❌'}")
                    if stats['rag_loaded']:
                        print(f"  Чанков в индексе: {stats['rag_chunks']}")
                        print(f"  Символов в индексе: {stats['rag_chars']:,}")

                elif command == '/rag':
                    print(chat.show_rag_status())

                # ===== RAG-КОМАНДЫ (как в облачном примере) =====

                elif command.startswith('/index'):
                    parts = user_input.split()
                    strategy = parts[1] if len(parts) > 1 else "both"
                    if strategy in ["fixed", "structure", "both"]:
                        result = chat.build_indexes(strategy)
                        print(result)
                    else:
                        print(f"❌ Неизвестная стратегия: {strategy}")
                        print("   Используйте: fixed, structure, both")

                elif command == '/indexes':
                    files = chat.rag_indexer.index_manager.list_indexes()
                    print("\n📋 СОХРАНЁННЫЕ ИНДЕКСЫ:")
                    if not files:
                        print("  (нет сохранённых индексов)")
                    else:
                        for f in files:
                            fpath = os.path.join(RAG_INDEX_DIR, f)
                            if os.path.exists(fpath):
                                size = os.path.getsize(fpath)
                                print(f"  📄 {f} ({size:,} байт)")

                elif command.startswith('/search '):
                    query = user_input[8:].strip()
                    if not query:
                        print("❌ Укажите запрос: /search <текст>")
                        continue
                    result = chat.search_index(query)
                    print(result)

                elif command.startswith('/ask '):
                    question = user_input[5:].strip()
                    if not question:
                        print("❌ Укажите вопрос: /ask <вопрос>")
                        continue
                    print(f"\n🔍 RAG-ЗАПРОС: {question}")
                    print("=" * 60)
                    chat.request_count += 1
                    response = chat.ask_with_rag(question)
                    print(f"\n🤖 {response}")

                elif command.startswith('/ask0 '):
                    question = user_input[6:].strip()
                    if not question:
                        print("❌ Укажите вопрос: /ask0 <вопрос>")
                        continue
                    print(f"\n🔍 ЗАПРОС БЕЗ RAG: {question}")
                    print("=" * 60)
                    chat.request_count += 1
                    response = chat.ask_without_rag(question)
                    print(f"\n🤖 {response}")

                elif command.startswith('/compare '):
                    question = user_input[9:].strip()
                    if not question:
                        print("❌ Укажите вопрос: /compare <вопрос>")
                        continue
                    chat.compare_rag(question)

                elif command == '/compare_all':
                    print(chat.run_rag_comparison())

                else:
                    print(f"⚠️ Неизвестная команда: {command}")
                    print("   Введите /help для списка команд")
                continue

            # Обычное сообщение (без RAG)
            chat.request_count += 1
            response = chat.ask_without_rag(user_input)
            print(f"\n🤖 {response}")

        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()