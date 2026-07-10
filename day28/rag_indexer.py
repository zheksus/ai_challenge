import os
import re
import json
import ast
import time
import uuid
import hashlib
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from sentence_transformers import SentenceTransformer


#
# import requests
# import numpy as np


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass
class Document:
    source: str
    content: str
    title: str = ""


@dataclass
class Chunk:
    content: str
    source: str
    title: str
    section: str
    chunk_id: str
    chunk_type: str  # 'fixed' or 'structure'
    strategy: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "strategy": self.strategy,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }


@dataclass
class IndexData:
    strategy_name: str
    chunks: List[Chunk]
    created_at: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "created_at": self.created_at or datetime.now().isoformat(),
            "stats": self.stats,
            "chunks": [c.to_dict() for c in self.chunks],
        }


# ============================================================================
# LOCAL EMBEDDING SERVICE (БЕЗ SBERBANK)
# ============================================================================

class LocalEmbeddingService:
    """
    Локальный сервис эмбеддингов.
    Использует sentence-transformers для генерации эмбеддингов локально.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", debug: bool = False):
        """
        Args:
            model_name: Название модели sentence-transformers
                       Доступные модели:
                       - "all-MiniLM-L6-v2" (384 dims, ~80MB) - быстрая
                       - "all-mpnet-base-v2" (768 dims, ~420MB) - точная
                       - "paraphrase-MiniLM-L3-v2" (384 dims, ~30MB) - самая лёгкая
                       - "distiluse-base-multilingual-cased-v2" (512 dims, ~300MB) - мультиязычная
        """
        self.model_name = model_name
        self.debug = debug
        self._model = None
        self._call_count = 0
        self._total_chars = 0
        self._cache: Dict[str, List[float]] = {}
        self._cache_file = "embedding_cache.pkl"
        self._load_cache()

        # Инициализация модели при первом вызове
        self._init_model()

    def _init_model(self):
        """Ленивая инициализация модели"""
        if self._model is not None:
            return

        try:
            if self.debug:
                print(f"  📥 Загрузка модели {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            if self.debug:
                print(f"  ✅ Модель загружена. Размерность: {self._model.get_sentence_embedding_dimension()}")
        except ImportError:
            print("❌ Установите sentence-transformers: pip install sentence-transformers")
            raise
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise

    def _load_cache(self):
        """Загрузка кэша эмбеддингов"""
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                if self.debug:
                    print(f"  📂 Загружен кэш эмбеддингов: {len(self._cache)} записей")
            except Exception as e:
                if self.debug:
                    print(f"  ⚠️ Ошибка загрузки кэша: {e}")
                self._cache = {}

    def _save_cache(self):
        """Сохранение кэша эмбеддингов"""
        try:
            # Ограничиваем размер кэша для экономии памяти
            if len(self._cache) > 10000:
                # Удаляем старые записи (по ключу)
                keys = sorted(self._cache.keys())
                for key in keys[:len(self._cache) - 5000]:
                    del self._cache[key]

            with open(self._cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            if self.debug:
                print(f"  ⚠️ Ошибка сохранения кэша: {e}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Получение эмбеддинга для текста.
        Использует кэширование для ускорения.
        """
        self._call_count += 1
        self._total_chars += len(text)

        # Создаём хэш текста для кэша
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

        # Проверяем кэш
        if text_hash in self._cache:
            if self.debug:
                print(f"  ✅ [CACHE] {text_hash[:8]}...")
            return self._cache[text_hash]

        # Генерируем эмбеддинг
        if self.debug:
            print(f"  📤 [EMBED #{self._call_count}] {len(text)} символов")

        try:
            self._init_model()
            embedding = self._model.encode(text, normalize_embeddings=True)
            emb_list = embedding.tolist()

            # Сохраняем в кэш
            self._cache[text_hash] = emb_list
            if len(self._cache) % 100 == 0:
                self._save_cache()

            if self.debug:
                print(f"  ✅ Эмбеддинг получен (размерность: {len(emb_list)})")

            return emb_list

        except Exception as e:
            print(f"  ❌ Ошибка генерации эмбеддинга: {e}")
            # Возвращаем нулевой вектор в случае ошибки
            return [0.0] * 384

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Получение эмбеддингов для списка текстов"""
        return self.get_embeddings_batch(texts)

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Пакетное получение эмбеддингов.
        """
        if self.debug:
            print(f"\n📦 [BATCH] Всего текстов: {len(texts)}")
            total_chars = sum(len(t) for t in texts)
            print(f"📦 [BATCH] Всего символов: {total_chars:,}")

        # Проверяем, сколько текстов уже в кэше
        results = []
        texts_to_encode = []
        text_indices = []

        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if text_hash in self._cache:
                results.append((i, self._cache[text_hash]))
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
                results.append((i, None))

        # Кодируем недостающие тексты
        if texts_to_encode:
            self._init_model()

            for i in range(0, len(texts_to_encode), batch_size):
                batch = texts_to_encode[i:i + batch_size]
                indices = text_indices[i:i + batch_size]

                if self.debug:
                    print(f"  📦 Батч {i // batch_size + 1}: {len(batch)} текстов")

                try:
                    embeddings = self._model.encode(batch, normalize_embeddings=True)

                    for j, emb in enumerate(embeddings):
                        emb_list = emb.tolist()
                        text_hash = hashlib.md5(batch[j].encode('utf-8')).hexdigest()
                        self._cache[text_hash] = emb_list

                        # Обновляем результат
                        idx = indices[j]
                        results[idx] = (idx, emb_list)

                except Exception as e:
                    print(f"  ❌ Ошибка кодирования батча: {e}")
                    # Вставляем нулевые векторы
                    for j, idx in enumerate(indices):
                        results[idx] = (idx, [0.0] * 384)

        # Сортируем по индексу
        results.sort(key=lambda x: x[0])

        # Сохраняем кэш
        if len(self._cache) % 100 == 0:
            self._save_cache()

        return [r[1] for r in results]


# ============================================================================
# DOCUMENT LOADER
# ============================================================================

class DocumentLoader:
    EXTENSIONS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini"}

    def __init__(self, source_dir: str):
        self.source_dir = source_dir

    def load_all(self) -> List[Document]:
        docs = []
        for root, dirs, files in os.walk(self.source_dir):
            dirs[:] = [d for d in dirs if
                       not d.startswith((".", "_")) and d not in ("venv", ".venv", "__pycache__", ".idea")]
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.EXTENSIONS:
                    continue
                fpath = os.path.join(root, fname)
                if os.path.getsize(fpath) == 0:
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception as e:
                    print(f"  ⚠️  Ошибка чтения {fpath}: {e}")
                    continue
                relpath = os.path.relpath(fpath, os.path.dirname(self.source_dir))
                docs.append(Document(source=relpath, content=content, title=fname))
        return docs


# ============================================================================
# CHUNKING STRATEGIES
# ============================================================================

class BaseChunker:
    strategy_name: str = "base"

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    strategy_name = "fixed_size"

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        chunks = []
        for doc in documents:
            text = doc.content
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                if end < len(text):
                    end = text.rfind("\n", start, end)
                    if end == -1 or end <= start:
                        end = min(start + self.chunk_size, len(text))
                content = text[start:end].strip()
                if content:
                    chunks.append(Chunk(
                        content=content,
                        source=doc.source,
                        title=doc.title,
                        section=f"offset_{start}",
                        chunk_id=f"{doc.title}_fixed_{chunk_idx:04d}",
                        chunk_type="fixed",
                        strategy=self.strategy_name,
                        metadata={"offset_start": start, "offset_end": end, "char_count": len(content)},
                    ))
                    chunk_idx += 1
                start = end - self.overlap if end < len(text) else len(text)
        return chunks


class StructureChunker(BaseChunker):
    """
    Чанкинг по структуре кода с разбиением больших чанков.
    Сохраняет метаданные о принадлежности к родительскому элементу.
    """
    strategy_name = "structure"
    MAX_CHUNK_CHARS = 1800  # ~450 токенов

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        chunks = []
        for doc in documents:
            if doc.source.endswith(".py"):
                chunks.extend(self._chunk_python(doc))
            elif doc.source.endswith(".md"):
                chunks.extend(self._chunk_markdown(doc))
            else:
                chunks.extend(self._chunk_by_lines(doc))
        return chunks

    def _chunk_python(self, doc: Document) -> List[Chunk]:
        """Разбивает Python-файл на классы, функции и методы с ограничением размера."""
        chunks = []
        try:
            tree = ast.parse(doc.content)
        except SyntaxError:
            return self._chunk_by_lines(doc)

        lines = doc.content.split("\n")
        top_nodes = [n for n in ast.iter_child_nodes(tree) if
                     isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]

        if not top_nodes:
            return self._chunk_by_lines(doc)

        # 1. ПРЕАМБУЛА (импорты, глобальные переменные)
        preamble_end = top_nodes[0].lineno - 1 if top_nodes else 0
        if preamble_end > 0:
            preamble = "\n".join(lines[:preamble_end]).strip()
            if preamble:
                chunks.append(Chunk(
                    content=preamble,
                    source=doc.source,
                    title=doc.title,
                    section="preamble",
                    chunk_id=f"{doc.title}_struct_preamble",
                    chunk_type="preamble",
                    strategy=self.strategy_name,
                    metadata={
                        "node_type": "preamble",
                        "start_line": 1,
                        "end_line": preamble_end,
                        "is_partial": False,
                        "parent": None,
                    },
                ))

        # 2. ОБРАБОТКА УЗЛОВ
        for node in top_nodes:
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._chunk_class(node, doc, lines))
            else:  # FunctionDef или AsyncFunctionDef
                chunks.append(self._chunk_function(node, doc, lines, parent_class=None))

        return chunks

    def _chunk_class(self, node: ast.ClassDef, doc: Document, lines: List[str]) -> List[Chunk]:
        """
        Разбивает класс на:
        1. Сам класс (если не слишком большой)
        2. Каждый метод как отдельный чанк
        """
        chunks = []
        start_line = getattr(node, "lineno", 1) - 1
        end_line = getattr(node, "end_lineno", start_line + 1)

        # Получаем методы класса
        methods = [n for n in ast.iter_child_nodes(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

        # 2.1. КАЖДЫЙ МЕТОД — ОТДЕЛЬНЫЙ ЧАНК
        for method in methods:
            m_start = getattr(method, "lineno", 1) - 1
            m_end = getattr(method, "end_lineno", m_start + 1)
            m_content = "\n".join(lines[m_start:m_end]).strip()
            if m_content:
                chunks.append(Chunk(
                    content=m_content,
                    source=doc.source,
                    title=doc.title,
                    section=f"class:{node.name}.method:{method.name}",
                    chunk_id=f"{doc.title}_struct_{len(chunks):04d}",
                    chunk_type="method",
                    strategy=self.strategy_name,
                    metadata={
                        "node_type": "method",
                        "class": node.name,
                        "method": method.name,
                        "start_line": m_start + 1,
                        "end_line": m_end,
                        "is_partial": False,
                        "parent": f"class:{node.name}",
                    },
                ))

        # 2.2. САМ КЛАСС (без методов, только атрибуты и документация)
        # Вырезаем методы из тела класса
        method_ranges = [(getattr(m, "lineno", 1) - 1, getattr(m, "end_lineno", getattr(m, "lineno", 1))) for m in
                         methods]
        class_lines = []
        for i in range(start_line, end_line):
            is_method_line = False
            for ms, me in method_ranges:
                if ms <= i < me:
                    is_method_line = True
                    break
            if not is_method_line:
                class_lines.append(lines[i])

        class_content = "\n".join(class_lines).strip()

        # Добавляем сам класс, если он не пустой
        if class_content:
            # Проверяем размер
            if len(class_content) <= self.MAX_CHUNK_CHARS:
                chunks.append(Chunk(
                    content=class_content,
                    source=doc.source,
                    title=doc.title,
                    section=f"class:{node.name}",
                    chunk_id=f"{doc.title}_struct_{len(chunks):04d}",
                    chunk_type="class",
                    strategy=self.strategy_name,
                    metadata={
                        "node_type": "class",
                        "class": node.name,
                        "start_line": start_line + 1,
                        "end_line": end_line,
                        "is_partial": False,
                        "parent": None,
                    },
                ))
            else:
                # Если класс большой — разбиваем на части
                chunks.extend(self._split_large_chunk(
                    content=class_content,
                    source=doc.source,
                    title=doc.title,
                    section=f"class:{node.name}",
                    chunk_type="class_part",
                    metadata={
                        "node_type": "class",
                        "class": node.name,
                        "start_line": start_line + 1,
                        "end_line": end_line,
                        "is_partial": True,
                        "parent": None,
                    },
                    max_chars=self.MAX_CHUNK_CHARS
                ))

        return chunks

    def _chunk_function(self, node: ast.FunctionDef, doc: Document, lines: List[str],
                        parent_class: str = None) -> Chunk:
        """Создаёт чанк для функции."""
        start_line = getattr(node, "lineno", 1) - 1
        end_line = getattr(node, "end_lineno", start_line + 1)
        content = "\n".join(lines[start_line:end_line]).strip()

        section = f"function:{node.name}"
        if parent_class:
            section = f"class:{parent_class}.function:{node.name}"

        metadata = {
            "node_type": "function",
            "function": node.name,
            "start_line": start_line + 1,
            "end_line": end_line,
        }
        if parent_class:
            metadata["class"] = parent_class

        return Chunk(
            content=content,
            source=doc.source,
            title=doc.title,
            section=section,
            chunk_id=f"{doc.title}_struct_{uuid.uuid4().hex[:6]}",
            chunk_type="function",
            strategy=self.strategy_name,
            metadata=metadata,
        )

    def _split_large_chunk(self, content: str, source: str, title: str, section: str,
                           chunk_type: str, metadata: Dict, max_chars: int) -> List[Chunk]:
        """
        Разбивает большой чанк на под-чанки с сохранением метаданных.
        """
        if len(content) <= max_chars:
            return [Chunk(
                content=content,
                source=source,
                title=title,
                section=section,
                chunk_id=f"{title}_struct_{uuid.uuid4().hex[:6]}",
                chunk_type=chunk_type,
                strategy=self.strategy_name,
                metadata={**metadata, "total_parts": 1, "part": 1},
            )]

        lines = content.split("\n")
        chunks = []
        current_lines = []
        current_len = 0
        part = 1

        for line in lines:
            if current_len + len(line) + 1 > max_chars and current_lines:
                chunks.append(Chunk(
                    content="\n".join(current_lines),
                    source=source,
                    title=title,
                    section=f"{section}_part{part}",
                    chunk_id=f"{title}_struct_{uuid.uuid4().hex[:6]}",
                    chunk_type=chunk_type,
                    strategy=self.strategy_name,
                    metadata={
                        **metadata,
                        "part": part,
                        "total_parts": 0,  # будет обновлено позже
                        "is_partial": True,
                        "parent_section": section,
                    },
                ))
                part += 1
                current_lines = []
                current_len = 0
            current_lines.append(line)
            current_len += len(line) + 1

        if current_lines:
            chunks.append(Chunk(
                content="\n".join(current_lines),
                source=source,
                title=title,
                section=f"{section}_part{part}",
                chunk_id=f"{title}_struct_{uuid.uuid4().hex[:6]}",
                chunk_type=chunk_type,
                strategy=self.strategy_name,
                metadata={
                    **metadata,
                    "part": part,
                    "total_parts": 0,  # будет обновлено позже
                    "is_partial": True,
                    "parent_section": section,
                },
            ))

        # Обновляем total_parts во всех чанках
        total = len(chunks)
        for c in chunks:
            c.metadata["total_parts"] = total

        return chunks

    def _chunk_markdown(self, doc: Document) -> List[Chunk]:
        """Разбивает Markdown-файл по заголовкам с ограничением размера."""
        chunks = []
        lines = doc.content.split("\n")
        current_section = "header"
        current_lines = []
        section_idx = 0

        def flush():
            nonlocal section_idx
            content = "\n".join(current_lines).strip()
            if content:
                if len(content) <= self.MAX_CHUNK_CHARS:
                    chunks.append(Chunk(
                        content=content,
                        source=doc.source,
                        title=doc.title,
                        section=current_section,
                        chunk_id=f"{doc.title}_struct_{len(chunks):04d}",
                        chunk_type="section",
                        strategy=self.strategy_name,
                        metadata={"section_type": "markdown_section", "section_idx": section_idx,
                                  "is_partial": False},
                    ))
                else:
                    # Разбиваем большой раздел
                    chunks.extend(self._split_large_chunk(
                        content=content,
                        source=doc.source,
                        title=doc.title,
                        section=current_section,
                        chunk_type="section_part",
                        metadata={"section_type": "markdown_section", "section_idx": section_idx,
                                  "is_partial": True},
                        max_chars=self.MAX_CHUNK_CHARS
                    ))
                section_idx += 1

        for line in lines:
            if line.startswith("#"):
                flush()
                current_section = line.lstrip("# ").strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        flush()
        return chunks

    def _chunk_by_lines(self, doc: Document) -> List[Chunk]:
        """Fallback: разбивка по строкам с ограничением размера."""
        chunks = []
        lines = doc.content.split("\n")
        current_lines = []
        current_len = 0
        chunk_idx = 0

        for line in lines:
            if current_len + len(line) + 1 > self.MAX_CHUNK_CHARS and current_lines:
                chunks.append(Chunk(
                    content="\n".join(current_lines),
                    source=doc.source,
                    title=doc.title,
                    section=f"lines_{chunk_idx * 20 + 1}_{chunk_idx * 20 + len(current_lines)}",
                    chunk_id=f"{doc.title}_struct_{len(chunks):04d}",
                    chunk_type="lines",
                    strategy=self.strategy_name,
                    metadata={"is_partial": False},
                ))
                chunk_idx += 1
                current_lines = []
                current_len = 0
            current_lines.append(line)
            current_len += len(line) + 1

        if current_lines:
            chunks.append(Chunk(
                content="\n".join(current_lines),
                source=doc.source,
                title=doc.title,
                section=f"lines_{chunk_idx * 20 + 1}_end",
                chunk_id=f"{doc.title}_struct_{len(chunks):04d}",
                chunk_type="lines",
                strategy=self.strategy_name,
                metadata={"is_partial": False},
            ))

        return chunks


# ============================================================================
# INDEX MANAGER
# ============================================================================

class IndexManager:
    def __init__(self, index_dir: str = "rag_index"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

    def save(self, index: IndexData, filename: str = "") -> str:
        if not filename:
            filename = f"index_{index.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        fpath = os.path.join(self.index_dir, filename)

        # Конвертируем numpy массивы в списки для JSON
        data = index.to_dict()

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  💾 Индекс сохранён: {fpath}")
        return fpath

    def load(self, filepath: str) -> IndexData:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        for c in data["chunks"]:
            # Восстанавливаем эмбеддинг (если есть)
            embedding = c.get("embedding")
            chunks.append(Chunk(
                content=c["content"],
                source=c["source"],
                title=c["title"],
                section=c["section"],
                chunk_id=c["chunk_id"],
                chunk_type=c["chunk_type"],
                strategy=c.get("strategy", ""),
                embedding=embedding,
                metadata=c.get("metadata", {}),
            ))

        return IndexData(
            strategy_name=data["strategy_name"],
            chunks=chunks,
            created_at=data.get("created_at", ""),
            stats=data.get("stats", {}),
        )

    def list_indexes(self) -> List[str]:
        if not os.path.isdir(self.index_dir):
            return []
        return sorted([f for f in os.listdir(self.index_dir) if f.endswith(".json")])


# ============================================================================
# RAG INDEXER (ORCHESTRATOR)
# ============================================================================

class RAGIndexer:
    def __init__(self, auth_key: str = "", source_dir: str = ".",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Args:
            auth_key: Не используется в локальной версии, оставлен для совместимости
            source_dir: Директория с исходниками
            embedding_model: Модель для эмбеддингов (sentence-transformers)
        """
        self.source_dir = source_dir
        self.loader = DocumentLoader(source_dir)

        # Используем локальный сервис эмбеддингов (без Sberbank)
        self.embedding_service = LocalEmbeddingService(
            model_name=embedding_model,
            debug=False
        )

        self.index_manager = IndexManager()
        self.indexes: Dict[str, IndexData] = {}
        self._auto_load_indexes()

    def _auto_load_indexes(self):
        """Автоматическая загрузка индексов из файлов"""
        mapping = {"fixed_index.json": "fixed_size", "struct_index.json": "structure"}
        for fname, strategy in mapping.items():
            fpath = os.path.join(self.index_manager.index_dir, fname)
            if os.path.exists(fpath):
                try:
                    index = self.index_manager.load(fpath)
                    self.indexes[strategy] = index
                    print(f"  📂 Автозагрузка: {fname} → {strategy} ({len(index.chunks)} чанков)")
                except Exception as e:
                    print(f"  ⚠️  Ошибка загрузки {fname}: {e}")

    def build_index(self, strategy: str = "both", file_filter: str = "") -> Dict[str, IndexData]:
        """
        Создание индексов.

        Args:
            strategy: "fixed", "structure", "both"
            file_filter: Фильтр по имени файла (например, "main.py")
        """
        print(f"\n{'=' * 60}")
        print("📂 ЗАГРУЗКА ДОКУМЕНТОВ")
        print(f"{'=' * 60}")

        docs = self.loader.load_all()
        if file_filter:
            docs = [d for d in docs if file_filter in d.source]

        print(f"  Загружено документов: {len(docs)}")
        if file_filter:
            print(f"  Фильтр: {file_filter}")

        total_chars = sum(len(d.content) for d in docs)
        print(f"  Всего символов: {total_chars:,}")
        print(f"  Эквивалент страниц: ~{total_chars // 3000} стр.")

        result = {}

        if strategy in ("fixed", "both"):
            print(f"\n{'=' * 60}")
            print("✂️  СТРАТЕГИЯ: Фиксированный размер (500 символов, overlap 50)")
            print(f"{'=' * 60}")
            fixed_chunker = FixedSizeChunker(chunk_size=500, overlap=50)
            chunks = fixed_chunker.chunk(docs)
            print(f"  Чанков: {len(chunks)}")
            print(f"  Средний размер: {sum(len(c.content) for c in chunks) // max(len(chunks), 1)} символов")

            print("  Генерация эмбеддингов...")
            texts = [c.content for c in chunks]
            embeddings = self.embedding_service.get_embeddings(texts)
            for c, emb in zip(chunks, embeddings):
                c.embedding = emb

            index = IndexData(
                strategy_name="fixed_size",
                chunks=chunks,
                created_at=datetime.now().isoformat(),
                stats={
                    "total_chunks": len(chunks),
                    "total_chars": sum(len(c.content) for c in chunks),
                    "avg_chunk_size": sum(len(c.content) for c in chunks) // max(len(chunks), 1),
                    "min_chunk_size": min(len(c.content) for c in chunks) if chunks else 0,
                    "max_chunk_size": max(len(c.content) for c in chunks) if chunks else 0,
                    "total_documents": len(docs),
                },
            )
            fpath = self.index_manager.save(index, "fixed_index.json")
            self.indexes["fixed_size"] = index
            result["fixed_size"] = index

        if strategy in ("structure", "both"):
            print(f"\n{'=' * 60}")
            print("🏗️  СТРАТЕГИЯ: По структуре (классы, функции, секции)")
            print(f"{'=' * 60}")
            struct_chunker = StructureChunker()
            chunks = struct_chunker.chunk(docs)
            print(f"  Чанков: {len(chunks)}")
            print(f"  Средний размер: {sum(len(c.content) for c in chunks) // max(len(chunks), 1)} символов")

            print("  Генерация эмбеддингов...")
            texts = [c.content for c in chunks]
            embeddings = self.embedding_service.get_embeddings(texts)
            for c, emb in zip(chunks, embeddings):
                c.embedding = emb

            index = IndexData(
                strategy_name="structure",
                chunks=chunks,
                created_at=datetime.now().isoformat(),
                stats={
                    "total_chunks": len(chunks),
                    "total_chars": sum(len(c.content) for c in chunks),
                    "avg_chunk_size": sum(len(c.content) for c in chunks) // max(len(chunks), 1),
                    "min_chunk_size": min(len(c.content) for c in chunks) if chunks else 0,
                    "max_chunk_size": max(len(c.content) for c in chunks) if chunks else 0,
                    "total_documents": len(docs),
                    "type_distribution": self._type_distribution(chunks),
                },
            )
            fpath = self.index_manager.save(index, "struct_index.json")
            self.indexes["structure"] = index
            result["structure"] = index

        return result

    def _type_distribution(self, chunks: List[Chunk]) -> Dict[str, int]:
        dist = {}
        for c in chunks:
            t = c.chunk_type
            dist[t] = dist.get(t, 0) + 1
        return dist

    def search(self, query: str, strategy: str = "fixed_size", top_k: int = 5) -> List[Chunk]:
        """Поиск по индексу"""
        if strategy not in self.indexes:
            raise ValueError(f"Индекс '{strategy}' не загружен. Доступны: {list(self.indexes.keys())}")

        index = self.indexes[strategy]
        if not index.chunks:
            return []

        query_emb = self.embedding_service.get_embedding(query)

        scored = []
        for chunk in index.chunks:
            if chunk.embedding is None:
                continue
            score = self._cosine_similarity(query_emb, chunk.embedding)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Вычисление косинусного сходства"""
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def query_to_context(self, question: str, strategy: str = "fixed_size",
                         top_k: int = 5, max_chars: int = 4000) -> str:
        """Поиск и форматирование контекста для LLM"""
        chunks = self.search(question, strategy=strategy, top_k=top_k)
        if not chunks:
            return ""

        parts = []
        total = 0
        for c in chunks:
            header = f"[{c.source}] {c.section}"
            part = f"{header}\n{c.content}"
            total += len(part)
            if total > max_chars:
                break
            parts.append(part)

        return "\n\n---\n\n".join(parts)


# ============================================================================
# COMPARISON
# ============================================================================

def compare_strategies(fixed_index: IndexData, struct_index: IndexData) -> str:
    """Сравнение стратегий чанкинга"""
    lines = []
    lines.append("=" * 70)
    lines.append("📊 СРАВНЕНИЕ СТРАТЕГИЙ ЧАНКИНГА")
    lines.append("=" * 70)

    f_stats = fixed_index.stats
    s_stats = struct_index.stats

    lines.append("")
    lines.append(f"{'Параметр':<35} {'Фиксированный':<20} {'По структуре':<20}")
    lines.append("-" * 75)
    lines.append(f"{'Количество чанков':<35} {f_stats.get('total_chunks', 0):<20} {s_stats.get('total_chunks', 0):<20}")
    lines.append(f"{'Всего символов':<35} {f_stats.get('total_chars', 0):<20,} {s_stats.get('total_chars', 0):<20,}")
    lines.append(
        f"{'Средний размер чанка':<35} {f_stats.get('avg_chunk_size', 0):<20} {s_stats.get('avg_chunk_size', 0):<20}")
    lines.append(
        f"{'Мин. размер чанка':<35} {f_stats.get('min_chunk_size', 0):<20} {s_stats.get('min_chunk_size', 0):<20}")
    lines.append(
        f"{'Макс. размер чанка':<35} {f_stats.get('max_chunk_size', 0):<20} {s_stats.get('max_chunk_size', 0):<20}")

    lines.append("")
    lines.append("📌 РАСПРЕДЕЛЕНИЕ ТИПОВ (по структуре):")
    dist = s_stats.get("type_distribution", {})
    for t, count in sorted(dist.items()):
        lines.append(f"  {t}: {count}")

    lines.append("")
    lines.append("🔍 АНАЛИЗ:")
    f_count = f_stats.get("total_chunks", 0)
    s_count = s_stats.get("total_chunks", 0)

    if s_count < f_count and f_count > 0:
        ratio = f_count / max(s_count, 1)
        lines.append(f"  ✅ Структурный чанкинг даёт в {ratio:.1f}x меньше чанков —")
        lines.append(f"      каждый чанк семантически целостен (класс/функция целиком).")

    f_avg = f_stats.get("avg_chunk_size", 0)
    s_avg = s_stats.get("avg_chunk_size", 0)
    if s_avg > f_avg and s_avg > 0:
        lines.append(f"  ✅ Структурные чанки крупнее ({s_avg} vs {f_avg} символов) —")
        lines.append(f"      сохраняют контекст функции/класса целиком.")

    lines.append(f"  ✅ Фиксированный чанкинг предсказуем — все чанки одинакового размера.")
    lines.append(f"  ✅ Структурный чанкинг сохраняет логические границы кода.")

    lines.append("")
    lines.append("💡 РЕКОМЕНДАЦИЯ:")
    if s_count > 0 and s_avg > 100:
        lines.append("  Структурный чанкинг предпочтительнее для кода —")
        lines.append("  он сохраняет целостность классов и функций.")
    else:
        lines.append("  Фиксированный чанкинг проще и достаточен для большинства случаев.")

    lines.append("=" * 70)
    return "\n".join(lines)