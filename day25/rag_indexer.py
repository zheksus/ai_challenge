import os
import re
import json
import ast
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

import requests


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
# DOCUMENT LOADER
# ============================================================================

class DocumentLoader:
    EXTENSIONS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini"}

    def __init__(self, source_dir: str):
        self.source_dir = source_dir

    def load_all(self) -> List[Document]:
        docs = []
        for root, dirs, files in os.walk(self.source_dir):
            dirs[:] = [d for d in dirs if not d.startswith((".", "_")) and d not in ("venv", ".venv", "__pycache__", ".idea")]
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
    MAX_CHUNK_CHARS = 1800  # ~450 токенов (безопасно для GigaChat Embeddings)

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
        top_nodes = [n for n in ast.iter_child_nodes(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]

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
        method_ranges = [(getattr(m, "lineno", 1) - 1, getattr(m, "end_lineno", getattr(m, "lineno", 1))) for m in methods]
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

    def _chunk_function(self, node: ast.FunctionDef, doc: Document, lines: List[str], parent_class: str = None) -> Chunk:
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
# EMBEDDING SERVICE (GigaChat API)
# ============================================================================

class EmbeddingService:
    EMBEDDING_MODEL = "Embeddings"
    MAX_TOKENS = 500
    TOKEN_CHARS = 4

    def __init__(self, auth_key: str, debug: bool = False):
        self.auth_key = auth_key
        self.debug = debug
        self._token = None
        self._token_expires_at = None
        self._cache: Dict[str, List[float]] = {}
        self._call_count = 0
        self._total_chars = 0

    def _get_token(self) -> str:
        if self._token and self._token_expires_at and datetime.now() < self._token_expires_at:
            return self._token
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": self.auth_key,
        }
        resp = requests.post(url, headers=headers, data={"scope": "GIGACHAT_API_PERS"}, verify=False, timeout=30)
        data = resp.json()
        self._token = data.get("access_token")
        from datetime import timedelta
        self._token_expires_at = datetime.now() + timedelta(minutes=25)
        return self._token

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text) // 4 + 1

    def _split_text(self, text: str) -> List[str]:
        """Разбивает текст на сегменты по 1500 символов."""
        MAX_CHARS = 1500  # Уменьшаем с 2000 до 1500 для безопасности
        if len(text) <= MAX_CHARS:
            return [text]

        if self.debug:
            print(f"  🔄 [SPLIT] Текст {len(text)} символов разбивается на сегменты")

        segments = []
        # Разбиваем по строкам, чтобы не разрывать посередине
        lines = text.split("\n")
        current = []
        current_len = 0

        for line in lines:
            if current_len + len(line) + 1 > MAX_CHARS and current:
                segments.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += len(line) + 1

        if current:
            segments.append("\n".join(current))

        return segments

    def _average_vectors(self, vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []
        if len(vectors) == 1:
            return vectors[0]
        dim = len(vectors[0])
        avg = [0.0] * dim
        for vec in vectors:
            for i in range(dim):
                avg[i] += vec[i]
        norm = sum(v * v for v in avg) ** 0.5
        if norm > 0:
            avg = [v / norm for v in avg]
        return avg

    def get_embedding(self, text: str) -> List[float]:
        self._call_count += 1
        self._total_chars += len(text)

        # Отладочная печать
        if self.debug:
            print(f"\n  📤 [EMBED #{self._call_count}]")
            print(f"     Размер текста: {len(text)} символов (~{self._estimate_tokens(text)} токенов)")
            print(f"     Первые 100 символов: {text[:100].replace(chr(10), ' ')}...")

        cache_key = text[:200]
        if cache_key in self._cache:
            if self.debug:
                print(f"     ✅ Использован кэш")
            return self._cache[cache_key]

        segments = self._split_text(text)
        if self.debug and len(segments) > 1:
            print(f"     🔄 Разбито на {len(segments)} сегментов")

        vectors = []
        for i, seg in enumerate(segments):
            if self.debug and len(segments) > 1:
                print(f"     📤 Сегмент {i + 1}/{len(segments)}: {len(seg)} символов")
            vec = self._call_embedding_api(seg)
            vectors.append(vec)

        emb = self._average_vectors(vectors) if len(vectors) > 1 else vectors[0]
        self._cache[cache_key] = emb

        if self.debug:
            print(f"     ✅ Эмбеддинг получен (размерность: {len(emb)})")

        return emb

    def _call_embedding_api(self, text: str, max_retries: int = 3) -> List[float]:
        """Вызов API с повторными попытками и автоматическим обрезанием."""
        token = self._get_token()

        for model in [self.EMBEDDING_MODEL, "Embeddings-2"]:
            for attempt in range(max_retries):
                try:
                    # 👇 ОБРЕЗАЕМ ТЕКСТ ДО БЕЗОПАСНОГО РАЗМЕРА
                    # Для кода безопасный лимит ~1500 символов (~375 токенов)
                    MAX_SAFE_CHARS = 1500
                    text_to_send = text
                    if len(text_to_send) > MAX_SAFE_CHARS:
                        if self.debug:
                            print(f"        ✂️ Обрезаем текст: {len(text_to_send)} → {MAX_SAFE_CHARS} символов")
                        text_to_send = text_to_send[:MAX_SAFE_CHARS]

                    if self.debug and attempt > 0:
                        print(f"        🔄 Попытка {attempt + 1}/{max_retries}")

                    resp = requests.post(
                        "https://gigachat.devices.sberbank.ru/api/v1/embeddings",
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                            "Authorization": f"Bearer {token}",
                        },
                        json={"model": model, "input": text_to_send},
                        verify=False,
                        timeout=60,
                    )

                    if self.debug:
                        print(f"        Ответ: HTTP {resp.status_code}")

                    if resp.status_code == 200:
                        result = resp.json()
                        if "data" in result and len(result["data"]) > 0:
                            return result["data"][0]["embedding"]

                    if resp.status_code == 413:
                        # Текст всё ещё слишком большой — обрезаем ещё сильнее
                        if self.debug:
                            print(f"        ⚠️ Текст всё ещё слишком большой, обрезаем до 1000")
                        text = text[:1000]
                        continue

                    if resp.status_code == 402:
                        wait_time = 2 ** attempt
                        if self.debug:
                            print(f"        ⚠️ Код 402, ждём {wait_time}с...")
                        time.sleep(wait_time)
                        continue

                    if resp.status_code == 429:
                        wait_time = 5 * (attempt + 1)
                        if self.debug:
                            print(f"        ⚠️ Rate limit, ждём {wait_time}с...")
                        time.sleep(wait_time)
                        continue

                    if resp.status_code != 200:
                        raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")

                except requests.exceptions.Timeout:
                    if self.debug:
                        print(f"        ⏰ Таймаут, попытка {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise

        raise RuntimeError("Embeddings API недоступен после всех попыток")


    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.get_embeddings_batch(texts)

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        if self.debug:
            print(f"\n📦 [BATCH] Всего текстов: {len(texts)}")
            total_chars = sum(len(t) for t in texts)
            print(f"📦 [BATCH] Всего символов: {total_chars:,}")

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if self.debug:
                print(f"\n  📦 Батч {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
                print(f"     Текстов в батче: {len(batch)}")

            batch_embeddings = []
            for j, text in enumerate(batch):
                if self.debug:
                    print(f"     Текст {j + 1}/{len(batch)}: {len(text)} символов")
                emb = self.get_embedding(text)
                batch_embeddings.append(emb)
                time.sleep(0.05)
            results.extend(batch_embeddings)
            print(f"  📊 Прогресс: {min(i + batch_size, len(texts))}/{len(texts)}")

        return results

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
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(index.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"  💾 Индекс сохранён: {fpath}")
        return fpath

    def load(self, filepath: str) -> IndexData:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [Chunk(**c) for c in data["chunks"]]
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
# RELEVANCE FILTER
# ============================================================================

@dataclass
class FilterResult:
    chunks: List[Chunk]
    scores: List[float]
    top_k_before: int
    top_k_after: int
    min_score: float
    stats: Dict[str, Any] = field(default_factory=dict)


class RelevanceFilter:
    def __init__(self, min_score: float = 0.25, top_k_before: int = 30, top_k_after: int = 10):
        self.min_score = min_score
        self.top_k_before = top_k_before
        self.top_k_after = top_k_after

    def filter_and_rerank(self, query_emb: List[float], chunks: List[Chunk],
                          similarities: Dict[str, float] = None) -> FilterResult:
        scored = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            if similarities and chunk.chunk_id in similarities:
                score = similarities[chunk.chunk_id]
            else:
                score = self._cosine_similarity(query_emb, chunk.embedding)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = scored[:self.top_k_before]
        filtered = [(s, c) for s, c in candidates if s >= self.min_score]
        filtered = filtered[:self.top_k_after]

        result = FilterResult(
            chunks=[c for _, c in filtered],
            scores=[s for s, _ in filtered],
            top_k_before=len(candidates),
            top_k_after=len(filtered),
            min_score=self.min_score,
            stats={
                "total_scored": len(scored),
                "candidates_k": len(candidates),
                "after_filter": len(filtered),
                "avg_score_before": sum(s for s, _ in candidates) / len(candidates) if candidates else 0,
                "avg_score_after": sum(s for s, _ in filtered) / len(filtered) if filtered else 0,
            },
        )
        return result

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ============================================================================
# QUERY REWRITER (через GigaChat Chat API)
# ============================================================================

class QueryRewriter:
    REWRITE_PROMPT = """
Ты — система улучшения поисковых запросов по кодовой базе проекта.
Перепиши вопрос пользователя так, чтобы он лучше подходил для поиска по коду.
Используй термины из программирования (класс, функция, метод, импорт, и т.д.).
Если вопрос про конкретный код — добавь имена классов/функций в предположении.
Ответь ТОЛЬКО переписанным запросом, без пояснений.

Исходный вопрос: {question}

Улучшенный запрос:"""

    def __init__(self, auth_key: str):
        self.auth_key = auth_key
        self._token = None
        self._token_expires_at = None

    def _get_token(self) -> str:
        if self._token and self._token_expires_at and datetime.now() < self._token_expires_at:
            return self._token
        import uuid
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": self.auth_key,
        }
        resp = requests.post(url, headers=headers, data={"scope": "GIGACHAT_API_PERS"}, verify=False, timeout=30)
        data = resp.json()
        self._token = data.get("access_token")
        from datetime import timedelta
        self._token_expires_at = datetime.now() + timedelta(minutes=25)
        return self._token

    def rewrite(self, question: str) -> str:
        token = self._get_token()
        payload = {
            "model": "GigaChat",
            "messages": [
                {"role": "system", "content": "Ты — система улучшения запросов для поиска по коду. Отвечай только переписанным запросом."},
                {"role": "user", "content": self.REWRITE_PROMPT.format(question=question)},
            ],
            "temperature": 0.3,
            "max_tokens": 200,
        }
        resp = requests.post(
            "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json=payload,
            verify=False,
            timeout=30,
        )
        result = resp.json()
        if "choices" in result and result["choices"]:
            rewritten = result["choices"][0]["message"]["content"].strip()
            return rewritten
        return question


# ============================================================================
# RAG INDEXER (ORCHESTRATOR)
# ============================================================================

class RAGIndexer:
    def __init__(self, auth_key: str, source_dir: str = "."):
        self.auth_key = auth_key
        self.source_dir = source_dir
        self.loader = DocumentLoader(source_dir)
        self.embedding_service = EmbeddingService(auth_key)
        self.index_manager = IndexManager()
        self.indexes: Dict[str, IndexData] = {}
        self.rewriter = QueryRewriter(auth_key)
        self.filter = RelevanceFilter()
        self._auto_load_indexes()

    def _auto_load_indexes(self):
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

    def build_index(self, strategy: str = "both", file_filter: str = "weather_mcp_server.py") -> Dict[str, IndexData]:
        print(f"\n{'='*60}")
        print("📂 ЗАГРУЗКА ДОКУМЕНТОВ")
        print(f"{'='*60}")
        docs = [d for d in self.loader.load_all() if file_filter in d.source]
        print(f"  Загружено документов: {len(docs)} (фильтр: {file_filter})")
        total_chars = sum(len(d.content) for d in docs)
        print(f"  Всего символов: {total_chars:,}")
        print(f"  Эквивалент страниц: ~{total_chars // 3000} стр.")

        result = {}

        if strategy in ("fixed", "both"):
            print(f"\n{'='*60}")
            print("✂️  СТРАТЕГИЯ: Фиксированный размер (500 символов, overlap 50)")
            print(f"{'='*60}")
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
            fpath = self.index_manager.save(index, filename="fixed_index.json")
            self.indexes["fixed_size"] = index
            result["fixed_size"] = index

        if strategy in ("structure", "both"):
            print(f"\n{'='*60}")
            print("🏗️  СТРАТЕГИЯ: По структуре (классы, функции, секции)")
            print(f"{'='*60}")
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
            fpath = self.index_manager.save(index, filename="struct_index.json")
            self.indexes["structure"] = index
            result["structure"] = index

        return result

    def _type_distribution(self, chunks: List[Chunk]) -> Dict[str, int]:
        dist = {}
        for c in chunks:
            t = c.chunk_type
            dist[t] = dist.get(t, 0) + 1
        return dist

    def search(self, query: str, strategy: str = "structure", top_k: int = 10) -> List[Chunk]:
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

    def search_filtered(self, query: str, strategy: str = "structure",
                         min_score: float = None, top_k_before: int = None,
                         top_k_after: int = 10) -> FilterResult:
        if strategy not in self.indexes:
            raise ValueError(f"Индекс '{strategy}' не загружен")
        index = self.indexes[strategy]
        if not index.chunks:
            return FilterResult([], [], 0, 0, 0)
        query_emb = self.embedding_service.get_embedding(query)
        f = self.filter
        if min_score is not None:
            f.min_score = min_score
        if top_k_before is not None:
            f.top_k_before = top_k_before
        if top_k_after is not None:
            f.top_k_after = top_k_after
        return f.filter_and_rerank(query_emb, index.chunks)

    def search_with_rewrite(self, query: str, strategy: str = "structure",
                             top_k: int = 5) -> Tuple[str, List[Chunk]]:
        rewritten = self.rewriter.rewrite(query)
        chunks = self.search(rewritten, strategy=strategy, top_k=top_k)
        return rewritten, chunks

    def query_to_context(self, question: str, strategy: str = "structure",
                         top_k: int = 10, max_chars: int = 6000,
                         use_filter: bool = True, use_rewrite: bool = False,
                         min_score: float = None) -> Tuple[str, Dict]:
        info = {}
        scores = None
        if use_rewrite:
            rewritten, chunks = self.search_with_rewrite(question, strategy=strategy, top_k=top_k)
            info["rewritten_query"] = rewritten
        else:
            rewritten = question
            if use_filter:
                result = self.search_filtered(rewritten, strategy=strategy,
                                               min_score=min_score, top_k_after=top_k)
                chunks = result.chunks
                scores = result.scores
                info["filter_stats"] = result.stats
            else:
                chunks = self.search(rewritten, strategy=strategy, top_k=top_k)

        if not chunks:
            return "", info

        # Compute scores if not already computed (non-filtered path)
        if scores is None:
            query_emb = self.embedding_service.get_embedding(rewritten)
            scores = []
            for c in chunks:
                if c.embedding is not None:
                    scores.append(self._cosine_similarity(query_emb, c.embedding))
                else:
                    scores.append(0.0)

        parts = []
        chunks_info = []
        total = 0
        for i, c in enumerate(chunks):
            header = f"[{c.source}] {c.section}"
            part = f"{header}\n{c.content}"
            total += len(part)
            if total > max_chars:
                break
            parts.append(part)
            chunks_info.append({
                "source": c.source,
                "section": c.section,
                "chunk_id": c.chunk_id,
                "content": c.content,
                "score": round(float(scores[i]), 4),
            })

        if chunks_info:
            info["chunks_info"] = chunks_info
            info["max_score"] = max(c["score"] for c in chunks_info)
            info["avg_score"] = sum(c["score"] for c in chunks_info) / len(chunks_info)

        return "\n\n---\n\n".join(parts), info

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

# ============================================================================
# COMPARISON
# ============================================================================

def compare_strategies(fixed_index: IndexData, struct_index: IndexData) -> str:
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
    lines.append(f"{'Средний размер чанка':<35} {f_stats.get('avg_chunk_size', 0):<20} {s_stats.get('avg_chunk_size', 0):<20}")
    lines.append(f"{'Мин. размер чанка':<35} {f_stats.get('min_chunk_size', 0):<20} {s_stats.get('min_chunk_size', 0):<20}")
    lines.append(f"{'Макс. размер чанка':<35} {f_stats.get('max_chunk_size', 0):<20} {s_stats.get('max_chunk_size', 0):<20}")

    lines.append("")
    lines.append("📌 РАСПРЕДЕЛЕНИЕ ТИПОВ (по структуре):")
    dist = s_stats.get("type_distribution", {})
    for t, count in sorted(dist.items()):
        lines.append(f"  {t}: {count}")

    lines.append("")
    lines.append("🔍 АНАЛИЗ:")
    f_count = f_stats.get("total_chunks", 0)
    s_count = s_stats.get("total_chunks", 0)

    if s_count < f_count:
        ratio = f_count / max(s_count, 1)
        lines.append(f"  ✅ Структурный чанкинг даёт в {ratio:.1f}x меньше чанков —")
        lines.append(f"      каждый чанк семантически целостен (класс/функция целиком).")
    else:
        lines.append(f"  ℹ️ Фиксированный чанкинг даёт {f_count} чанков, структурный — {s_count}.")

    f_avg = f_stats.get("avg_chunk_size", 0)
    s_avg = s_stats.get("avg_chunk_size", 0)
    if s_avg > f_avg:
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
