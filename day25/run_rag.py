"""
Отдельный запускатор RAG-индексации для задания 21-го дня.

Запуск:  python3 run_rag.py

Результат:
  - rag_index/fixed_index.json    — чанки по фиксированному размеру
  - rag_index/struct_index.json   — чанки по структуре (классы/функции)
  - rag_index/index_summary.json  — сводка сравнения
  - Сравнение двух стратегий в консоли
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_indexer import *

AUTH_KEY = "Basic <ваш_base64_ключ>"


def progress(iterable, desc=""):
    items = list(iterable)
    for i, item in enumerate(items):
        yield item
        if (i + 1) % 100 == 0 or i == len(items) - 1:
            print(f"  {desc}: {i + 1}/{len(items)}")


def chunk_and_embed(chunker, docs, emb_svc, strategy_name, debug: bool = True):
    """Чанкинг и эмбеддинг с отладкой."""
    print(f"\n📌 {strategy_name.upper()} - НАЧАЛО ОБРАБОТКИ")

    # 1. Чанкинг
    chunks = chunker.chunk(docs)
    avg = sum(len(c.content) for c in chunks) // max(len(chunks), 1)
    mn = min(len(c.content) for c in chunks) if chunks else 0
    mx = max(len(c.content) for c in chunks) if chunks else 0
    print(f"  📄 Чанков: {len(chunks)} | средний {avg} | мин {mn} | макс {mx}")

    # Показываем распределение размеров
    if debug and chunks:
        sizes = [len(c.content) for c in chunks]
        ranges = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, float('inf'))]
        print("  📊 Распределение по размерам:")
        for low, high in ranges:
            count = sum(1 for s in sizes if low <= s < high)
            if high == float('inf'):
                label = f"> {low}"
            else:
                label = f"{low}-{high}"
            if count:
                print(f"     {label}: {count} чанков")

    # 2. Эмбеддинги
    print(f"  🧬 Генерация эмбеддингов для {len(chunks)} чанков...")

    # Включаем дебаг для EmbeddingService
    if hasattr(emb_svc, 'debug'):
        emb_svc.debug = debug

    failed = []
    for i, c in enumerate(chunks):
        try:
            if debug and i % 10 == 0:
                print(f"  📤 Чанк {i + 1}/{len(chunks)}: {len(c.content)} символов, источник: {c.source}")
            c.embedding = emb_svc.get_embedding(c.content)
        except Exception as e:
            print(f"  ❌ Ошибка на чанке {i + 1}: {e}")
            print(f"     Размер: {len(c.content)} символов")
            print(f"     Источник: {c.source}")
            print(f"     Секция: {c.section}")
            print(f"     Первые 100 символов: {c.content[:100].replace(chr(10), ' ')}...")
            failed.append((i, str(e)))

    if failed:
        print(f"  ⚠️ Не удалось обработать {len(failed)} чанков:")
        for idx, err in failed:
            print(f"     Чанк {idx + 1}: {err}")
    else:
        print(f"  ✅ Все {len(chunks)} чанков успешно обработаны")

    return chunks, avg, mn, mx

def main():
    print("=" * 70)
    print("  RAG-ИНДЕКСАТОР  |  День 21: Индексация документов")
    print("=" * 70)

    # 1. ЗАГРУЗКА ДОКУМЕНТОВ
    print("\n📂 ЗАГРУЗКА ДОКУМЕНТОВ")
    loader = DocumentLoader(".")
    docs = [d for d in loader.load_all() if d.source.endswith("main.py")]
    total_chars = sum(len(d.content) for d in docs)
    pages = total_chars // 3000
    print(f"  Файлов: {len(docs)}, символов: {total_chars:,} (~{pages} стр.)")
    for d in docs:
        print(f"    {d.source} ({len(d.content):,} симв.)")
    print(f"  {'✅' if pages >= 20 else '⚠️'} Требование 20-30 стр.: {'выполнено' if pages >= 20 else f'только {pages}'}")

    emb_svc = EmbeddingService(AUTH_KEY)

    # 2. СТРАТЕГИЯ 1: ФИКСИРОВАННЫЙ РАЗМЕР
    print(f"\n{'=' * 60}")
    print("  ✂️  СТРАТЕГИЯ 1: Фиксированный размер (500 символов, overlap 50)")
    print(f"{'=' * 60}")
    fc, fa, fmin, fmax = chunk_and_embed(FixedSizeChunker(500, 50), docs, emb_svc, "fixed")

    # 3. СТРАТЕГИЯ 2: ПО СТРУКТУРЕ
    print(f"\n{'=' * 60}")
    print("  🏗️  СТРАТЕГИЯ 2: По структуре (классы, функции, секции)")
    print(f"{'=' * 60}")
    sc, sa, smin, smax = chunk_and_embed(StructureChunker(), docs, emb_svc, "structure")

    type_dist = {}
    for c in sc:
        type_dist[c.chunk_type] = type_dist.get(c.chunk_type, 0) + 1
    print(f"  Типы: {type_dist}")

    # 4. СОХРАНЕНИЕ
    print(f"\n{'=' * 60}")
    print("  💾 СОХРАНЕНИЕ ИНДЕКСОВ")
    print(f"{'=' * 60}")
    im = IndexManager()

    fi = IndexData(strategy_name="fixed_size", chunks=fc, stats={
        "total_chunks": len(fc), "total_chars": sum(len(c.content) for c in fc),
        "avg_chunk_size": fa, "min_chunk_size": fmin, "max_chunk_size": fmax, "total_documents": len(docs)})
    fp = im.save(fi, filename="fixed_index.json")

    si = IndexData(strategy_name="structure", chunks=sc, stats={
        "total_chunks": len(sc), "total_chars": sum(len(c.content) for c in sc),
        "avg_chunk_size": sa, "min_chunk_size": smin, "max_chunk_size": smax,
        "total_documents": len(docs), "type_distribution": type_dist})
    sp = im.save(si, filename="struct_index.json")

    # 5. СРАВНЕНИЕ
    print()
    print(compare_strategies(fi, si))

    # 6. ПРИМЕРЫ
    print("\n📝 ПРИМЕРЫ ЧАНКОВ")
    print("=" * 60)
    for label, chunks in [("Fixed-size", fc), ("Structure (class)", [c for c in sc if c.chunk_type == "class"]), ("Structure (method)", [c for c in sc if c.chunk_type == "method"])]:
        if chunks:
            c = chunks[0]
            print(f"\n--- {label} ---")
            print(f"  id={c.chunk_id} src={c.source} section={c.section} size={len(c.content)}")
            print(f"  {c.content[:200]}...")

    # 7. СВОДКА
    summary = {
        "total_documents": len(docs), "total_chars": total_chars, "pages_equivalent": pages,
        "strategy_fixed": {"file": "fixed_index.json", "chunks": len(fc), "avg_size": fa},
        "strategy_structure": {"file": "struct_index.json", "chunks": len(sc), "avg_size": sa, "type_distribution": type_dist},
    }
    spath = os.path.join(im.index_dir, "index_summary.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("  ✅ ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    print(f"{'=' * 60}")
    print(f"  📂 {os.path.abspath(im.index_dir)}/")
    print(f"    {os.path.basename(fp)} ({os.path.getsize(fp):,} байт)")
    print(f"    {os.path.basename(sp)} ({os.path.getsize(sp):,} байт)")
    print(f"    index_summary.json")
    print(f"  📊 Всего чанков: {len(fc) + len(sc)}")
    print(f"  🧬 Эмбеддинги: GigaChat Embeddings API (384d, сегментация >500 токенов)")


if __name__ == "__main__":
    main()
