# -*- coding: utf-8 -*-
"""
Smart chunking for KOIB RAG.

v4.11 fixes made for OCR/DOCX/CSV artifacts:
- service prefixes such as ``passage:`` are stripped before storage/display;
- false ``formula`` fragments produced by OCR/PDF extraction are reclassified;
- text is flushed on source/page/heading changes, improving citations;
- figure/table/formula chunk ids include artifact-specific metadata to avoid collisions.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from langchain_core.documents import Document
except Exception:  # lightweight fallback for tests / minimal environments
    class Document:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

from .parsing import DocumentElement
from .utils import (
    clean_text,
    detect_doc_category,
    estimate_tokens,
    normalize_ocr_text,
    strip_embedding_prefix,
    text_hash,
)
from .text_processing import (
    generate_formula_summary as _generate_formula_summary,
    generate_table_summary as _generate_table_summary,
    is_noise_text,
    normalized_element_type,
    sanitize_chunk_content,
)
from config import (
    MIN_CHUNK_LENGTH,
    TABLE_MAX_CHARS,
    TABLE_MIN_COLS,
    TABLE_MIN_CONTENT_CHARS,
    TABLE_MIN_ROWS,
    TEXT_CHUNK_OVERLAP,
    TEXT_CHUNK_SIZE,
)

logger = logging.getLogger("koib.chunking")



@dataclass
class Chunk:
    """Document chunk ready for indexing."""

    chunk_id: str
    content: str
    full_content: Optional[str] = None
    chunk_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.content = sanitize_chunk_content(self.content)
        if self.full_content is not None:
            self.full_content = sanitize_chunk_content(self.full_content)
        self.chunk_type = normalized_element_type(self.chunk_type, self.full_content or self.content, self.metadata)

    def to_langchain_doc(self) -> Document:
        """Convert to LangChain Document for vector indexing. The content is never prefixed here."""
        return Document(
            page_content=self.content,
            metadata={
                "chunk_id": self.chunk_id,
                "chunk_type": self.chunk_type,
                **self.metadata,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "full_content": self.full_content,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata,
        }

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    @property
    def page(self) -> int:
        try:
            return int(self.metadata.get("page", 0))
        except Exception:
            return 0

    @property
    def heading(self) -> str:
        return str(self.metadata.get("heading", ""))

    @property
    def model(self) -> str:
        return str(self.metadata.get("model", "unknown"))

    @property
    def dedup_key(self) -> str:
        """Ключ дедупликации: хэш нормализованного содержимого + тип.

        Совпадает с ключом в build_ideal_index.load_clean_chunks, чтобы оба пути
        сборки (инкрементальный ингест и пересборка из артефактов) вели себя
        одинаково и не плодили дубли между почти идентичными руководствами
        КОИБ-2010/2017.
        """
        body = (self.full_content or self.content or "").lower()[:2000]
        return text_hash(body + "|" + self.chunk_type)

    @property
    def quality(self) -> int:
        """Богатство метаданных — больше = предпочтительнее при дедупликации.

        Совпадает с критерием в build_ideal_index.CleanChunk.quality: ценятся
        наличие full_content, известная модель, номер страницы и заголовок.
        """
        q = len(self.content)
        if self.full_content:
            q += 200
        if int(self.metadata.get("page", 0) or 0) > 0:
            q += 500
        if self.metadata.get("model") in {"koib2010", "koib2017a", "koib2017b"}:
            q += 300
        if self.metadata.get("heading"):
            q += 100
        return q


def deduplicate_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Убрать дубли по нормализованному содержимому, сохранив все источники.

    Раньше дедупликация работала только при пересборке индекса из артефактов
    (build_ideal_index.load_clean_chunks), а обычный ингест складывал дубли в
    индекс как есть — отсюда повторы почти идентичных руководств КОИБ-2010/2017
    (до 65% у рисунков, 19% у таблиц). Теперь единое правило для обоих путей.

    Из нескольких дублей оставляем чанк с наибольшим ``Chunk.quality``, но в его
    метаданные дописываем ``also_from`` — список остальных источников со
    страницами, чтобы цитирование не потеряло ни одной ссылки.
    """
    best: Dict[str, Chunk] = {}
    losers: Dict[str, List[Chunk]] = {}
    for c in chunks:
        key = c.dedup_key
        existing = best.get(key)
        if existing is None:
            best[key] = c
            continue
        if c.quality > existing.quality:
            losers.setdefault(key, []).append(existing)
            best[key] = c
        else:
            losers.setdefault(key, []).append(c)
    result: List[Chunk] = []
    for key, winner in best.items():
        same = losers.get(key, [])
        if same:
            others = []
            for dup in same:
                src = str(dup.metadata.get("source") or "")
                pg = dup.metadata.get("page", "")
                if src and (src, pg) not in {(o[0], o[1]) for o in others}:
                    others.append((src, pg))
            if others:
                md = dict(winner.metadata)
                # не дублируем первоисточник самого winner'а
                own = (str(md.get("source") or ""), md.get("page", ""))
                others = [o for o in others if o != own]
                if others:
                    md["also_from"] = [{"source": s, "page": p} for s, p in others]
                winner.metadata = md
        result.append(winner)
    return result


def _table_rows(markdown: str) -> tuple[List[str], Optional[str], Optional[str]]:
    """Разобрать markdown-таблицу на (строки данных, заголовок, разделитель)."""
    lines = [ln for ln in markdown.split("\n") if ln.strip()]
    if not lines:
        return [], None, None
    header = lines[0]
    separator = lines[1] if len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= set("-: ") else None
    data_start = 2 if separator else 1
    data = lines[data_start:]
    return data, header, separator


def is_degenerate_table(markdown: str) -> bool:
    """True для вырожденных таблиц: ≤1 строки данных или ≤1 столбца (Проблема 3).

    Это, как правило, ложные срабатывания детектора таблиц PyMuPDF на элементах
    вёрстки. Такие «таблицы» не несут пользы как отдельная модальность: они
    переводятся в text (если есть содержимое), а не складываются как table-чанк.
    """
    data_rows, header, _ = _table_rows(markdown)
    # число столбцов — по заголовку (надёжнее, чем по строке данных с пропусками)
    header_cells = [c for c in (header or "").strip().strip("|").split("|") if c.strip()]
    cols = len(header_cells)
    if cols < TABLE_MIN_COLS:
        return True
    if len(data_rows) < TABLE_MIN_ROWS:
        return True
    cells = _all_table_cells_local(markdown)
    if sum(len(c) for c in cells) < TABLE_MIN_CONTENT_CHARS:
        return True
    return False


def _all_table_cells_local(markdown: str) -> List[str]:
    cells = [c.strip() for c in sanitize_chunk_content(markdown).replace("\n", "|").split("|")]
    return [c for c in cells if c and c != "---"]


def split_large_table(markdown: str, max_chars: int = TABLE_MAX_CHARS) -> List[str]:
    """Нарезать крупную markdown-таблицу по строкам, сохраняя заголовок.

    Каждый кусок начинается со строки-заголовка и разделителя, чтобы оставаться
    осмысленным table-чанком. Если таблица помещается в лимит — возвращается как
    есть (один элемент).
    """
    if len(markdown) <= max_chars:
        return [markdown]
    data, header, separator = _table_rows(markdown)
    if not header or not data:
        return [markdown]
    sep_line = separator or ("| " + " | ".join(["---"] * len([c for c in header.strip().strip("|").split("|")])) + " |")
    head_block = f"{header}\n{sep_line}\n"
    head_len = len(head_block)

    pieces: List[str] = []
    current = head_block
    for row in data:
        if len(current) + len(row) + 1 > max_chars and current != head_block:
            pieces.append(current.rstrip("\n"))
            current = head_block
        current += row + "\n"
    if current.strip():
        pieces.append(current.rstrip("\n"))
    return pieces or [markdown]


# Границы для дробления сверхдлинного одиночного куска: предложения и переносы.
_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def _char_windows(s: str, max_tokens: int) -> List[str]:
    """Последняя линия защиты: режем по символам, если иных границ нет.

    Шаг согласован с estimate_tokens (1 токен ≈ 2.5 символа, коэффициент 0.4),
    чтобы каждое окно укладывалось в max_tokens.
    """
    step = max(1, int(max_tokens / 0.4))
    return [s[i:i + step] for i in range(0, len(s), step)]


def _atoms(unit: str, max_tokens: int) -> List[str]:
    """Разложить переразмерный кусок на атомы, каждый <= max_tokens.

    Стратегия по убыванию структурности: предложения/строки -> слова ->
    жёсткое окно по символам. Куски, помещающиеся в лимит, не трогаются.
    Это закрывает утечку, при которой одиночный «абзац» (например, .docx-таблица,
    пришедшая по текстовому пути) больше лимита уходил в индекс одним чанком.
    """
    if estimate_tokens(unit) <= max_tokens:
        return [unit]
    parts = [u for u in _SENT_SPLIT.split(unit) if u and u.strip()]
    if len(parts) == 1:
        parts = unit.split(" ")
    atoms: List[str] = []
    for p in parts:
        atoms.extend(_char_windows(p, max_tokens) if estimate_tokens(p) > max_tokens else [p])
    return atoms


def _pack(atoms: List[str], sep: str, max_tokens: int) -> List[str]:
    """Жадно упаковать атомы, измеряя РЕАЛЬНУЮ длину кандидата (с разделителями).

    Прежняя версия складывала ``current_tokens`` как сумму токенов кусков и не
    учитывала длину разделителей при склейке, из-за чего итоговый чанк мог
    превышать лимит. Здесь длина проверяется по фактической строке-кандидату.
    """
    chunks: List[str] = []
    cur = ""
    for a in atoms:
        cand = (cur + sep + a) if cur else a
        if cur and estimate_tokens(cand) > max_tokens:
            chunks.append(cur)
            cur = a
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks


def _split_text_semantic(
    text: str,
    max_tokens: int = TEXT_CHUNK_SIZE,
    overlap_tokens: int = TEXT_CHUNK_OVERLAP,
) -> List[str]:
    """Абзацно-ориентированная нарезка с ГАРАНТИЕЙ верхнего предела по токенам.

    В отличие от прежней версии:
    * одиночный абзац/строка больше лимита дробится (`_atoms`) — больше нет
      переразмеренных чанков из .docx-таблиц, пришедших по текстовому пути;
    * упаковка считает фактическую длину со разделителями (`_pack`);
    * overlap добавляется как «хвост» предыдущего чанка в пределах бюджета.
    """
    text = sanitize_chunk_content(text)
    if not text or len(text.strip()) < MIN_CHUNK_LENGTH:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1 and "\n" in text:
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    atoms: List[str] = []
    for para in paragraphs:
        if is_noise_text(para):
            continue
        atoms.extend(_atoms(para, max_tokens))

    if not atoms:
        return []

    # overlap уже учтён внутри бюджета max_tokens: пакуем с запасом, затем
    # пришиваем «хвост» предыдущего чанка к началу следующего.
    if overlap_tokens > 0:
        budget = max(MIN_CHUNK_LENGTH // 4, max_tokens - overlap_tokens)
        base_chunks = _pack(atoms, "\n\n", budget)
        chunks: List[str] = []
        prev_tail = ""
        for ch in base_chunks:
            stitched = (prev_tail + "\n\n" + ch).strip() if prev_tail else ch
            chunks.append(stitched)
            words = ch.split(" ")
            tail: List[str] = []
            ttok = 0
            for w in reversed(words):
                wt = estimate_tokens(w)
                if ttok + wt > overlap_tokens:
                    break
                tail.insert(0, w)
                ttok += wt
            prev_tail = " ".join(tail)
    else:
        chunks = _pack(atoms, "\n\n", max_tokens)

    return [c for c in chunks if estimate_tokens(c) >= MIN_CHUNK_LENGTH // 4]


class SmartChunker:
    """Chunk text and structured document elements."""

    def __init__(
        self,
        chunk_size: int = TEXT_CHUNK_SIZE,
        chunk_overlap: int = TEXT_CHUNK_OVERLAP,
        min_chunk_length: int = MIN_CHUNK_LENGTH,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def chunk_elements(self, elements: List[DocumentElement]) -> List[Chunk]:
        chunks: List[Chunk] = []
        text_buffer: List[DocumentElement] = []
        current_heading = ""
        last_scope: tuple[str, int, str] | None = None

        def flush() -> None:
            nonlocal text_buffer
            if text_buffer:
                chunks.extend(self._chunk_text_buffer(text_buffer, current_heading))
                text_buffer = []

        for element in elements:
            element.content = sanitize_chunk_content(element.content)
            if is_noise_text(element.content) and element.element_type != "table":
                continue

            if element.element_type == "heading":
                flush()
                current_heading = element.content or element.heading or current_heading
                last_scope = None
                continue

            effective_type = normalized_element_type(element.element_type, element.content, element.metadata)
            element.element_type = effective_type
            if element.heading:
                current_heading = element.heading

            scope = (element.source, int(element.page or 0), current_heading or element.heading or "")
            if effective_type in {"table", "formula", "figure"}:
                flush()
                # Вырожденные таблицы (Проблема 3): ложные срабатывания детектора
                # таблиц на элементах вёрстки. Такие пускаем обычным текстом, а не
                # отдельной table-модальностью — так они хотя бы находятся поиском.
                if effective_type == "table" and is_degenerate_table(element.content):
                    effective_type = "text"
                    element.element_type = "text"
                    text_buffer.append(element)
                    last_scope = scope
                    continue
                structured_list = self._chunk_structured_element(element, current_heading)
                for structured in structured_list:
                    if structured.content and not is_noise_text(structured.content, min_alpha=0):
                        chunks.append(structured)
                last_scope = None
            else:
                # .docx нередко отдаёт таблицу как обычный текстовый элемент с
                # markdown-разметкой (| ... | ... |). Перехватываем такие случаи и
                # отправляем по табличному пути, чтобы сработали split_large_table
                # и table-саммари, а не текстовый сплиттер (иначе крупная таблица
                # уходила в индекс одним переразмеренным text-чанком — Проблема 1).
                stripped = (element.content or "").lstrip()
                looks_like_table = stripped.startswith("|") and element.content.count("\n|") >= 2
                if looks_like_table and not is_degenerate_table(element.content):
                    flush()
                    element.element_type = "table"
                    for structured in self._chunk_structured_element(element, current_heading):
                        if structured.content and not is_noise_text(structured.content, min_alpha=0):
                            chunks.append(structured)
                    last_scope = None
                    continue
                if last_scope is not None and scope != last_scope:
                    flush()
                text_buffer.append(element)
                last_scope = scope

        flush()
        # Сквозной порядковый номер в пределах документа — опора для Context
        # Expansion (Проблема 5): соседние чанки = тот же source, seq ± окно.
        seq_by_source: Dict[str, int] = {}
        for c in chunks:
            src = str(c.metadata.get("source", ""))
            seq_by_source[src] = seq_by_source.get(src, -1) + 1
            c.metadata["seq"] = seq_by_source[src]
        logger.info("Создано %s чанков из %s элементов", len(chunks), len(elements))
        return chunks

    def _chunk_text_buffer(self, elements: List[DocumentElement], heading: str) -> List[Chunk]:
        combined = "\n\n".join(sanitize_chunk_content(e.content) for e in elements if e.content.strip())
        if not combined or len(combined.strip()) < self.min_chunk_length:
            return []

        text_chunks = _split_text_semantic(combined, max_tokens=self.chunk_size, overlap_tokens=self.chunk_overlap)
        chunks: List[Chunk] = []
        source = elements[0].source if elements else ""
        pages = [int(e.page or 0) for e in elements if int(e.page or 0) > 0]
        page = min(pages) if pages else (elements[0].page if elements else 0)
        page_end = max(pages) if pages else page
        model = elements[0].model if elements else "unknown"
        heading_value = heading or elements[0].heading if elements else heading

        for i, text in enumerate(text_chunks):
            text = sanitize_chunk_content(text)
            if len(text) < self.min_chunk_length or is_noise_text(text):
                continue
            chunk_id = f"txt_{text_hash(f'{source}:{page}:{page_end}:{i}:{text[:800]}')}"
            metadata = {
                "source": source,
                "page": page,
                "heading": heading_value,
                "model": model,
                "doc_category": detect_doc_category(source),
                "chunk_index": i,
            }
            if page_end and page_end != page:
                metadata["page_end"] = page_end
            chunks.append(Chunk(chunk_id=chunk_id, content=text, chunk_type="text", metadata=metadata))
        return chunks

    def _chunk_structured_element(self, element: DocumentElement, heading: str) -> List[Chunk]:
        """Превратить структурный элемент в один или несколько чанков.

        Таблицы сверх TABLE_MAX_CHARS режутся по строкам (Проблема 3): каждый
        кусок получает свой table-чанк с собственным саммари, фокусным на свою
        группу строк, и общий ``table_id`` для последующей сборки. Раньше таблица
        на 26 КБ уходила в индекс одним чанком и ломала эмбеддинг + контекст.
        """
        element.content = sanitize_chunk_content(element.content)
        effective_type = normalized_element_type(element.element_type, element.content, element.metadata)
        base_meta = {
            "source": element.source,
            "page": element.page,
            "heading": heading or element.heading,
            "model": element.model,
            "doc_category": detect_doc_category(element.source),
            "element_id": element.element_id,
            **element.metadata,
        }

        if effective_type == "table":
            # Нарезаем крупную таблицу на части. Для каждой строим свой саммари
            # по своей группе строк, чтобы конкретные значения попадали в поиск.
            pieces = split_large_table(element.content)
            table_id = element.element_id or text_hash(element.content[:300])
            chunks: List[Chunk] = []
            for idx, piece in enumerate(pieces):
                piece_meta = dict(base_meta)
                piece_meta["table_id"] = table_id
                piece_meta["table_part"] = idx
                piece_meta["table_parts"] = len(pieces)
                if "num_rows" in piece_meta and idx > 0:
                    # у части строк меньше, чем у целой таблицы — пересчитаем грубо
                    data_rows, _, _ = _table_rows(piece)
                    piece_meta["num_rows"] = len(data_rows)
                summary = _generate_table_summary(piece, piece_meta)
                id_material = "|".join(
                    [str(element.source), str(element.page), "table", table_id, str(idx), piece[:800]]
                )
                chunks.append(
                    Chunk(
                        chunk_id=f"table_{text_hash(id_material)}",
                        content=summary,
                        full_content=piece,
                        chunk_type="table",
                        metadata=piece_meta,
                    )
                )
            return chunks

        # formula / figure / прочее — один чанк, как и раньше
        if effective_type == "formula":
            summary = _generate_formula_summary(element.content, element.metadata)
            full_content: Optional[str] = element.content
        elif effective_type == "figure":
            summary = element.content
            full_content = element.content
        else:
            summary = element.content
            full_content = None

        id_material = "|".join(
            [
                str(element.source),
                str(element.page),
                str(effective_type),
                str(element.element_id),
                str(element.metadata.get("image_path", "")),
                str(element.metadata.get("table_index", "")),
                str(element.metadata.get("figure_index", "")),
                element.content[:800],
            ]
        )
        chunk_id = f"{effective_type}_{text_hash(id_material)}"
        return [
            Chunk(
                chunk_id=chunk_id,
                content=summary,
                full_content=full_content,
                chunk_type=effective_type,
                metadata=base_meta,
            )
        ]
