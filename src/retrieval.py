# -*- coding: utf-8 -*-
import json, logging, sqlite3, hashlib, threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .indexing import IndexBuilder, get_global_embeddings
from .utils import estimate_tokens, normalize_ocr_text, strip_embedding_prefix, KNOWN_MODELS
from .text_processing import sanitize_chunk_content as _strip_passage_prefix
from config import QUERY_PREFIX, PASSAGE_PREFIX, VECTOR_SEARCH_K, BM25_SEARCH_K, FINAL_TOP_K, HYBRID_ALPHA, USE_RERANKER, RERANKER_MODEL, USE_ONNX_RERANKER, USE_HYDE, EMBEDDING_PROVIDER, METADATA_DIR, SEMANTIC_CACHE_ENABLED, SEMANTIC_CACHE_THRESHOLD, SEMANTIC_CACHE_MAX_CANDIDATES, USE_USHAPED_CONTEXT, MAX_TABLE_ROWS_IN_PROMPT, MAX_TABLE_TOKENS_IN_PROMPT, MODEL_FILTER_STRICT, MODEL_STRICT_MIN_EXACT, CONTEXT_EXPANSION_ENABLED, CONTEXT_EXPANSION_WINDOW, CONTEXT_EXPANSION_MAX_CHARS

logger = logging.getLogger("koib.retrieval")


def reorder_u_shape(chunks: List) -> List:
    if len(chunks) <= 2: return chunks
    reordered, left, right = [], 0, len(chunks) - 1
    while left <= right:
        reordered.append(chunks[left])
        if left != right: reordered.append(chunks[right])
        left += 1; right -= 1
    return reordered

def truncate_table_for_prompt(markdown: str, max_rows: int = MAX_TABLE_ROWS_IN_PROMPT, max_tokens: int = MAX_TABLE_TOKENS_IN_PROMPT) -> str:
    if not markdown: return markdown
    lines = markdown.split('\n')
    if len(lines) <= max_rows + 2 and estimate_tokens(markdown) <= max_tokens: return markdown
    header, separator = lines[0] if lines else "", lines[1] if len(lines) > 1 else ""
    data_lines = [l for l in lines[2:] if l.strip()][:max_rows]
    result_lines, current_tokens = [header, separator], estimate_tokens(header + separator)
    for line in data_lines:
        lt = estimate_tokens(line)
        if current_tokens + lt > max_tokens: break
        result_lines.append(line); current_tokens += lt
    truncated = '\n'.join(result_lines)
    if len(result_lines) - 2 < max(0, len(lines) - 2):
        truncated += f"\n\n[...таблица обрезана: показано {len(result_lines) - 2} из {len(lines) - 2} строк...]"
    return truncated

class SemanticCache:
    def __init__(self, path: Optional[Path] = None, threshold: float = SEMANTIC_CACHE_THRESHOLD):
        self.path = path or METADATA_DIR / "semantic_cache.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.conn: self.conn.execute('''CREATE TABLE IF NOT EXISTS cache (query_hash TEXT PRIMARY KEY, query_text TEXT, embedding BLOB, answer TEXT, sources TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, hit_count INTEGER DEFAULT 0)''')
    def get(self, query: str, query_embedding: Optional[List[float]]) -> Optional[Dict]:
        if not SEMANTIC_CACHE_ENABLED or query_embedding is None: return None
        try:
            cur = self.conn.cursor()
            cur.execute('SELECT query_text, embedding, answer, sources, query_hash FROM cache ORDER BY hit_count DESC, created_at DESC LIMIT ?', (SEMANTIC_CACHE_MAX_CANDIDATES,))
            q_vec = np.array(query_embedding, dtype=np.float32); q_norm = np.linalg.norm(q_vec)
            if q_norm == 0: return None
            best_match, best_sim = None, 0.0
            for row in cur.fetchall():
                cached_emb = np.frombuffer(row[1], dtype=np.float32); c_norm = np.linalg.norm(cached_emb)
                if c_norm == 0: continue
                sim = float(np.dot(q_vec, cached_emb) / (q_norm * c_norm))
                if sim > best_sim: best_sim = sim; best_match = (row, sim)
            if best_match and best_match[1] >= self.threshold:
                row, sim = best_match
                with self.lock:
                    self.conn.execute('UPDATE cache SET hit_count = hit_count + 1 WHERE query_hash = ?', (row[4],)); self.conn.commit()
                return {"answer": row[2], "sources": json.loads(row[3]) if row[3] else [], "similarity": sim}
        except Exception: pass
        return None
    def set(self, query: str, query_embedding: Optional[List[float]], answer: str, sources: List[Dict]) -> None:
        if not SEMANTIC_CACHE_ENABLED or query_embedding is None: return
        try:
            q_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
            emb_blob = np.array(query_embedding, dtype=np.float32).tobytes()
            with self.lock, self.conn: self.conn.execute('INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 0)', (q_hash, query, emb_blob, answer, json.dumps(sources, ensure_ascii=False)))
        except Exception: pass
    def purge_stale(self, days: int = 30, min_hits: int = 1) -> int:
        try:
            with self.lock, self.conn:
                cur = self.conn.execute('DELETE FROM cache WHERE hit_count <= ? AND julianday("now") - julianday(created_at) > ?', (min_hits, days))
                return cur.rowcount
        except Exception: return 0

class ResponseCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or METADATA_DIR / "response_cache.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.conn: self.conn.execute('CREATE TABLE IF NOT EXISTS cache (query_hash TEXT PRIMARY KEY, hypothetical TEXT)')
    def get(self, query: str) -> Optional[str]:
        with self.lock:
            cur = self.conn.cursor(); cur.execute('SELECT hypothetical FROM cache WHERE query_hash = ?', (hashlib.md5(query.lower().strip().encode()).hexdigest(),))
            row = cur.fetchone(); return row[0] if row else None
    def set(self, query: str, hypothetical: str) -> None:
        with self.lock, self.conn: self.conn.execute('INSERT OR REPLACE INTO cache VALUES (?, ?)', (hashlib.md5(query.lower().strip().encode()).hexdigest(), hypothetical))

@dataclass
class RetrievalResult:
    chunk_id: str; content: str; full_content: Optional[str] = None; score: float = 0.0
    source: str = ""; page: int = 0; heading: str = ""; model: str = "unknown"
    chunk_type: str = "text"; metadata: Dict[str, Any] = None
    def __post_init__(self):
        if self.metadata is None: self.metadata = {}
    def to_context_string(self) -> str:
        parts = [f"[Документ: {self.source}, стр. {self.page}]"]
        if self.heading: parts.append(f"Раздел: {self.heading}")
        display_content = self.full_content or self.content
        if self.chunk_type == "table": parts.append(f"ТАБЛИЦА:\n{truncate_table_for_prompt(display_content)}")
        elif self.chunk_type == "formula": parts.append(f"ФОРМУЛА: {display_content}")
        elif self.chunk_type == "figure": parts.append(f"РИСУНОК: {display_content}")
        else: parts.append(display_content)
        return "\n".join(parts)

def _detect_query_intent(query: str) -> Dict[str, float]:
    query_lower = query.lower()
    intent = {"table": 0.0, "formula": 0.0, "figure": 0.0, "text": 1.0}
    t = sum(1 for kw in {"таблиц", "значени", "параметр"} if kw in query_lower)
    f = sum(1 for kw in {"формул", "вычислен", "расчёт"} if kw in query_lower)
    g = sum(1 for kw in {"схем", "рисунок", "диаграмм"} if kw in query_lower)
    total = t + f + g
    if total > 0:
        intent["table"] = min(t / 2.0, 1.0); intent["formula"] = min(f / 2.0, 1.0)
        intent["figure"] = min(g / 2.0, 1.0); intent["text"] = max(0.3, 1.0 - total * 0.2)
    return intent

class HybridRetriever:
    def __init__(self, index_builder: Optional[IndexBuilder] = None):
        self.index_builder = index_builder or IndexBuilder()
        self.index_builder.load()
        self._reranker = None
        self._cache = ResponseCache()

    def search(self, query: str, k: int = FINAL_TOP_K, model_filter: str = "", use_hyde: Optional[bool] = None, strict_model: Optional[bool] = None) -> List[RetrievalResult]:
        intent = _detect_query_intent(query)
        search_query = query
        if (use_hyde if use_hyde is not None else USE_HYDE):
            hyde = self._apply_hyde(query)
            if hyde: search_query = hyde

        vector_results = self._vector_search(search_query, intent, model_filter)
        bm25_results = self._bm25_search(query, model_filter)
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        try:
            from .quarantine import filter_quarantined_chunks
            fused = filter_quarantined_chunks(fused)
        except Exception: pass

        # Политика по модели (Проблема 4): соседние модели уже отсеяны на уровне
        # источников; здесь ставим точную модель вперёд и при строгом режиме
        # убираем 'unknown', если точных фрагментов достаточно.
        fused = self._apply_model_policy(fused, model_filter, strict_model)

        if USE_RERANKER and len(fused) > k:
            reranker = self._get_reranker()
            if reranker: fused = self._rerank(query, fused, reranker)

        results = fused[:k]
        for r in results:
            full = self.index_builder.docstore.get_content(r.chunk_id)
            if full:
                if r.chunk_type in ("table", "formula", "figure"):
                    r.full_content = _strip_passage_prefix(full)
                else:
                    r.content = _strip_passage_prefix(full)
            else:
                r.content = _strip_passage_prefix(r.content)

        # Context Expansion (Проблема 5): дотягиваем соседние текстовые чанки,
        # восстанавливая разорванные при чанкинге списки и абзацы.
        if CONTEXT_EXPANSION_ENABLED:
            self._expand_context(results)
        return results

    def _apply_model_policy(self, results: List[RetrievalResult], model_filter: str, strict_model: Optional[bool]) -> List[RetrievalResult]:
        if not model_filter or model_filter not in KNOWN_MODELS:
            return results
        strict = MODEL_FILTER_STRICT if strict_model is None else strict_model
        exact = [r for r in results if r.model == model_filter]
        unknown = [r for r in results if r.model not in KNOWN_MODELS]
        # чанки других известных моделей сюда не попадают (отсеяны в _vector/_bm25)
        if strict and len(exact) >= MODEL_STRICT_MIN_EXACT:
            return exact  # не подмешиваем 'unknown', чтобы не путать интерфейсы
        return exact + unknown

    def _expand_context(self, results: List[RetrievalResult]) -> None:
        docstore = self.index_builder.docstore
        if not hasattr(docstore, "get_neighbors"):
            return
        for r in results:
            if r.chunk_type != "text":
                continue
            seq = r.metadata.get("seq") if isinstance(r.metadata, dict) else None
            if seq is None:
                continue
            try:
                neighbors = docstore.get_neighbors(r.source, int(seq), window=CONTEXT_EXPANSION_WINDOW)
            except Exception:
                continue
            if not neighbors:
                continue
            before = [c for s, _cid, c, _t in neighbors if s < int(seq)]
            after = [c for s, _cid, c, _t in neighbors if s > int(seq)]
            base = r.full_content or r.content or ""
            stitched = "\n".join([*before, base, *after]).strip()
            stitched = _strip_passage_prefix(stitched)
            if len(stitched) > CONTEXT_EXPANSION_MAX_CHARS:
                stitched = stitched[:CONTEXT_EXPANSION_MAX_CHARS].rsplit(" ", 1)[0] + "…"
            if stitched and stitched != base:
                r.content = stitched
                r.metadata = dict(r.metadata or {})
                r.metadata["context_expanded"] = True

    def _vector_search(self, query: str, intent: Dict[str, float], model_filter: str) -> List[RetrievalResult]:
        results, seen_ids = [], set()
        search_text = f"{QUERY_PREFIX}{query}" if EMBEDDING_PROVIDER == "local" else query
        if self.index_builder.text_vectorstore:
            try:
                docs = self.index_builder.text_vectorstore.similarity_search_with_score(search_text, k=int(VECTOR_SEARCH_K * intent.get("text", 1.0)) + 3)
                for doc, score in docs:
                    cid = doc.metadata.get("chunk_id", "")
                    if cid in seen_ids: continue
                    seen_ids.add(cid)
                    if model_filter and doc.metadata.get("model", "unknown") not in ("unknown", model_filter): continue
                    results.append(RetrievalResult(chunk_id=cid, content=_strip_passage_prefix(doc.page_content), score=float(score), source=doc.metadata.get("source", ""), page=doc.metadata.get("page", 0), heading=doc.metadata.get("heading", ""), model=doc.metadata.get("model", "unknown"), chunk_type="text", metadata=doc.metadata))
            except Exception: pass
        if self.index_builder.summary_vectorstore:
            try:
                docs = self.index_builder.summary_vectorstore.similarity_search_with_score(search_text, k=int(VECTOR_SEARCH_K * max(intent["table"], intent["formula"], 0.3)) + 3)
                for doc, score in docs:
                    cid = doc.metadata.get("chunk_id", "")
                    if cid in seen_ids: continue
                    seen_ids.add(cid)
                    if model_filter and doc.metadata.get("model", "unknown") not in ("unknown", model_filter): continue
                    ct = doc.metadata.get("chunk_type", "text")
                    if ct not in ("table", "formula", "figure"): ct = "text"
                    results.append(RetrievalResult(chunk_id=cid, content=_strip_passage_prefix(doc.page_content), score=float(score), source=doc.metadata.get("source", ""), page=doc.metadata.get("page", 0), heading=doc.metadata.get("heading", ""), model=doc.metadata.get("model", "unknown"), chunk_type=ct, metadata=doc.metadata))
            except Exception: pass
        return results

    def _bm25_search(self, query: str, model_filter: str) -> List[RetrievalResult]:
        results = []
        for metadata, score in self.index_builder.bm25.search(query, k=BM25_SEARCH_K):
            if model_filter and metadata.get("model", "unknown") not in ("unknown", model_filter): continue
            results.append(RetrievalResult(chunk_id=metadata.get("chunk_id", ""), content=_strip_passage_prefix(metadata.get("content", "")), score=score, source=metadata.get("source", ""), page=metadata.get("page", 0), heading=metadata.get("heading", ""), model=metadata.get("model", "unknown"), chunk_type=metadata.get("chunk_type", "text"), metadata=metadata))
        return results

    def _reciprocal_rank_fusion(self, v_res, b_res, k_rrf=60) -> List[RetrievalResult]:
        scores, cmap = {}, {}
        for rank, r in enumerate(v_res, 1): scores.setdefault(r.chunk_id, 0.0); cmap[r.chunk_id] = r; scores[r.chunk_id] += HYBRID_ALPHA / (k_rrf + rank)
        for rank, r in enumerate(b_res, 1): scores.setdefault(r.chunk_id, 0.0); cmap[r.chunk_id] = r; scores[r.chunk_id] += (1 - HYBRID_ALPHA) / (k_rrf + rank)
        return [cmap[cid] for cid in sorted(scores.keys(), key=lambda x: scores[x], reverse=True)]

    def _rerank(self, query, results, reranker):
        try:
            scores = reranker.predict([(query, r.content) for r in results])
            for r, s in zip(results, scores): r.score = float(s)
            results.sort(key=lambda x: x.score, reverse=True)
        except Exception: pass
        return results

    def _apply_hyde(self, query: str) -> Optional[str]:
        cached = self._cache.get(query)
        if cached: return cached
        try:
            from .generation import LLMClient
            hyp = LLMClient().generate(f"Ответь кратко как эксперт:\n{query}", max_tokens=300)
            if hyp and len(hyp) > 20: self._cache.set(query, hyp); return hyp
        except Exception: pass
        return None

    def _get_reranker(self):
        if self._reranker: return self._reranker
        if not USE_RERANKER: return None
        try:
            from sentence_transformers import CrossEncoder
            try:
                self._reranker = CrossEncoder(RERANKER_MODEL, backend="onnx") if USE_ONNX_RERANKER else CrossEncoder(RERANKER_MODEL)
            except TypeError:
                self._reranker = CrossEncoder(RERANKER_MODEL)
            return self._reranker
        except Exception as exc:
            logger.warning(f"Reranker недоступен: {exc}")
            return None
