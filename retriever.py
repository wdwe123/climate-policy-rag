"""
retriever.py
混合检索模块：向量检索（Qdrant）+ BM25 + RRF 融合 + LLM 重排（claude-haiku）

公开接口：
    search(query, top_k=30, top_n=10, geo_override=None) -> list[dict]
    TRIBES, COUNTIES, STATES  ← 地理词表（供 app.py 导出）

CLI 测试：
    python retriever.py "What water programs are available for Navajo Nation?"
"""

import json, os, pickle, re, sys, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import pandas as pd
from portkey_ai import Portkey
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

# ─── 配置 ────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PORTKEY_BASE_URL  = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PORTKEY_API_KEY   = os.environ.get("PORTKEY_API_KEY", "")
EMBED_MODEL       = "@vertexai/gemini-embedding-001"
HAIKU_MODEL       = "@vertexai/anthropic.claude-haiku-4-5@20251001"

# Qdrant: cloud mode (env vars) → local mode (fallback)
QDRANT_URL        = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
QDRANT_LOCAL_PATH = os.path.join(_BASE_DIR, "qdrant_storage")

COLLECTION_NAME   = "climate_policy"
BM25_INDEX_PATH   = os.path.join(_BASE_DIR, "bm25_index.pkl")
BM25_IDS_PATH     = os.path.join(_BASE_DIR, "bm25_corpus_ids.pkl")
METADATA_CSV      = os.path.join(_BASE_DIR, "pdf_data", "metadata", "policy_metadata_4.csv")

RRF_K             = 60
TIER_WEIGHT       = {"primary": 1.3, "secondary": 1.0, "low_priority": 0.6, "exclude": 0.0}
NOISE_THRESHOLD   = 0.2   # cross_score below this is treated as noise (0–1 scale from LLM)

# ─── 分层级检索常量 ───────────────────────────────────────────────────────────
LEVEL_ORDER = ["tribe", "city", "county", "state", "federal"]
LEVEL_LABEL = {
    "tribe":   "Tribal",
    "city":    "City",
    "county":  "County",
    "state":   "State",
    "federal": "Federal",
}
# Qdrant payload 中 policy_level 字段的实际值（首字母大写）
LEVEL_POLICY_VALUE = {
    "tribe":   ["Tribe"],
    "city":    ["City"],
    "county":  ["County"],
    "state":   ["State"],
    "federal": ["Federal", "Federal-Tribal Collab"],
}

# ─── 地理词典（从 metadata CSV 自动构建）────────────────────────────────────
def _load_geo_vocab():
    import ast
    df = pd.read_csv(METADATA_CSV)
    def collect(col):
        vals = set()
        for v in df[col].dropna():
            try:
                for x in ast.literal_eval(v):
                    if x and isinstance(x, str) and len(x.strip()) > 2:
                        vals.add(x.strip())
            except Exception:
                pass
        return vals
    tribes   = collect("tribe_list")
    counties = collect("county_list")
    cities   = collect("city_list")
    states   = collect("state_list")
    aliases = {
        "Navajo": "Navajo Nation", "Diné": "Navajo Nation",
        "Navajo Tribe": "Navajo Nation",
        "White Mountain Apache": "White Mountain Apache Tribe",
        "Hopi": "Hopi Tribe", "Tohono O'odham": "Tohono O'odham Nation",
        "Papago": "Tohono O'odham Nation",
        "Jicarilla Apache": "Jicarilla Apache Nation",
        "Mescalero Apache": "Mescalero Apache Tribe",
    }
    return tribes, counties, cities, states, aliases

TRIBES, COUNTIES, CITIES, STATES, ALIASES = _load_geo_vocab()

# ─── 初始化客户端 ─────────────────────────────────────────────────────────────
portkey = Portkey(base_url=PORTKEY_BASE_URL, api_key=PORTKEY_API_KEY)
if QDRANT_URL:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    qdrant = QdrantClient(path=QDRANT_LOCAL_PATH)

with open(BM25_INDEX_PATH, "rb") as f:
    _bm25 = pickle.load(f)
with open(BM25_IDS_PATH, "rb") as f:
    _bm25_ids = pickle.load(f)
_bm25_id_set = set(_bm25_ids)

# ─── 嵌入查询 ─────────────────────────────────────────────────────────────────
@lru_cache(maxsize=256)
def _embed_query(text: str) -> list[float]:
    resp = portkey.embeddings.create(
        model=EMBED_MODEL, input=text, encoding_format="float"
    )
    return resp.data[0].embedding

# ─── 地理实体提取 ─────────────────────────────────────────────────────────────
def _rule_match(text: str, vocab: set) -> list[str]:
    found = []
    tl = text.lower()
    for term in sorted(vocab, key=len, reverse=True):
        if re.search(r'\b' + re.escape(term.lower()) + r'\b', tl):
            found.append(term)
    return found

def _llm_geo_extract(query: str) -> dict:
    prompt = (
        "Extract geographic entities from this query. "
        "Return ONLY a JSON object with keys: tribes, counties, cities, states. "
        "Each value is a list of strings (empty list if none found). "
        "Example: {\"tribes\": [\"Navajo Nation\"], \"counties\": [], \"cities\": [], \"states\": [\"Arizona\"]}\n\n"
        f"Query: {query}"
    )
    try:
        resp = portkey.chat.completions.create(
            model=HAIKU_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"tribes": [], "counties": [], "cities": [], "states": []}

def extract_geo_entities(query: str) -> dict:
    """规则优先 + LLM 兜底。返回 {tribes, counties, cities, states}"""
    q = query
    for alias, canonical in ALIASES.items():
        q = re.sub(r'\b' + re.escape(alias) + r'\b', canonical, q, flags=re.IGNORECASE)

    tribes   = _rule_match(q, TRIBES)
    counties = _rule_match(q, COUNTIES)
    cities   = _rule_match(q, CITIES)
    states   = _rule_match(q, STATES)

    # G1：tribe 名首词与 county 名重叠时，tribe 优先
    if tribes:
        tribes_lower = {t.lower().split()[0] for t in tribes}
        counties = [c for c in counties if c.lower() not in tribes_lower]

    if not any([tribes, counties, cities, states]):
        llm_geo = _llm_geo_extract(query)
        tribes   = llm_geo.get("tribes", [])
        counties = llm_geo.get("counties", [])
        cities   = llm_geo.get("cities", [])
        states   = llm_geo.get("states", [])

    return {"tribes": tribes, "counties": counties, "cities": cities, "states": states}

# ─── Qdrant 过滤条件 ──────────────────────────────────────────────────────────
_EXCLUDE_FILTER = FieldCondition(key="retrieval_tier", match=MatchValue(value="exclude"))

def _not_exclude() -> Filter:
    return Filter(must_not=[_EXCLUDE_FILTER])

def _geo_filter(geo: dict, level: str) -> Filter | None:
    must_not = [_EXCLUDE_FILTER]
    if level == "tribe" and geo["tribes"]:
        return Filter(
            must=[FieldCondition(key="tribe_list", match=MatchAny(any=geo["tribes"]))],
            must_not=must_not,
        )
    if level == "county" and geo["counties"]:
        return Filter(
            must=[FieldCondition(key="county_list", match=MatchAny(any=geo["counties"]))],
            must_not=must_not,
        )
    if level == "city" and geo["cities"]:
        return Filter(
            must=[FieldCondition(key="city_list", match=MatchAny(any=geo["cities"]))],
            must_not=must_not,
        )
    if level == "state" and geo["states"]:
        return Filter(
            must=[FieldCondition(key="state_list", match=MatchAny(any=geo["states"]))],
            must_not=must_not,
        )
    return _not_exclude()

# ─── 向量检索 ─────────────────────────────────────────────────────────────────
def _vector_search(query_vec: list[float], filt: Filter, limit: int) -> list[dict]:
    resp = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=filt,
        limit=limit,
        with_payload=True,
    )
    return [{"id": h.id, "score": h.score, "payload": h.payload} for h in resp.points]

# ─── BM25 检索 ────────────────────────────────────────────────────────────────
def _bm25_search(query: str, limit: int, id_whitelist: set | None = None) -> list[dict]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    scores = _bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked:
        if score <= 0:
            break
        pid = _bm25_ids[idx]
        if id_whitelist and pid not in id_whitelist:
            continue
        results.append({"id": pid, "score": float(score)})
        if len(results) >= limit:
            break
    return results

# ─── RRF 融合 ─────────────────────────────────────────────────────────────────
def _rrf_fuse(
    vec_results: list[dict],
    bm25_results: list[dict],
    payload_map: dict,
    k: int = RRF_K,
) -> list[dict]:
    scores: dict[str, float] = {}
    for rank, r in enumerate(vec_results):
        pid = r["id"]
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    for rank, r in enumerate(bm25_results):
        pid = r["id"]
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)

    fused = []
    for pid, rrf_score in scores.items():
        payload = payload_map.get(pid, {})
        tier = payload.get("retrieval_tier", "secondary")
        weighted = rrf_score * TIER_WEIGHT.get(tier, 1.0)
        fused.append({"id": pid, "rrf_score": rrf_score, "score": weighted, "payload": payload})

    return sorted(fused, key=lambda x: x["score"], reverse=True)

# ─── 获取 Qdrant payload（for BM25 结果）────────────────────────────────────
def _fetch_payloads(ids: list[str]) -> dict:
    if not ids:
        return {}
    results = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=ids,
        with_payload=True,
    )
    return {r.id: r.payload for r in results}

# ─── LLM 重排（claude-haiku 批量打分）────────────────────────────────────────
_RERANK_CAP = 25   # 传给 haiku 的最大候选数，避免响应 JSON 被截断

def _rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """
    用 claude-haiku 对候选 chunk 批量打分（0–10），一次 API 调用完成重排。
    score 归一化为 cross_score（0–1）。解析失败时降级为 RRF 顺序。

    注意：最多传 _RERANK_CAP 条候选，避免 JSON 响应超出 max_tokens 被截断。
    """
    if not candidates:
        return []

    # 截断候选数，防止 haiku 响应 JSON 被 max_tokens 截断
    to_score = candidates[:_RERANK_CAP]

    passages = []
    for i, c in enumerate(to_score):
        text = (c["payload"].get("text") or "")[:400]
        passages.append(f"[{i}] {text}")

    # 每条记录约 20 chars，_RERANK_CAP=25 → 需要约 500 tokens；留 1.5× 余量
    resp_tokens = max(600, _RERANK_CAP * 20 + 200)

    prompt = (
        "You are a relevance scoring expert for climate and Indigenous policy documents.\n"
        "Rate each passage's relevance to the query on a scale of 0–10 (integer, 0=irrelevant, 10=directly answers the query).\n"
        "Return ONLY a JSON array: [{\"idx\": 0, \"score\": 8}, {\"idx\": 1, \"score\": 2}, ...]\n\n"
        f"Query: {query}\n\n"
        "Passages:\n" + "\n\n".join(passages)
    )

    try:
        resp = portkey.chat.completions.create(
            model=HAIKU_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=resp_tokens,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            scores_list = json.loads(m.group())
            score_map = {item["idx"]: item["score"] for item in scores_list}
            for i, c in enumerate(to_score):
                raw_score = score_map.get(i, 0)
                c["cross_score"] = float(raw_score) / 10.0
            # 未进入 to_score 的候选赋 0 分排最后
            for c in candidates[_RERANK_CAP:]:
                c["cross_score"] = 0.0
            return sorted(candidates, key=lambda x: x.get("cross_score", 0), reverse=True)
    except Exception:
        pass

    # 降级：RRF 顺序，设置中性分数
    for c in candidates:
        c["cross_score"] = 0.5
    return candidates

# ─── 分层级过滤条件（policy_level + 地理字段双重过滤）────────────────────────
def _level_filter(geo: dict, level: str) -> "Filter | None":
    """
    为指定层级构建 Qdrant 过滤器。
    同时过滤 policy_level 字段和地理字段。
    若该层级对应的地理实体为空，返回 None（跳过该层）。
    """
    must_not = [_EXCLUDE_FILTER]
    pl_cond  = FieldCondition(key="policy_level",
                              match=MatchAny(any=LEVEL_POLICY_VALUE[level]))
    if level == "tribe":
        if not geo.get("tribes"):
            return None
        return Filter(
            must=[pl_cond, FieldCondition(key="tribe_list",
                  match=MatchAny(any=geo["tribes"]))],
            must_not=must_not,
        )
    elif level == "city":
        if not geo.get("cities"):
            return None
        return Filter(
            must=[pl_cond, FieldCondition(key="city_list",
                  match=MatchAny(any=geo["cities"]))],
            must_not=must_not,
        )
    elif level == "county":
        if not geo.get("counties"):
            return None
        return Filter(
            must=[pl_cond, FieldCondition(key="county_list",
                  match=MatchAny(any=geo["counties"]))],
            must_not=must_not,
        )
    elif level == "state":
        if not geo.get("states"):
            return None
        return Filter(
            must=[pl_cond, FieldCondition(key="state_list",
                  match=MatchAny(any=geo["states"]))],
            must_not=must_not,
        )
    elif level == "federal":
        # Federal 层：只过滤 policy_level，无地理限制
        return Filter(must=[pl_cond], must_not=must_not)
    return None


# ─── 分层级检索主函数 ─────────────────────────────────────────────────────────
def search_all_levels(
    query: str,
    top_k_per_level: int = 8,
    top_n_per_level: int = 5,
    geo_override: dict | None = None,
) -> dict:
    """
    从用户指定的最具体地理层级开始，向上搜索所有层级（不向下）。

    层级顺序：tribe → city → county → state → federal
    - 用户指定 tribe  → 搜索 tribe + county + state + federal（city 跳过，无关联）
    - 用户指定 city   → 搜索 city + county + state + federal
    - 用户指定 county → 搜索 county + state + federal
    - 用户指定 state  → 搜索 state + federal
    - 无 geo         → 只搜 federal

    返回：dict[level_name → {chunks, has_results, geo_label}]
    has_results = True 当且仅当 该层有 chunk 且 max(cross_score) >= NOISE_THRESHOLD
    """
    query_vec = _embed_query(query)
    geo = geo_override or extract_geo_entities(query)

    # 确定起始层级
    if geo.get("tribes"):
        start_level = "tribe"
    elif geo.get("cities"):
        start_level = "city"
    elif geo.get("counties"):
        start_level = "county"
    elif geo.get("states"):
        start_level = "state"
    else:
        start_level = "federal"

    active_levels = LEVEL_ORDER[LEVEL_ORDER.index(start_level):]

    # 并行检索各层级（ThreadPoolExecutor，Qdrant 本地客户端是线程安全的）
    def _search_one_level(level: str):
        filt = _level_filter(geo, level)
        if filt is None:
            return level, None   # 跳过

        vec_hits  = _vector_search(query_vec, filt, top_k_per_level)
        whitelist = {r["id"] for r in vec_hits} if vec_hits else None
        bm25_hits = _bm25_search(query, top_k_per_level, id_whitelist=whitelist)

        payload_map = {r["id"]: r["payload"] for r in vec_hits}
        bm25_only_ids = [r["id"] for r in bm25_hits if r["id"] not in payload_map]
        if bm25_only_ids:
            payload_map.update(_fetch_payloads(bm25_only_ids))

        fused = _rrf_fuse(vec_hits, bm25_hits, payload_map)
        # 每层送入 rerank 前最多取 top_n_per_level+2 条，控制总候选数
        pre_rank_cap = top_n_per_level + 2
        for r in fused[:pre_rank_cap]:
            r["geo_level"] = level
        return level, fused[:pre_rank_cap]

    level_fused: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=len(active_levels)) as pool:
        futures = {pool.submit(_search_one_level, lv): lv for lv in active_levels}
        for fut in as_completed(futures):
            lv, fused = fut.result()
            if fused is not None:
                level_fused[lv] = fused

    searched_levels = [lv for lv in active_levels if lv in level_fused]
    all_candidates  = [r for lv in searched_levels for r in level_fused[lv]]

    if not all_candidates:
        return {}

    # 一次 haiku rerank（候选数已被 pre_rank_cap 控制，不超过 _RERANK_CAP）
    ranked = _rerank(query, all_candidates, len(all_candidates))

    # 按层级分组，计算 has_results
    level_results: dict = {}
    for level in searched_levels:
        chunks = [r for r in ranked if r.get("geo_level") == level][:top_n_per_level]
        best   = max((r.get("cross_score", 0) for r in chunks), default=0)
        # has_results：数量 > 0 且最高分 >= 阈值（两者结合）
        has_results = len(chunks) > 0 and best >= NOISE_THRESHOLD

        if level == "tribe":
            geo_label = ", ".join(geo.get("tribes", [])) or "Tribal Area"
        elif level == "city":
            geo_label = ", ".join(geo.get("cities", [])) or "City"
        elif level == "county":
            geo_label = ", ".join(geo.get("counties", [])) or "County"
        elif level == "state":
            geo_label = ", ".join(geo.get("states", [])) or "State"
        else:
            geo_label = "Federal"

        # 附加 geo 信息供 app.py 调试用
        for r in chunks:
            r["geo_tier"] = start_level
            r["geo"]      = geo

        level_results[level] = {
            "chunks":      chunks,
            "has_results": has_results,
            "geo_label":   geo_label,
        }

    return level_results


# ─── 主检索函数 ───────────────────────────────────────────────────────────────
def search(
    query: str,
    top_k: int = 30,
    top_n: int = 10,
    geo_override: dict | None = None,
) -> list[dict]:
    """
    混合检索入口。
    返回 top_n 个 chunk dict，每个含原始 payload + score + cross_score + geo_tier + geo_level。
    geo_level: "local"（tribe/county/city 级）| "regional"（state 级）| "none"
    """
    query_vec = _embed_query(query)
    geo = geo_override or extract_geo_entities(query)

    # 确定地理过滤层级
    geo_tier = "none"
    if geo["tribes"]:
        geo_tier = "tribe"
    elif geo["counties"]:
        geo_tier = "county"
    elif geo["cities"]:
        geo_tier = "city"
    elif geo["states"]:
        geo_tier = "state"

    # 向量检索：本地级（精确）+ 州级（补充）
    primary_filter = _geo_filter(geo, geo_tier)
    state_filter   = _geo_filter(geo, "state") if geo_tier not in ("state", "none") else None

    vec_main  = _vector_search(query_vec, primary_filter, top_k)
    vec_state = _vector_search(query_vec, state_filter, top_k // 2) if state_filter else []

    # 记录 local / regional 来源 ID
    local_ids    = {r["id"] for r in vec_main}
    regional_ids = {r["id"] for r in vec_state} - local_ids

    # 合并向量结果，去重
    seen_ids: set[str] = set()
    vec_all: list[dict] = []
    for r in vec_main + vec_state:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            vec_all.append(r)

    # BM25 检索（地理白名单保持一致性）
    whitelist    = {r["id"] for r in vec_all} if geo_tier != "none" else None
    bm25_results = _bm25_search(query, top_k, id_whitelist=whitelist)

    # 合并 payload map
    payload_map = {r["id"]: r["payload"] for r in vec_all}
    bm25_only_ids = [r["id"] for r in bm25_results if r["id"] not in payload_map]
    if bm25_only_ids:
        payload_map.update(_fetch_payloads(bm25_only_ids))

    # RRF 融合
    fused = _rrf_fuse(vec_all, bm25_results, payload_map)[:top_k]

    # LLM 重排
    ranked = _rerank(query, fused, top_n)

    # D1：per-document 去重，每个文档最多 2 个 chunk
    seen_docs: dict[str, int] = {}
    deduped = []
    for r in ranked:
        doc = r["payload"].get("doc_title", r["id"])
        if seen_docs.get(doc, 0) < 2:
            seen_docs[doc] = seen_docs.get(doc, 0) + 1
            deduped.append(r)
        if len(deduped) >= top_n * 2:
            break

    # N1：过滤低相关性噪声（保底保留至少 5 条）
    filtered = [r for r in deduped if r.get("cross_score", 1) >= NOISE_THRESHOLD]
    results = filtered if len(filtered) >= min(5, top_n) else deduped
    results = results[:top_n]

    # 附加 geo 信息 + geo_level 标签
    for r in results:
        r["geo_tier"] = geo_tier
        r["geo"] = geo
        pid = r["id"]
        if geo_tier in ("tribe", "county", "city"):
            r["geo_level"] = "local" if pid in local_ids else "regional"
        elif geo_tier == "state":
            r["geo_level"] = "regional"
        else:
            r["geo_level"] = "none"

    return results

# ─── CLI 测试入口 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import atexit
    atexit.register(qdrant.close)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "water resources programs for Navajo Nation"
    print(f"\nQuery: {query}\n{'='*60}")

    results = search(query)
    geo = results[0]["geo"] if results else {}
    print(f"Geo entities: {geo}")
    print(f"Returned {len(results)} results\n{'-'*60}")

    for i, r in enumerate(results, 1):
        p = r["payload"]
        print(f"[{i}] cross={r['cross_score']:.2f}  level={r['geo_level']}  tier={p.get('retrieval_tier')}  tag={p.get('primary_tag')}")
        print(f"    Doc:     {p.get('doc_title','')[:60]}")
        print(f"    Section: {p.get('section','')[:60]}  Pages: {p.get('pages','')}")
        print(f"    URL:     {p.get('source_url','(none)')}")
        print(f"    Text:    {(p.get('text') or '')[:200].replace(chr(10),' ')}")
        print()
