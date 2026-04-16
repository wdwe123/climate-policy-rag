"""
embed_chunks.py
将 all_chunks_classified.jsonl 中的 chunks embed 并写入本地 Qdrant。

- 模型: gemini-embedding-001 via NYU Portkey (@vertexai/gemini-embedding-001, dim=3072)
- 跳过 retrieval_tier == "exclude" 的 chunks
- 支持断点续传（通过 Qdrant scroll 检查已有 ID）
- 文本格式: "{doc_title} | {section}\n{text[:2000]}"
- Qdrant 本地存储: ./qdrant_storage/
"""

import json
import os
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from portkey_ai import Portkey
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType
)

# ─── 配置 ────────────────────────────────────────────────────────────────────
PORTKEY_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PORTKEY_API_KEY  = os.environ.get("PORTKEY_API_KEY", "")
EMBED_MODEL      = "@vertexai/gemini-embedding-001"
VECTOR_DIM       = 3072

INPUT_PATH       = "E:/2026_capstone/policy_data/chunking_output/all_chunks_classified.jsonl"
QDRANT_PATH      = "E:/2026_capstone/policy_data/qdrant_storage"
COLLECTION_NAME  = "climate_policy"

MAX_WORKERS      = 6    # 并发 embedding 请求数
BATCH_SIZE       = 10   # 每次 API 调用的 chunk 数
RETRY_LIMIT      = 3
RETRY_DELAY      = 5
TEXT_MAX_CHARS   = 6000  # Gemini embedding 输入上限约 2048 tokens ≈ 8000 chars，保守取 6000

# ─── 初始化 ──────────────────────────────────────────────────────────────────
portkey = Portkey(base_url=PORTKEY_BASE_URL, api_key=PORTKEY_API_KEY)

qdrant = QdrantClient(path=QDRANT_PATH)
_qdrant_lock = threading.Lock()  # 本地 Qdrant 不支持并发写，需串行化


def ensure_collection():
    """创建或验证 Qdrant collection，维度不匹配时自动重建。"""
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        # 检查维度是否匹配
        info = qdrant.get_collection(COLLECTION_NAME)
        existing_dim = info.config.params.vectors.size
        if existing_dim != VECTOR_DIM:
            print(f"[Qdrant] 维度不匹配（现有 {existing_dim} ≠ 目标 {VECTOR_DIM}），删除并重建。")
            qdrant.delete_collection(COLLECTION_NAME)
        else:
            count = qdrant.count(COLLECTION_NAME).count
            print(f"[Qdrant] Collection '{COLLECTION_NAME}' 已存在，维度 {existing_dim}，当前 {count} 条向量。")
            return

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    # 创建 payload 索引，加速过滤
    for field, schema in [
        ("policy_level",   PayloadSchemaType.KEYWORD),
        ("policy_type",    PayloadSchemaType.KEYWORD),
        ("primary_tag",    PayloadSchemaType.KEYWORD),
        ("retrieval_tier", PayloadSchemaType.KEYWORD),
        ("is_table",       PayloadSchemaType.BOOL),
        ("is_appendix",    PayloadSchemaType.BOOL),
        ("policy_score",   PayloadSchemaType.FLOAT),
    ]:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )
    print(f"[Qdrant] Collection '{COLLECTION_NAME}' 创建完成（dim={VECTOR_DIM}），已建立 payload 索引。")


def make_chunk_id(chunk: dict, idx: int) -> str:
    """生成确定性 UUID 字符串，用于断点续传去重。"""
    key = f"{chunk.get('policy_id','')}_p{chunk.get('pages','')}_c{idx}"
    h = hashlib.md5(key.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def build_embed_text(chunk: dict) -> str:
    """构建 embedding 输入文本：标题 + 章节 + 正文。"""
    parts = []
    if chunk.get("doc_title"):
        parts.append(chunk["doc_title"])
    if chunk.get("section") and chunk["section"] != "(no heading)":
        parts.append(chunk["section"])
    prefix = " | ".join(parts)
    body = (chunk.get("text") or "").strip()
    full = f"{prefix}\n{body}" if prefix else body
    return full[:TEXT_MAX_CHARS]


def embed_batch(texts: list[str]) -> list[list[float]] | None:
    """调用 Portkey embedding API，返回向量列表。"""
    for attempt in range(RETRY_LIMIT):
        try:
            resp = portkey.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
                encoding_format="float",
            )
            return [item.embedding for item in resp.data]
        except Exception as e:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"[ERROR] embedding failed: {e}")
                return None


def chunk_to_payload(chunk: dict) -> dict:
    """提取 Qdrant payload（去掉大字段 text，单独存 text_preview）。"""
    skip = {"_idx"}
    payload = {k: v for k, v in chunk.items() if k not in skip}
    # text 保留完整（RAG 需要）
    return payload


# ─── 主流程 ──────────────────────────────────────────────────────────────────
def main():
    if not PORTKEY_API_KEY:
        print("[ERROR] PORTKEY_API_KEY 未设置")
        return

    ensure_collection()

    # 读取 chunks
    all_chunks = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_idx"] = i
            all_chunks.append(obj)

    print(f"总 chunks: {len(all_chunks)}")

    # 过滤 exclude
    chunks = [c for c in all_chunks if c.get("retrieval_tier") != "exclude"]
    print(f"跳过 exclude: {len(all_chunks) - len(chunks)} 条")
    print(f"待 embed: {len(chunks)} 条")

    # 生成 ID，检查已存在（断点续传）
    ids = [make_chunk_id(c, c["_idx"]) for c in chunks]
    existing_ids = set()
    try:
        offset = None
        while True:
            result, offset = qdrant.scroll(
                COLLECTION_NAME, limit=1000, offset=offset, with_payload=False, with_vectors=False
            )
            for pt in result:
                existing_ids.add(pt.id)
            if offset is None:
                break
    except Exception:
        pass
    print(f"Qdrant 中已有: {len(existing_ids)} 条，跳过。")

    # 过滤待处理
    pending = [(c, cid) for c, cid in zip(chunks, ids) if cid not in existing_ids]
    print(f"实际需要 embed: {len(pending)} 条")

    if not pending:
        print("全部已完成！")
        return

    # 分 batch 并发处理
    start_time = time.time()
    done_count = 0
    errors = 0

    def process_batch(batch: list[tuple[dict, str]]) -> int:
        """embed 一个 batch 并写入 Qdrant，返回成功数。"""
        batch_chunks, batch_ids = zip(*batch)
        texts = [build_embed_text(c) for c in batch_chunks]
        vectors = embed_batch(list(texts))
        if vectors is None:
            return 0
        points = [
            PointStruct(
                id=bid,
                vector=vec,
                payload=chunk_to_payload(bc),
            )
            for bid, vec, bc in zip(batch_ids, vectors, batch_chunks)
        ]
        with _qdrant_lock:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    # 将 pending 切成 batches
    batches = [pending[i:i+BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_batch, b): b for b in batches}
        for future in as_completed(futures):
            try:
                n = future.result()
                done_count += n
            except Exception as e:
                errors += 1
                print(f"[WARN] batch error: {e}")

            if done_count % 500 == 0 or done_count == len(pending):
                elapsed = time.time() - start_time
                speed = done_count / max(elapsed, 1)
                remaining = (len(pending) - done_count) / max(speed, 0.01)
                pct = done_count / len(pending) * 100
                print(f"进度: {done_count}/{len(pending)} ({pct:.1f}%) "
                      f"— {speed:.1f} chunks/s — 剩余 {remaining/60:.1f} min")

    # 最终统计
    total_in_db = qdrant.count(COLLECTION_NAME).count
    print(f"\n=== 完成 ===")
    print(f"成功写入: {done_count}，错误: {errors}")
    print(f"Qdrant 总向量数: {total_in_db}")
    print(f"存储路径: {QDRANT_PATH}")


if __name__ == "__main__":
    main()
