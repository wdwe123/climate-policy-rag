"""
build_bm25.py
从 all_chunks_classified.jsonl 构建 BM25 索引并持久化。
输出：bm25_index.pkl, bm25_corpus_ids.pkl
"""
import json, pickle, re, time
from pathlib import Path
from rank_bm25 import BM25Okapi

INPUT   = "E:/2026_capstone/policy_data/chunking_output/all_chunks_classified.jsonl"
OUT_IDX = "E:/2026_capstone/policy_data/bm25_index.pkl"
OUT_IDS = "E:/2026_capstone/policy_data/bm25_corpus_ids.pkl"
TEXT_MAX = 6000

def build_embed_text(chunk: dict) -> str:
    parts = []
    if chunk.get("doc_title"):
        parts.append(chunk["doc_title"])
    if chunk.get("section") and chunk["section"] != "(no heading)":
        parts.append(chunk["section"])
    prefix = " | ".join(parts)
    body = (chunk.get("text") or "").strip()
    full = f"{prefix}\n{body}" if prefix else body
    return full[:TEXT_MAX]

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def main():
    corpus_tokens = []
    corpus_ids = []   # Qdrant point ID（与 embed_chunks.py 相同的 make_chunk_id 逻辑）

    import hashlib
    def make_id(chunk, idx):
        key = f"{chunk.get('policy_id','')}_p{chunk.get('pages','')}_c{idx}"
        h = hashlib.md5(key.encode()).hexdigest()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"

    skipped = 0
    with open(INPUT, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("retrieval_tier") == "exclude":
                skipped += 1
                continue
            text = build_embed_text(obj)
            corpus_tokens.append(tokenize(text))
            corpus_ids.append(make_id(obj, i))

    print(f"构建 BM25：{len(corpus_tokens)} 条（跳过 {skipped} 条 exclude）")
    t0 = time.time()
    bm25 = BM25Okapi(corpus_tokens)
    print(f"BM25 构建完成，耗时 {time.time()-t0:.1f}s")

    with open(OUT_IDX, "wb") as f:
        pickle.dump(bm25, f)
    with open(OUT_IDS, "wb") as f:
        pickle.dump(corpus_ids, f)
    print(f"已保存：{OUT_IDX}")
    print(f"已保存：{OUT_IDS}")

if __name__ == "__main__":
    main()
