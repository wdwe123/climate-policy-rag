"""
classify_chunks.py
为 all_chunks_v11_6.jsonl 中每条 chunk 添加分类字段，用于 RAG 检索加权与噪声过滤。

新增字段：
  primary_tag     : str   — 主标签（见下方 PRIMARY_TAGS）
  secondary_tags  : list  — 0-2 个附加标签
  policy_score    : float — 0.0~1.0，对政策问答的有用程度
  retrieval_tier  : str   — primary / secondary / low_priority / exclude

primary_tag 选项（8 类）：
  action_policy      直接可执行的政策条款、目标、规定、合规要求
  implementation     实施细节、时间表、责任分工、程序步骤
  funding_program    资金项目、补贴、申请渠道、grant 信息
  background_context 背景说明、风险评估、情况概述（非 actionable）
  table_data         含实质数据的结构化表格
  reference_appendix 定义、缩写表、参考文献、附录索引
  administrative     封面、签名页、致谢、会议记录、联系信息、目录
  noise              OCR 乱码、近空白、无意义碎片

retrieval_tier 映射（默认权重）：
  primary      >= 0.75  (action_policy, funding_program, implementation)
  secondary    >= 0.40  (background_context, table_data)
  low_priority >= 0.10  (reference_appendix, administrative)
  exclude      <  0.10  (noise)

模型：anthropic.claude-haiku-4-5@20251001 via NYU Portkey
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from portkey_ai import Portkey

# ─── 配置 ────────────────────────────────────────────────────────────────────
PORTKEY_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PORTKEY_API_KEY  = os.environ.get("PORTKEY_API_KEY", "")
MODEL            = "@vertexai/anthropic.claude-haiku-4-5@20251001"

INPUT_PATH    = "E:/2026_capstone/policy_data/chunking_output/all_chunks_v11_6.jsonl"
OUTPUT_PATH   = "E:/2026_capstone/policy_data/chunking_output/all_chunks_classified.jsonl"
PROGRESS_PATH = "E:/2026_capstone/policy_data/chunking_output/classify_progress.json"

MAX_WORKERS  = 8
RETRY_LIMIT  = 3
RETRY_DELAY  = 5

PRIMARY_TAGS = {
    "action_policy", "implementation", "funding_program", "background_context",
    "table_data", "reference_appendix", "administrative", "noise",
}

# 默认 policy_score（快速规则用）
TAG_SCORE = {
    "action_policy":      1.00,
    "funding_program":    0.95,
    "implementation":     0.90,
    "table_data":         0.75,
    "background_context": 0.45,
    "reference_appendix": 0.25,
    "administrative":     0.10,
    "noise":              0.00,
}

def score_to_tier(score: float) -> str:
    if score >= 0.75: return "primary"
    if score >= 0.40: return "secondary"
    if score >= 0.10: return "low_priority"
    return "exclude"

# ─── Portkey 客户端 ──────────────────────────────────────────────────────────
client = Portkey(base_url=PORTKEY_BASE_URL, api_key=PORTKEY_API_KEY)

SYSTEM_PROMPT = """You are classifying chunks from US climate policy documents for a RAG system serving Native American communities in Arizona, New Mexico, and Oklahoma.

For each chunk, output ONLY a JSON object with these fields:
{
  "primary_tag": "<one of the 8 tags below>",
  "secondary_tags": ["<optional>", "<optional>"],
  "policy_score": <float 0.0-1.0>
}

PRIMARY TAG OPTIONS (choose exactly one):
- action_policy: Direct actionable policy text — regulations, requirements, goals, rules, mandates
- implementation: Implementation details — timelines, responsible parties, procedures, steps
- funding_program: Funding, grants, financial assistance programs, application processes, budgets
- background_context: Background, risk assessment, situation overview, statistics, history (not directly actionable)
- table_data: Structured tables with meaningful data (risk matrices, statistics, funding allocations)
- reference_appendix: Definitions, acronym lists, bibliographies, appendix indexes, legal citations
- administrative: Cover pages, signatures, acknowledgments, meeting minutes, contact info, TOC
- noise: Garbled OCR, near-empty content, repeated headers/footers, meaningless fragments

SECONDARY TAGS: 0-2 additional tags that also apply (use [] if none).

POLICY_SCORE: How useful is this chunk for answering policy questions?
  1.0 = directly answers "what policy applies / what can I apply for"
  0.0 = noise or purely administrative

Output ONLY the JSON. No explanation."""


def parse_llm_response(content: str) -> dict | None:
    """从 LLM 回复中提取 JSON，容忍格式不稳定。"""
    # 尝试直接解析
    try:
        return json.loads(content.strip())
    except Exception:
        pass
    # 提取 {...} 块
    match = re.search(r'\{.*?\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def classify_chunk(chunk: dict) -> dict:
    """调用 LLM 分类，返回添加了 4 个新字段的 chunk。"""
    text     = (chunk.get("text") or "").strip()
    section  = chunk.get("section") or ""
    is_table = chunk.get("is_table", False)
    is_toc   = chunk.get("is_toc", False)
    is_cover = chunk.get("is_cover", False)

    # ── 快速规则（无需 LLM）─────────────────────────────────────────────────
    def apply_tag(tag: str, score_override: float | None = None):
        score = score_override if score_override is not None else TAG_SCORE[tag]
        chunk["primary_tag"]    = tag
        chunk["secondary_tags"] = []
        chunk["policy_score"]   = score
        chunk["retrieval_tier"] = score_to_tier(score)
        return chunk

    if is_toc:
        return apply_tag("administrative")
    if is_cover:
        return apply_tag("administrative")
    if len(text) < 30:
        return apply_tag("noise")
    # 极短 table（可能是垃圾表格）
    if is_table and len(text) < 80:
        return apply_tag("noise")

    # ── LLM 分类 ─────────────────────────────────────────────────────────────
    user_content = (
        f"Document section: {section}\n"
        f"Is structured table: {is_table}\n\n"
        f"Chunk text (first 1500 chars):\n{text[:1500]}"
    )

    for attempt in range(RETRY_LIMIT):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=80,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = parse_llm_response(raw)

            if parsed:
                primary = parsed.get("primary_tag", "").strip().lower()
                if primary not in PRIMARY_TAGS:
                    # 尝试模糊匹配
                    for t in PRIMARY_TAGS:
                        if t in primary:
                            primary = t
                            break
                    else:
                        primary = "background_context"

                sec_raw = parsed.get("secondary_tags", [])
                secondary = [t for t in (sec_raw if isinstance(sec_raw, list) else [])
                             if t in PRIMARY_TAGS and t != primary][:2]

                raw_score = parsed.get("policy_score", TAG_SCORE.get(primary, 0.5))
                try:
                    score = float(raw_score)
                    score = max(0.0, min(1.0, score))
                except Exception:
                    score = TAG_SCORE.get(primary, 0.5)

                chunk["primary_tag"]    = primary
                chunk["secondary_tags"] = secondary
                chunk["policy_score"]   = round(score, 3)
                chunk["retrieval_tier"] = score_to_tier(score)
                return chunk

        except Exception as e:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"[ERROR] classify failed: {e}")

    # 兜底
    return apply_tag("background_context", 0.45)


# ─── 断点续传 ────────────────────────────────────────────────────────────────
def load_progress() -> set:
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return set(json.load(f).get("done_ids", []))
    return set()

def save_progress(done_ids: set):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump({"done_ids": list(done_ids)}, f)

# ─── 主流程 ──────────────────────────────────────────────────────────────────
def main():
    if not PORTKEY_API_KEY:
        print("[ERROR] PORTKEY_API_KEY 未设置，请先 export PORTKEY_API_KEY=...")
        return

    # 读取 chunks
    chunks = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            obj["_idx"] = i
            chunks.append(obj)
    print(f"总 chunks: {len(chunks)}")

    # 断点续传
    done_ids = load_progress()
    print(f"已完成: {len(done_ids)}，剩余: {len(chunks) - len(done_ids)}")

    results: dict[int, dict] = {}
    if done_ids and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                results[obj["_idx"]] = obj

    pending = [c for c in chunks if c["_idx"] not in done_ids]

    # 并发处理
    BATCH_SIZE = 200
    start_time = time.time()

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start : batch_start + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(classify_chunk, c): c["_idx"] for c in batch}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                    done_ids.add(idx)
                except Exception as e:
                    print(f"[WARN] idx={idx}: {e}")

        save_progress(done_ids)

        # 写出当前进度
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for i in sorted(results.keys()):
                obj = {k: v for k, v in results[i].items() if k != "_idx"}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        done_count = len(done_ids)
        elapsed = time.time() - start_time
        speed = (done_count - (len(chunks) - len(pending))) / max(elapsed, 1)
        remaining = (len(chunks) - done_count) / max(speed, 0.01)
        print(f"进度: {done_count}/{len(chunks)} ({done_count/len(chunks)*100:.1f}%) "
              f"— 速度 {speed:.1f} chunks/s — 预计剩余 {remaining/60:.1f} min")

    # 最终写出
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i in sorted(results.keys()):
            obj = {k: v for k, v in results[i].items() if k != "_idx"}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    tags   = [results[i].get("primary_tag", "unknown") for i in sorted(results)]
    tiers  = [results[i].get("retrieval_tier", "unknown") for i in sorted(results)]
    scores = [results[i].get("policy_score", 0) for i in sorted(results)]

    print("\n=== primary_tag 分布 ===")
    for tag, cnt in Counter(tags).most_common():
        print(f"  {tag:22s}: {cnt:5d} ({cnt/len(tags)*100:.1f}%)")

    print("\n=== retrieval_tier 分布 ===")
    for tier, cnt in Counter(tiers).most_common():
        print(f"  {tier:15s}: {cnt:5d} ({cnt/len(tiers)*100:.1f}%)")

    avg_score = sum(scores) / max(len(scores), 1)
    print(f"\n平均 policy_score: {avg_score:.3f}")
    print(f"输出: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
