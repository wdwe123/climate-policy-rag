"""
generator.py
基于检索结果生成带溯源的答案（claude-sonnet-4-6 via Portkey @vertexai）
"""
import os
from portkey_ai import Portkey

PORTKEY_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PORTKEY_API_KEY  = os.environ.get("PORTKEY_API_KEY", "")
GEN_MODEL        = "@vertexai/anthropic.claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a climate policy assistant helping Indigenous communities in Arizona, New Mexico, and Oklahoma find relevant policy information.

Answer the user's question using ONLY the provided context chunks. For each key claim, cite the source using [Doc N] notation.

Rules:
- Be concise and direct. Lead with the most actionable information.
- You will be given a "Geographic coverage summary" that lists each geographic level searched and whether relevant results were found. Structure your answer by level in this exact format:
    **[Level] ([Location]):** <summary of relevant policies, or "No relevant [level]-level policies were found.">
  Cover every level listed in the summary. Order: Tribal → City → County → State → Federal.
- If only one level is present (no summary provided), answer directly without level headers.
- If the context doesn't contain enough information, say so clearly.
- Never fabricate information not in the context.
- End with a "Sources" section listing each cited document with its URL."""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, r in enumerate(chunks, 1):
        p    = r["payload"]
        # Use display_name injected by app.py if available, else fall back to doc_title
        doc   = p.get("display_name") or p.get("doc_title", "Unknown")
        sec   = p.get("section", "")
        pages = p.get("pages", "")
        url   = p.get("source_url", "")
        text  = (p.get("text") or "").strip()[:1500]
        tier  = p.get("retrieval_tier", "")
        tag   = p.get("primary_tag", "")
        level = r.get("geo_level", "")
        geo_label = f" | Scope: {level}" if level and level != "none" else ""
        parts.append(
            f"[Doc {i}] {doc}\n"
            f"Section: {sec} | Pages: {pages} | Type: {tag} | Priority: {tier}{geo_label}\n"
            f"URL: {url}\n"
            f"{text}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    chunks: list[dict],
    level_summary: dict | None = None,
    max_tokens: int = 1200,
) -> dict:
    """
    生成答案。返回 {answer: str, model: str, input_chunks: int}

    level_summary: {level → {geo_label: str, has_results: bool}}
    若提供，会在 user_msg 末尾附加地理覆盖摘要，引导模型按层级结构作答。
    """
    if not chunks:
        return {"answer": "No relevant policy information was found for your query.",
                "model": GEN_MODEL, "input_chunks": 0}

    portkey = Portkey(base_url=PORTKEY_BASE_URL, api_key=PORTKEY_API_KEY)
    context = build_context(chunks)

    user_msg = f"Question: {query}\n\nContext:\n{context}"

    if level_summary:
        lines = []
        for level, info in level_summary.items():
            status = "has relevant results" if info["has_results"] else "NO relevant results found"
            lines.append(f"- {level.capitalize()} ({info['geo_label']}): {status}")
        user_msg += (
            "\n\nGeographic coverage summary (structure your answer by these levels):\n"
            + "\n".join(lines)
        )

    resp = portkey.chat.completions.create(
        model=GEN_MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=max_tokens,
    )
    answer = resp.choices[0].message.content.strip()
    return {"answer": answer, "model": GEN_MODEL, "input_chunks": len(chunks)}
