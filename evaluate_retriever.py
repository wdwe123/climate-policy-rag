"""
evaluate_retriever.py
RAG 检索质量评估脚本

指标：Recall@5, MRR@5 (文档级，跨所有活跃层级)

运行前必须：
    export PORTKEY_API_KEY="your-key"
    python evaluate_retriever.py

输出：
    - 每条问题的检索结果摘要
    - 最终 Recall@5 和 MRR@5 汇总
"""

import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retriever

# ─── Golden Set ──────────────────────────────────────────────────────────────
# 每条记录：(query, geo_override, expected_doc_titles, note)
# expected_doc_titles: list of substrings，任意一个出现在 doc_title 中即为命中
# geo_override: 模拟 UI 选择的地理实体（None = 让 retriever 自动提取）

GOLDEN_SET = [
    # ── Tribal Level: Arizona ──────────────────────────────────────────────
    (
        "What climate change adaptation strategies does Navajo Nation use for water resources?",
        {"tribes": ["Navajo Nation"], "counties": [], "cities": [], "states": ["Arizona"]},
        ["96955ab4f0", "Climate Change Adaptation Plan"],
        "Navajo AZ climate adaptation",
    ),
    (
        "What natural disaster risks are identified in the Hopi Tribe hazard mitigation plan?",
        {"tribes": ["Hopi Tribe"], "counties": [], "cities": [], "states": ["Arizona"]},
        ["52bdece409", "Hopi-Hazard-Mitigation"],
        "Hopi AZ hazard mitigation",
    ),
    (
        "How does Havasupai Tribe prepare for floods and wildfires?",
        {"tribes": ["Havasupai Tribe"], "counties": [], "cities": [], "states": ["Arizona"]},
        ["1134f85730", "Havasupai-Hazard-Mitigation"],
        "Havasupai AZ hazard mitigation",
    ),
    (
        "What environmental regulations apply to development on White Mountain Apache land?",
        {"tribes": ["White Mountain Apache Tribe"], "counties": [], "cities": [], "states": ["Arizona"]},
        ["d3ed9dfc5d", "Environmental Code"],
        "White Mountain Apache env code",
    ),
    (
        "What water management findings does the Kaibab-Paiute Tribe ten-year report show?",
        {"tribes": ["Kaibab- Paiute Tribe"], "counties": [], "cities": [], "states": ["Arizona"]},
        ["c4da2ca78b", "TenYearSummaryReport"],
        "Kaibab Paiute adaptive mgmt",
    ),

    # ── Tribal Level: New Mexico ───────────────────────────────────────────
    (
        "What are Sandia Pueblo's priorities for reducing greenhouse gas emissions?",
        {"tribes": ["Sandia Pueblo"], "counties": [], "cities": [], "states": ["New Mexico"]},
        ["0eeaa65ff4", "pueblo-of-sandia-pcap"],
        "Sandia Pueblo NM climate",
    ),
    (
        "What climate vulnerabilities does Tesuque Pueblo face and how do they plan to adapt?",
        {"tribes": ["Tesuque Pueblo"], "counties": [], "cities": [], "states": ["New Mexico"]},
        ["067ee46c19", "pueblo-of-tesuque-pcap"],
        "Tesuque Pueblo NM climate",
    ),
    (
        "What are Acoma Pueblo's water quality standards and environmental monitoring programs?",
        {"tribes": ["Acoma Pueblo"], "counties": [], "cities": [], "states": ["New Mexico"]},
        ["5abc0f2a17", "acoma-wqs"],
        "Acoma Pueblo NM water quality",
    ),
    (
        "What renewable energy or solar projects has Picuris Pueblo implemented?",
        {"tribes": ["Picuris Pueblo"], "counties": [], "cities": [], "states": ["New Mexico"]},
        ["22b520dec6", "picuris_pueblo"],
        "Picuris Pueblo NM energy",
    ),
    (
        "What energy efficiency programs does Zia Pueblo have?",
        {"tribes": ["Zia Pueblo"], "counties": [], "cities": [], "states": ["New Mexico"]},
        ["26a7f60261", "zia_pueblo"],
        "Zia Pueblo NM energy",
    ),
    (
        "What are Navajo Nation community leaders' recommendations on climate change in New Mexico?",
        {"tribes": ["Navajo Nation"], "counties": [], "cities": [], "states": ["New Mexico"]},
        ["80aae511fc", "Navajo Nation Community Leaders Clim"],
        "Navajo NM community climate report",
    ),
    (
        "What climate adaptation plan has Zuni Pueblo developed?",
        {"tribes": ["Zuni Pueblo"], "counties": [], "cities": [], "states": ["New Mexico"]},
        ["50812389", "ZuniFinal"],
        "Zuni Pueblo NM climate",
    ),

    # ── Tribal Level: Oklahoma ─────────────────────────────────────────────
    (
        "What are Choctaw Nation's priority actions for reducing carbon emissions?",
        {"tribes": ["Choctaw Nation"], "counties": [], "cities": [], "states": ["Oklahoma"]},
        ["5e985facdb", "priority-climate-action-plan"],
        "Choctaw Nation OK climate",
    ),
    (
        "How does Muscogee Creek Nation address climate adaptation and vulnerability?",
        {"tribes": ["Muscogee (Creek) Nation"], "counties": [], "cities": [], "states": ["Oklahoma"]},
        ["7f804ff08d", "muscogee-creek-nation-pcap", "0664da1088"],
        "Muscogee Creek Nation OK climate",
    ),
    (
        "What climate vulnerabilities and adaptation strategies has Delaware Tribe of Indians identified?",
        {"tribes": ["Delaware Tribe of Indians"], "counties": [], "cities": [], "states": ["Oklahoma"]},
        ["6622d84ba7", "Climate-Adaptation-Plan-and-Vulnerabil"],
        "Delaware Tribe OK climate",
    ),
    (
        "What are Kickapoo Tribe's climate priorities and action plan?",
        {"tribes": ["Kickapoo Tribe"], "counties": [], "cities": [], "states": ["Oklahoma"]},
        ["575022b4bb", "KTO_PCAP"],
        "Kickapoo Tribe OK climate",
    ),
    (
        "What climate adaptation plan does Kiowa Tribe have?",
        {"tribes": ["Kiowa Tribe"], "counties": [], "cities": [], "states": ["Oklahoma"]},
        ["7a05fa80b1", "kiowa-tribe-pcap"],
        "Kiowa Tribe OK climate",
    ),

    # ── State Level ────────────────────────────────────────────────────────
    (
        "What are Arizona's drought preparedness measures and water conservation policies?",
        {"tribes": [], "counties": [], "cities": [], "states": ["Arizona"]},
        ["2004 Arizona Drought Preparedness Plan"],
        "Arizona State drought plan",
    ),
    (
        "What is Arizona's plan for clean energy and reducing greenhouse gas emissions?",
        {"tribes": [], "counties": [], "cities": [], "states": ["Arizona"]},
        ["the-clean-arizona-plan"],
        "Arizona clean energy plan",
    ),
    (
        "What natural hazards does Arizona's state hazard mitigation plan address?",
        {"tribes": [], "counties": [], "cities": [], "states": ["Arizona"]},
        ["SHMP_2023_Final"],
        "Arizona SHMP 2023",
    ),
    (
        "What is Oklahoma's energy security strategy and renewable energy goals?",
        {"tribes": [], "counties": [], "cities": [], "states": ["Oklahoma"]},
        ["Oklahoma-Energy-Security-Plan"],
        "Oklahoma energy security",
    ),
    (
        "What is New Mexico's climate adaptation and resilience plan?",
        {"tribes": [], "counties": [], "cities": [], "states": ["New Mexico"]},
        ["NM-C.A.R.P.", "07b0a2ad75"],
        "New Mexico climate resilience plan",
    ),

    # ── Federal Level ──────────────────────────────────────────────────────
    (
        "What does Clean Water Act Section 106 require for tribal water quality programs?",
        {"tribes": [], "counties": [], "cities": [], "states": []},
        ["02bc95d3e1", "clean-water-act-section-106-tribal-gui"],
        "Federal CWA 106 tribal guidance",
    ),
    (
        "How can tribes build environmental program capacity through EPA grants?",
        {"tribes": [], "counties": [], "cities": [], "states": []},
        ["eddd2209c5", "GAP-guidance"],
        "Federal EPA GAP tribal capacity",
    ),
    (
        "What is the water settlement agreement for Choctaw Nation water rights?",
        {"tribes": [], "counties": [], "cities": [], "states": []},
        ["ef9b6a8603", "cno-water-settlement-agreement"],
        "Federal Choctaw water rights settlement",
    ),
    (
        "What drought contingency planning approaches exist for the Arbuckle-Simpson Aquifer in Oklahoma?",
        {"tribes": [], "counties": [], "cities": [], "states": ["Oklahoma"]},
        ["Fostering Partnerships", "Drought Contingency Plan"],
        "Federal OK Arbuckle drought",
    ),
]


# ─── Metrics ─────────────────────────────────────────────────────────────────
def hits_doc(chunks: list[dict], expected_substrings: list[str]) -> tuple[bool, int]:
    """
    在 chunks 的 doc_title 中搜索 expected_substrings（任意一个匹配即命中）。
    返回 (hit: bool, rank: int)，rank 从 1 开始，未命中返回 0。
    """
    for rank, chunk in enumerate(chunks, 1):
        doc_title = chunk.get("payload", {}).get("doc_title", "")
        for sub in expected_substrings:
            if sub.lower() in doc_title.lower():
                return True, rank
    return False, 0


def recall_at_k(hit: bool) -> float:
    return 1.0 if hit else 0.0


def reciprocal_rank(rank: int) -> float:
    return 1.0 / rank if rank > 0 else 0.0


# ─── Run Evaluation ──────────────────────────────────────────────────────────
def main():
    if not os.environ.get("PORTKEY_API_KEY"):
        print("ERROR: PORTKEY_API_KEY not set. Run: export PORTKEY_API_KEY='your-key'")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"RAG Evaluation — {len(GOLDEN_SET)} queries")
    print(f"{'='*70}\n")

    results_log = []
    total_recall = 0.0
    total_rr = 0.0
    n = len(GOLDEN_SET)

    for i, (query, geo_override, expected_subs, note) in enumerate(GOLDEN_SET, 1):
        print(f"[{i:02d}/{n}] {note}")
        print(f"       Query: {query[:75]}")

        t0 = time.time()
        try:
            level_results = retriever.search_all_levels(
                query,
                top_k_per_level=8,
                top_n_per_level=5,
                geo_override=geo_override if any(geo_override.values()) else None,
            )
        except Exception as e:
            print(f"       ERROR: {e}")
            results_log.append({"note": note, "hit": False, "rank": 0, "error": str(e)})
            n -= 1
            continue

        elapsed = time.time() - t0

        # 把所有层级的 chunks 合并（按层级顺序），取前 5 条用于 Recall@5
        all_chunks = []
        for level in retriever.LEVEL_ORDER:
            lvl = level_results.get(level, {})
            all_chunks.extend(lvl.get("chunks", []))

        top5 = all_chunks[:5]
        hit, rank = hits_doc(top5, expected_subs)
        r5 = recall_at_k(hit)
        rr = reciprocal_rank(rank)
        total_recall += r5
        total_rr += rr

        # 层级摘要
        level_summary = {
            lv: ("✅ has_results" if d.get("has_results") else "❌ no results")
            for lv, d in level_results.items()
        }

        # 前 5 条命中的 doc_title
        top5_titles = [c.get("payload", {}).get("doc_title", "?")[:45] for c in top5]

        status = f"HIT@{rank}" if hit else "MISS"
        print(f"       Status: {status}  Recall={r5:.1f}  RR={rr:.3f}  ({elapsed:.1f}s)")
        print(f"       Levels: {level_summary}")
        print(f"       Top-5 docs:")
        for j, t in enumerate(top5_titles, 1):
            print(f"          [{j}] {t}")
        print()

        results_log.append({
            "note": note,
            "query": query,
            "hit": hit,
            "rank": rank,
            "recall@5": r5,
            "rr": rr,
            "elapsed_s": round(elapsed, 1),
        })

    # ─── Summary ─────────────────────────────────────────────────────────────
    recall5 = total_recall / n
    mrr5 = total_rr / n

    print(f"\n{'='*70}")
    print(f"RESULTS ({n} queries evaluated)")
    print(f"{'='*70}")
    print(f"  Recall@5 : {recall5:.3f}  ({total_recall:.0f}/{n})")
    print(f"  MRR@5    : {mrr5:.3f}")
    print(f"{'='*70}\n")

    # Missed queries
    misses = [r for r in results_log if not r.get("hit")]
    if misses:
        print("MISSED queries:")
        for m in misses:
            print(f"  - [{m['note']}] {m['query'][:70]}")

    # Save results to JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "recall@5": round(recall5, 4),
            "mrr@5": round(mrr5, 4),
            "n_queries": n,
            "n_hits": int(total_recall),
            "details": results_log,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
