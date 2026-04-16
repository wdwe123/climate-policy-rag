import os
import ast
import pandas as pd
from collections import OrderedDict

# =========================
# 配置
# =========================
IN_CSV  = r"E:/2026_capstone/policy_data/pdf_data/metadata/policy_metadata_2_dedup.csv"     # 你的去重结果CSV（含 is_duplicate, kept_path）
OUT_CSV = r"E:/2026_capstone/policy_data/pdf_data/metadata/policy_metadata_2_merged.csv"   # 输出合并后的CSV

# 这些列会做“并集去重合并”
GEO_COLS = ["state_list", "county_list", "city_list", "tribe_list"]

# 分组键：优先用 pdf_sha256（内容级），否则用 kept_path（你已有）
GROUP_KEY_CANDIDATES = ["pdf_sha256", "kept_path"]

# 是否真的删除磁盘上的重复文件（建议先 False 跑一遍看结果）
DO_DELETE_FILES = False

# 仅删除“重复行”的 file_path 对应文件，且 file_path != kept_path
# =========================


def parse_maybe_list(x):
    """把 '["A","B"]' / "['A']" / 'A, B' / NaN 统一解析成 list[str]"""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    # 尝试解析 python list 字符串
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(i).strip() for i in v if str(i).strip()]
        except Exception:
            pass
    # 兜底：按逗号分隔
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def unique_preserve_order(items):
    """去重但保留首次出现顺序"""
    od = OrderedDict()
    for it in items:
        it = str(it).strip()
        if it:
            od[it] = True
    return list(od.keys())


def join_commas(items):
    """按你要求：用 ', ' 连接，且不重复"""
    return ", ".join(unique_preserve_order(items))


# 读入
df = pd.read_csv(IN_CSV)

# 找分组键
group_key = None
for cand in GROUP_KEY_CANDIDATES:
    if cand in df.columns and df[cand].notna().any():
        group_key = cand
        break
if group_key is None:
    raise ValueError(f"找不到分组键：需要至少存在 {GROUP_KEY_CANDIDATES} 之一")

# 确保有 is_duplicate / kept_path
if "is_duplicate" not in df.columns or "kept_path" not in df.columns:
    raise ValueError("需要列 is_duplicate 和 kept_path（你之前 dedup 脚本应已生成）")

# 解析地理列为 list
for c in GEO_COLS:
    if c in df.columns:
        df[c + "__list"] = df[c].apply(parse_maybe_list)
    else:
        df[c + "__list"] = [[] for _ in range(len(df))]

# 合并：按 group_key 把重复内容的多行合并为一行
rows = []
for key, g in df.groupby(group_key, dropna=False, sort=False):
    # 选保留行：优先 is_duplicate==0 的那行；否则取第一行
    kept = g[g["is_duplicate"] == 0]
    base = kept.iloc[0] if len(kept) else g.iloc[0]

    out = base.copy()

    # 合并地理列：并集去重
    for c in GEO_COLS:
        all_items = []
        for lst in g[c + "__list"].tolist():
            all_items.extend(lst)
        # 你要求：state_list 这种不要重复 Oklahoma；county_list 每行不同用逗号隔开
        out[c] = join_commas(all_items)

    # 额外：给你一个统计，看看这个合并组到底合并了多少行
    out["merged_from_rows"] = len(g)

    rows.append(out)

merged = pd.DataFrame(rows)

# 删除中间列
drop_cols = [c + "__list" for c in GEO_COLS if c + "__list" in merged.columns]
merged.drop(columns=drop_cols, inplace=True, errors="ignore")

# 可选：删除磁盘重复文件（只删重复行对应的 file_path，且不是 kept_path）
deleted = 0
if DO_DELETE_FILES:
    if "file_path" not in df.columns:
        raise ValueError("要删文件需要 file_path 列")
    for _, r in df[df["is_duplicate"] == 1].iterrows():
        fp = str(r.get("file_path", "")).strip()
        kp = str(r.get("kept_path", "")).strip()
        if fp and kp and fp != kp and os.path.exists(fp):
            try:
                os.remove(fp)
                deleted += 1
            except Exception:
                pass

# 保存
merged.to_csv(OUT_CSV, index=False)

print("✅ Done")
print("Group key:", group_key)
print("Input rows:", len(df))
print("Output rows:", len(merged))
if DO_DELETE_FILES:
    print("Deleted files:", deleted)