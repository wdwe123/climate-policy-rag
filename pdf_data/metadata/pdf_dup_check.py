import os
import hashlib
import pandas as pd
from tqdm import tqdm

# ==============================
# 配置
# ==============================
INPUT_PATH  = r"E:/2026_capstone/policy_data/pdf_data/metadata/policy_metadata_2_checked.csv"
OUTPUT_PATH = r"E:/2026_capstone/policy_data/pdf_data/metadata/policy_metadata_2_dedup.csv"

PATH_COLUMN = "file_path"     # 存pdf路径的列
VALID_COL   = "pdf_valid"     # 你之前检测生成的列（没有也没关系）
HASH_COL    = "pdf_sha256"
DUP_COL     = "is_duplicate"
KEEP_COL    = "kept_path"     # 重复的指向保留文件
DO_DELETE   = False           # True 才会实际删磁盘文件（默认建议 False）
MIN_SIZE    = 1024            # 小于这个大小直接当无效/不算hash

# ==============================
# 工具函数
# ==============================
def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str | None:
    """返回文件sha256；文件不存在/太小/读失败返回 None"""
    try:
        if not isinstance(path, str) or not path.strip():
            return None
        path = path.strip()
        if not os.path.exists(path):
            return None
        if os.path.getsize(path) < MIN_SIZE:
            return None

        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None

# ==============================
# 读入
# ==============================
df = pd.read_csv(INPUT_PATH)

if PATH_COLUMN not in df.columns:
    raise ValueError(f"Missing column: {PATH_COLUMN}. Found: {list(df.columns)}")

# ==============================
# 1) 先按 path 去重（完全相同路径）
# ==============================
# 标记重复行（相同file_path的后续行）
df["_dup_path"] = df.duplicated(subset=[PATH_COLUMN], keep="first")

# 对 path 重复行：直接标记为 duplicate（不需要算hash）
df[DUP_COL] = df["_dup_path"].astype(int)
df[KEEP_COL] = df[PATH_COLUMN]  # 先默认自己

# 给 path 重复的行：kept_path 指向第一次出现的那条 path
first_path_idx = {}
for i, p in enumerate(df[PATH_COLUMN].astype(str).tolist()):
    if p not in first_path_idx:
        first_path_idx[p] = i

kept_paths = []
for i, p in enumerate(df[PATH_COLUMN].astype(str).tolist()):
    kept_paths.append(df.loc[first_path_idx[p], PATH_COLUMN])
df[KEEP_COL] = kept_paths

# ==============================
# 2) 内容去重：对“非 path 重复且有效的pdf”算 hash
# ==============================
need_hash_mask = ~df["_dup_path"]

# 如果有 pdf_valid 列，只对 pdf_valid==1 的算hash（更快）
if VALID_COL in df.columns:
    need_hash_mask = need_hash_mask & (df[VALID_COL] == 1)

df[HASH_COL] = None

paths_to_hash = df.loc[need_hash_mask, PATH_COLUMN].tolist()
hashes = []
for p in tqdm(paths_to_hash, desc="Hashing PDFs (sha256)"):
    hashes.append(sha256_file(p))

df.loc[need_hash_mask, HASH_COL] = hashes

# 对 hash 为空的（坏文件/不存在）不参与内容去重
hash_mask = need_hash_mask & df[HASH_COL].notna()

# ==============================
# 3) 找到内容重复：sha256 相同
# ==============================
# duplicated keep='first' 表示第一条保留
df["_dup_hash"] = False
df.loc[hash_mask, "_dup_hash"] = df.loc[hash_mask].duplicated(subset=[HASH_COL], keep="first")

# 更新 duplicate 标记（path重复 或 hash重复 都算）
df[DUP_COL] = ((df["_dup_path"]) | (df["_dup_hash"])).astype(int)

# 计算每条 hash 的“保留文件路径”
hash_to_kept_path = {}
for idx in df.index[hash_mask]:
    h = df.at[idx, HASH_COL]
    p = df.at[idx, PATH_COLUMN]
    if h not in hash_to_kept_path and isinstance(h, str):
        hash_to_kept_path[h] = p

# 对 hash重复的行，kept_path 指向第一个出现的路径
for idx in df.index[df["_dup_hash"]]:
    h = df.at[idx, HASH_COL]
    if isinstance(h, str) and h in hash_to_kept_path:
        df.at[idx, KEEP_COL] = hash_to_kept_path[h]

# ==============================
# 4) 可选：删除磁盘上的重复文件（只删“内容重复且不是保留路径”的文件）
# ==============================
deleted = 0
delete_errors = 0

if DO_DELETE:
    # 只删 hash 重复的（内容重复）；path重复的其实是同一个文件，不需要删
    for idx in df.index[df["_dup_hash"]]:
        p = df.at[idx, PATH_COLUMN]
        kept = df.at[idx, KEEP_COL]
        if isinstance(p, str) and isinstance(kept, str) and p != kept and os.path.exists(p):
            try:
                os.remove(p)
                deleted += 1
            except Exception:
                delete_errors += 1

# ==============================
# 5) 同步到 CSV：建议把重复行标记为无效，避免后续OCR再跑
# ==============================
if VALID_COL in df.columns:
    df.loc[df[DUP_COL] == 1, VALID_COL] = 0  # 让后续 pipeline 跳过重复

# 清理临时列
df.drop(columns=["_dup_path", "_dup_hash"], inplace=True, errors="ignore")

df.to_csv(OUTPUT_PATH, index=False)

print("Done.")
print("Saved:", OUTPUT_PATH)
print("Duplicates:", int(df[DUP_COL].sum()))
if DO_DELETE:
    print("Deleted files:", deleted, "Delete errors:", delete_errors)