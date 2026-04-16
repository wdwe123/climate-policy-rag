import os
import pandas as pd
import pypdfium2 as pdfium
from tqdm import tqdm

# ==============================
# 配置
# ==============================

CSV_PATH = r"E:/2026_capstone/policy_data/pdf_data/metadata/policy_metadata_2_checked.csv"   # 你的csv路径
OUTPUT_PATH = r"E:/2026_capstone/policy_data/pdf_data/metadata/policy_metadata_2_checked.csv"
PATH_COLUMN = "file_path"  # 改成你csv里存pdf路径的列名

# ==============================
# 检测函数
# ==============================

def is_pdf_valid(path: str) -> int:
    """
    返回:
    1 -> 可以被 PDFium 正常打开
    0 -> 打不开 / 文件不存在 / 不是合法PDF
    """
    try:
        if not isinstance(path, str):
            return 0

        path = path.strip()

        if not os.path.exists(path):
            return 0

        if os.path.getsize(path) < 1024:  # 小于1KB基本不可能是正常PDF
            return 0

        # 检查PDF头
        with open(path, "rb") as f:
            header = f.read(5)
            if header != b"%PDF-":
                return 0

        # 用PDFium尝试打开
        pdf = pdfium.PdfDocument(path)
        page_count = len(pdf)
        pdf.close()

        if page_count == 0:
            return 0

        return 1

    except Exception:
        return 0


# ==============================
# 主程序
# ==============================

df = pd.read_csv(CSV_PATH)

if PATH_COLUMN not in df.columns:
    raise ValueError(f"Column '{PATH_COLUMN}' not found in CSV")

print("Checking PDF validity...")

df["pdf_valid"] = [
    is_pdf_valid(p) for p in tqdm(df[PATH_COLUMN])
]

df.to_csv(OUTPUT_PATH, index=False)

print("Done.")
print("Saved to:", OUTPUT_PATH)

print("\nSummary:")
print(df["pdf_valid"].value_counts())