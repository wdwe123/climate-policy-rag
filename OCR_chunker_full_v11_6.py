import os, re, json, csv, ast
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import statistics

import pypdfium2 as pdfium
from PIL import Image
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from tqdm import tqdm

from paddleocr import PaddleOCR

# --- LLM (Portkey) ---
try:
    from portkey_ai import Portkey  # type: ignore
except Exception:
    Portkey = None  # type: ignore



# =============================================================================
# Config
# =============================================================================
DPI = 200
SAVE_PAGE_IMAGES = True
SAVE_PAGE_DEBUG_JSONL = True
MAX_PAGES_PER_PDF = None
LANG = "en"
USE_ANGLE_CLS = False
USE_GPU = True

# =============================================================================
# LLM via Portkey (used ONLY for smarter headings + reducing fragmentation)
# =============================================================================
USE_LLM = True

# Portkey settings
PORTKEY_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
PORTKEY_API_KEY_ENV = "PORTKEY_API_KEY"   # put your key in env var with this name
PORTKEY_MODEL = os.getenv("PORTKEY_MODEL", "@gpt-5-mini/gpt-5-mini")

# Cost control (per PDF)
MAX_LLM_CALLS_PER_PDF = 200        # hard cap
MAX_LLM_CALLS_PER_PAGE = 1        # merge/boundary at most 1 call per page
LLM_TIMEOUT_SEC = 60

# Trigger heuristics
LLM_MERGE_HEADINGS_MIN = 7        # if a page yields >= this many heading regions, consider LLM grouping
LLM_TINY_REGION_TOKENS = 140      # regions below this are considered "tiny"
LLM_TINY_REGIONS_RATIO = 0.55     # if >= this ratio of regions are tiny, consider grouping
LLM_BOUNDARY_PROBE = True         # allow LLM to repair missed section boundaries
LLM_BOUNDARY_MAX_LINES = 32       # send only first N lines for boundary check


# If True, write a separate .md file for each detected table page for debugging/QA
EXPORT_TABLE_DEBUG_FILES = False
TABLE_DEBUG_DIR = None  # will be set after OUTPUT_DIR

# If True, include bracketed page/section prefix inside text. For RAG, keep this False.
INCLUDE_PAGE_PREFIX_IN_TEXT = False

TARGET_TOKENS = 1200
OVERLAP_TOKENS = 120



# Post-processing: merge overly small chunks produced by aggressive heading splits.
# Recommended for RAG: keep semantic boundaries but avoid tiny embeddings.
MERGE_SMALL_CHUNKS = True
MIN_CHUNK_TOKENS = 250          # below this, try to merge forward/backward
MERGE_TARGET_TOKENS = 650       # keep merging until we reach this target (if possible)
MERGE_MAX_TOKENS = 1100         # never merge beyond this size

# Appendix trigger: only treat "APPENDIX" as appendix after this percentage of pages
APPENDIX_TRIGGER_PCT = 0.35

OUTPUT_DIR = "E:/2026_capstone/policy_data/chunking_output"
TABLE_DEBUG_DIR = os.path.join(OUTPUT_DIR, 'tables_debug_v11_6')
IMAGE_DIR = os.path.join(OUTPUT_DIR, "page_images")
FINAL_JSONL_PATH = os.path.join(OUTPUT_DIR, "final_chunks_v11_6.jsonl")
PAGE_DEBUG_JSONL_PATH = os.path.join(OUTPUT_DIR, "page_chunks_debug_v11_6.jsonl")

METADATA_CSV_PATH = "E:/2026_capstone/policy_data/pdf_data/metadata/policy_metadata_4.csv"
DIAGNOSTICS_CSV_PATH = os.path.join(OUTPUT_DIR, "chunking_diagnostics_v11_6.csv")


# =============================================================================
# Helpers
# =============================================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize_spaces(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

# For RAG: merge OCR line-breaks into paragraph-level breaks.
# Keeps bullet/numbered list items on their own lines, but otherwise joins single line breaks with spaces.
BULLET_RE = re.compile(r"^\s*([•·\-–—]|\d+[\.\)]|[A-Za-z]\)|\([A-Za-z0-9]+\))\s+")

def normalize_linebreaks_for_rag(text: str) -> str:
    if not text:
        return text
    lines = [ln.rstrip() for ln in text.splitlines()]
    out = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            out.append(buf.strip())
        buf = ""

    for ln in lines:
        if not ln.strip():
            flush()
            out.append("")
            continue
        if BULLET_RE.match(ln):
            flush()
            out.append(ln.strip())
            continue
        if not buf:
            buf = ln.strip()
        else:
            buf += " " + ln.strip()

    flush()
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s[:120].strip("_")


def export_table_debug(doc_title: str, pdf_path: str, page_idx_1based: int, table_text: str):
    """Write a detected table block to a standalone markdown file for QA/debug.

    This is optional and mainly for checking table detection quality.
    Remove or disable by setting EXPORT_TABLE_DEBUG_FILES=False.
    """
    if not EXPORT_TABLE_DEBUG_FILES:
        return
    try:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        fn = safe_filename(f"{base}__p{page_idx_1based:04d}__table.md")
        fp = os.path.join(TABLE_DEBUG_DIR, fn)
        with open(fp, 'w', encoding='utf-8') as f:
            f.write((table_text or '').strip() + '\n')
    except Exception:
        pass
def estimate_tokens_rough(text: str) -> int:
    # Very rough tokens approximation: count "word-ish" pieces
    parts = re.split(r"[^A-Za-z0-9]+", text.strip())
    parts = [p for p in parts if p]
    return len(parts)


# =============================================================================
# LLM helpers (Portkey)
# =============================================================================
def _extract_text_from_portkey_completion(resp) -> str:
    """Best-effort extraction across Portkey/OpenAI-like response shapes."""
    if resp is None:
        return ""
    # dict-like
    try:
        if isinstance(resp, dict):
            ch = resp.get("choices") or []
            if ch:
                msg = ch[0].get("message") or {}
                return (msg.get("content") or "").strip()
    except Exception:
        pass
    # object-like
    try:
        ch = getattr(resp, "choices", None)
        if ch:
            msg = getattr(ch[0], "message", None)
            if msg is not None:
                return (getattr(msg, "content", "") or "").strip()
            # some SDKs: choices[0].text
            return (getattr(ch[0], "text", "") or "").strip()
    except Exception:
        pass
    return str(resp).strip()

def _extract_json_obj(s: str) -> Optional[Dict[str, Any]]:
    """Parse the first JSON object found in a string."""
    if not s:
        return None
    s = s.strip()
    # direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # find first {...}
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

def get_portkey_client():
    if not USE_LLM:
        return None
    if Portkey is None:
        raise RuntimeError("portkey_ai is not installed. Please: pip install portkey-ai")
    key = os.getenv(PORTKEY_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"Missing Portkey API key env var: {PORTKEY_API_KEY_ENV}")
    return Portkey(base_url=PORTKEY_BASE_URL, api_key=key)

def llm_call_json(client, system: str, user: str) -> Optional[Dict[str, Any]]:
    if (not USE_LLM) or client is None:
        return None
    resp = client.chat.completions.create(
        model=PORTKEY_MODEL,
        messages=[
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ],
        timeout=LLM_TIMEOUT_SEC,
    )
    txt = _extract_text_from_portkey_completion(resp)
    return _extract_json_obj(txt)

def llm_normalize_heading(client, raw_heading: str, prev_path: List[str]) -> Optional[Dict[str, Any]]:
    sys = """You normalize document section headings.
Return ONLY JSON with keys:
- canonical: string (clean, human-readable)
- level: integer 1-3 (1=top, 2=subsection, 3=subsubsection)
- drop: boolean (true only if this is a running header/footer or obvious noise)
Rules:
- Remove leading 'None:' or similar artifacts.
- Keep meaningful numbering (e.g., '2.1', 'SECTION 3') when present.
- Do NOT invent content not present in the heading.
"""
    user = f"""Previous section path: {prev_path}
Raw heading: {raw_heading}
Output JSON only."""
    return llm_call_json(client, sys, user)

def llm_boundary_repair(client, prev_path: List[str], page_start_lines: List[str], heading_candidates: List[str]) -> Optional[Dict[str, Any]]:
    sys = """You detect whether a new major section starts on this page and propose the best section heading.
Return ONLY JSON with keys:
- new_section: boolean
- canonical: string (only if new_section=true)
- level: integer 1-3 (only if new_section=true)
Guidelines:
- Prefer true only when there is strong evidence of a new section heading.
- If the page continues the previous section, return new_section=false.
- Do NOT use page numbers or running headers as headings.
"""
    user = """Previous section path:
{prev}

Heading candidates detected (may be empty):
{cands}

Page start (OCR lines, in reading order):
{lines}

Output JSON only.""".format(
        prev=prev_path,
        cands=heading_candidates,
        lines="\n".join(page_start_lines[:LLM_BOUNDARY_MAX_LINES]),
    )
    return llm_call_json(client, sys, user)

def llm_group_dense_headings(client, prev_path: List[str], regions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ask LLM to group consecutive tiny heading-regions into larger semantic groups."""
    sys = """You help reduce over-fragmentation caused by many tiny subheadings on one page.
You will receive a list of regions in order. Each region has:
- id (int)
- heading (string)
- tokens (int)
- preview (string) (first 1-2 lines)
Task:
Group CONSECUTIVE regions into a smaller number of groups suitable for RAG chunks.
Return ONLY JSON:
{
  "groups": [
     {"region_ids": [0,1,2], "section_name": "...", "level": 1|2|3},
     ...
  ]
}
Rules:
- Every region id must appear exactly once, in order.
- Prefer 2-4 groups if there are many regions.
- If headings are minor (tiny content), it is OK to group under a parent-like section_name.
- Do NOT invent content beyond headings; section_name should be derived from headings.
"""
    items = []
    for r in regions:
        items.append({
            "id": r.get("id"),
            "heading": r.get("heading") or "",
            "tokens": int(r.get("tokens") or 0),
            "preview": r.get("preview") or "",
        })
    user = f"Previous section path: {prev_path}\nRegions JSON:\n{json.dumps(items, ensure_ascii=False)}\nOutput JSON only."
    return llm_call_json(client, sys, user)

# =============================================================================

def split_prefix_and_body(text: str) -> Tuple[str, str]:
    """Strip legacy bracketed prefix if present.

    Older versions prepended a bracketed line like:
    [DOC | Section: ... | Page: ...]
    For RAG we keep text clean, but we keep this to be backward compatible.
    """
    if not text:
        return '', ''
    t = text.strip()
    if t.startswith('['):
        first, *rest = t.splitlines()
        if ('Section:' in first) and ('Page:' in first) and first.endswith(']'):
            return first, '\n'.join(rest).lstrip('\n')
    return '', t

def _norm_path(p: str) -> str:
    """Normalize paths for robust matching between CSV file_path and runtime pdf_path."""
    if not p:
        return ""
    # Normalize both Windows and POSIX separators to '/'
    p2 = str(p).replace("\\", "/").replace("\\", "/")
    p2 = os.path.normpath(p2).replace("\\", "/").replace("\\", "/")
    return p2.lower()

def parse_list_cell(val) -> List[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    s = str(val).strip()
    if not s or s in {"[]", "nan", "None"}:
        return []
    try:
        out = ast.literal_eval(s)
        if isinstance(out, list):
            return [str(x) for x in out if str(x).strip()]
    except Exception:
        pass
    # fallback: split by comma
    return [t.strip() for t in s.split(",") if t.strip()]

def load_policy_metadata(csv_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load policy_metadata_2.csv and build two lookups:
    - by_full_path: normalized full file_path -> metadata dict
    - by_basename: lower(basename) -> metadata dict (first seen) for fallback matching
    """
    if not csv_path or (not os.path.exists(csv_path)):
        print(f"[WARN] METADATA_CSV_PATH not found: {csv_path}")
        return {}, {}

    df = pd.read_csv(csv_path)
    by_full: Dict[str, Dict[str, Any]] = {}
    by_base: Dict[str, Dict[str, Any]] = {}

    for _, r in df.iterrows():
        fp = ""
        for _c in ["file_path","pdf_path","local_path","path","filepath","pdf_file_path"]:
            _v = r.get(_c, "")
            if isinstance(_v, float) and np.isnan(_v):
                _v = ""
            _v = str(_v or "").strip()
            if _v:
                fp = _v
                break
        if not fp:
            continue
        k = _norm_path(fp)
        meta = {
            "policy_id": str(r.get("policy_id", "") or "").strip() or None,
            "source_url": str(r.get("source_url", "") or "").strip() or None,
            "policy_level": str(r.get("policy_level", "") or "").strip() or None,
            "policy_type": str(r.get("policy_type", "") or "").strip() or None,
            "state_list": parse_list_cell(r.get("state_list")),
            "county_list": parse_list_cell(r.get("county_list")),
            "city_list": parse_list_cell(r.get("city_list")),
            "tribe_list": parse_list_cell(r.get("tribe_list")),
        }
        by_full[k] = meta
        base = os.path.basename(fp).lower()
        if base and base not in by_base:
            by_base[base] = meta

    return by_full, by_base

def lookup_policy_meta(pdf_path: str, by_full: Dict[str, Dict[str, Any]], by_base: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    k = _norm_path(pdf_path)
    if k in by_full:
        return by_full[k]
    base = os.path.basename(pdf_path).lower()
    if base in by_base:
        return by_base[base]
    return {}


# =============================================================================
# OCR init
# =============================================================================
def init_paddleocr() -> PaddleOCR:
    """Initialize PaddleOCR with version-robust arguments.

    Different PaddleOCR/PaddlePaddle builds accept different init kwargs.
    We try a few common variants and fall back to CPU if needed.
    """
    common = dict(
        use_angle_cls=USE_ANGLE_CLS,
        lang=LANG,
        show_log=False,
        drop_score=0.80,
    )

    candidates = [
        {**common, "use_gpu": USE_GPU},          # older API
        {**common, "device": "gpu" if USE_GPU else "cpu"},  # some newer builds
        {**common},                               # let backend decide / CPU fallback
    ]

    last_err = None
    for kw in candidates:
        try:
            return PaddleOCR(**kw)
        except Exception as e:
            last_err = e
            continue
    raise last_err



# =============================================================================
# Render PDF page to image
# =============================================================================
def render_pdf_page_to_pil(pdf_path: str, page_idx: int, dpi: int) -> Image.Image:
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf.get_page(page_idx)
    scale = dpi / 72
    bitmap = page.render(scale=scale)
    pil = bitmap.to_pil()
    page.close()
    pdf.close()
    return pil


# =============================================================================
# OCR
# =============================================================================
def merge_ocr_items_into_lines(ocr_items, y_threshold=12) -> List[str]:
    """
    将OCR返回的文本框（带坐标）按垂直位置合并为自然行。
    ocr_items: list of (box, text, score)
    返回按阅读顺序排列的行文本列表。
    """
    if not ocr_items:
        return []

    # 按垂直位置排序（按最小y）
    items_sorted = sorted(ocr_items, key=lambda it: min(p[1] for p in it[0]))

    lines = []  # each element: [(y, x, text), ...]
    for box, text, score in items_sorted:
        text = normalize_spaces(text)
        if not text:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        y = float(min(ys))
        x = float(min(xs))

        placed = False
        for line in lines:
            if abs(line[0][0] - y) <= y_threshold:
                line.append((y, x, text))
                placed = True
                break
        if not placed:
            lines.append([(y, x, text)])

    out_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda t: t[1])
        merged = " ".join([t[2] for t in line_sorted])
        out_lines.append(normalize_spaces(merged))

    return [ln for ln in out_lines if ln]




def merge_ocr_items_into_lines_with_pos(ocr_items, y_threshold=12,
                                         col_gap_min_px: float = 60.0) -> List[Dict[str, float]]:
    """Group OCR boxes into visual lines, with multi-column awareness.

    Returns list of dicts: {text, y_min, y_max, x_min, x_max}

    Multi-column handling
    ---------------------
    Within a y-band (same "line" bucket), items that are separated by a
    horizontal gap >= col_gap_min_px are considered to belong to different
    columns.  Each column segment is emitted as a *separate* line object,
    preventing left- and right-column text from being concatenated into
    garbled prose.

    After splitting, all output lines are sorted by (y_min, x_min) so that
    within the same horizontal band left-column content precedes right-column
    content, while full vertical ordering is preserved across bands.
    """
    if not ocr_items:
        return []

    # Store (x_min, x_max, y_min, y_max, text) per item
    items_sorted = sorted(ocr_items, key=lambda it: min(p[1] for p in it[0]))

    lines = []
    for box, text, score in items_sorted:
        text = normalize_spaces(text)
        if not text:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        y_min = float(min(ys)); y_max = float(max(ys))
        x_min = float(min(xs)); x_max = float(max(xs))
        y_ref = y_min

        placed = False
        for ln in lines:
            if abs(ln['y_ref'] - y_ref) <= y_threshold:
                ln['items'].append((x_min, x_max, y_min, y_max, text))
                ln['y_min'] = min(ln['y_min'], y_min)
                ln['y_max'] = max(ln['y_max'], y_max)
                ln['x_min'] = min(ln['x_min'], x_min)
                ln['x_max'] = max(ln['x_max'], x_max)
                placed = True
                break
        if not placed:
            lines.append({
                'y_ref': y_ref,
                'y_min': y_min,
                'y_max': y_max,
                'x_min': x_min,
                'x_max': x_max,
                'items': [(x_min, x_max, y_min, y_max, text)],
            })

    out = []
    for ln in lines:
        # Sort items in this y-band left-to-right
        items_in_band = sorted(ln['items'], key=lambda t: t[0])

        if len(items_in_band) < 2:
            # Single item — emit directly
            it = items_in_band[0]
            if it[4]:
                out.append({'text': it[4], 'y_min': ln['y_min'],
                            'y_max': ln['y_max'], 'x_min': it[0], 'x_max': it[1]})
            continue

        # Detect column gaps: find the largest horizontal gap between consecutive items
        # (gap = x_min of next item - x_max of current item)
        segments = []   # list of (segment_items,)
        cur_seg = [items_in_band[0]]
        for i in range(1, len(items_in_band)):
            prev_x_max = items_in_band[i-1][1]
            cur_x_min  = items_in_band[i][0]
            gap = cur_x_min - prev_x_max
            if gap >= col_gap_min_px:
                segments.append(cur_seg)
                cur_seg = [items_in_band[i]]
            else:
                cur_seg.append(items_in_band[i])
        segments.append(cur_seg)

        for seg in segments:
            merged = normalize_spaces(' '.join(t[4] for t in seg))
            if merged:
                out.append({
                    'text':  merged,
                    'y_min': ln['y_min'],
                    'y_max': ln['y_max'],
                    'x_min': float(seg[0][0]),
                    'x_max': float(seg[-1][1]),
                })

    # Sort by (y_min, x_min): preserves top-to-bottom order; within the same
    # horizontal band left column precedes right column.
    out.sort(key=lambda d: (d['y_min'], d['x_min']))
    return out

def ocr_image_full(ocr: PaddleOCR, pil_img: Image.Image):
    """
    Returns:
      ocr_items: list of (box, text, score)
      img_w, img_h
    """
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    result = ocr.ocr(img, cls=USE_ANGLE_CLS)
    ocr_items = []

    if result and result[0]:
        for box, (text, score) in result[0]:
            text = normalize_spaces(text)
            if text:
                ocr_items.append((box, text, float(score)))

    return ocr_items, int(w), int(h)


# =============================================================================
# Structure patterns
# =============================================================================
HEADING_NUMERIC = re.compile(r"^\s*(\d+(?:\.\d+){0,6})\s*[:\.\-]?\s+(.+?)\s*$")
HEADING_WORDY = re.compile(
    r"^\s*(CHAPTER|SECTION|PART)\s+([IVXLC\d]+)\b[:\.\-]?\s*(.*)\s*$",
    re.IGNORECASE
)

APPENDIX_RE = re.compile(r"^\s*(APPENDIX|ANNEX|EXHIBIT|ATTACHMENT)\b", re.IGNORECASE)

# TOC signals
TOC_KEYWORDS = re.compile(r"(TABLE OF CONTENTS?|CONTENTS|DOCUMENT OUTLINE|INDEX|LIST OF FIGURES|LIST OF TABLES)", re.IGNORECASE)
TOC_LINE_WITH_DOTS = re.compile(r"\.{2,}\s*\d+\s*$")
TOC_LINE_ENDNUM = re.compile(r".*\s(\d{1,4})\s*$")
TOC_ENTRY_LIKE = re.compile(
    r"^\s*((SECTION|CHAPTER|PART)\s+\w+|\d+(\.\d+){0,6}|[IVXLC]+)\b.*\s\d{1,4}\s*$",
    re.IGNORECASE
)

# Figure caption signals
FIGURE_CAPTION_RE = re.compile(r"^\s*(FIGURE|TABLE|MAP|EXHIBIT)\b", re.IGNORECASE)


# Form / questionnaire signals
FORM_KEYWORDS = re.compile(r"\b(FORM|QUESTIONNAIRE|SURVEY|CHECKLIST|APPLICATION|REPORT FORM|INSPECTION FORM)\b", re.IGNORECASE)
FORM_FIELD_LINE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 \-/\(\),]{1,60}:\s*(?:_{2,}|\.+|—+)?\s*$")
FORM_UNDERSCORE = re.compile(r"_{3,}|—{3,}|\.{3,}")
FORM_CHECKBOX = re.compile(r"(\[\s*\]|\[\]|☐|□|◻|▢|\(\s*\)|\(\)|YES\s*/\s*NO|YES\s+NO)", re.IGNORECASE)


# =============================================================================
# Page type detection (no ML)
# =============================================================================
def extract_heading_spans(
    line_objs: List[Dict[str, float]],
    page_h: float,
    page_w: float,
    header_blacklist: Optional[set] = None,
    table_y0: Optional[float] = None,
    table_y1: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Extract multiple heading spans across a page with conservative filters.

    Key behaviors (to fix your issues):
    - Numeric / SECTION headings can be detected anywhere on the page (handles 6.1 / 6.2 near mid/bottom).
    - Unnumbered title-like headings are ONLY allowed in the top band (avoids list items / citations being misread as headings).
    - Running headers are excluded via header_blacklist.
    - Table bands are excluded; plus "table-row-like" lines are rejected (dates/money/high digit density).
    - If too many headings are found on a page, we fall back to numeric/SECTION headings only.
    """
    if not line_objs or page_h <= 0:
        return []

    header_blacklist = header_blacklist or set()
    # Pre-compute vertical gaps to allow mid-page unnumbered headings when they are visually separated.
    # This helps capture subheadings like "Purpose" that may appear mid-page and continue across pages.
    sorted_objs = sorted([(i, o) for i, o in enumerate(line_objs)], key=lambda t: (float(t[1].get("y_min", 0.0)), float(t[1].get("x_min", 0.0))))
    gaps_above: Dict[int, float] = {}
    gaps_below: Dict[int, float] = {}
    for j, (i, o) in enumerate(sorted_objs):
        y_min = float(o.get("y_min", 0.0))
        y_max = float(o.get("y_max", y_min))
        prev_y_max = float(sorted_objs[j-1][1].get("y_max", sorted_objs[j-1][1].get("y_min", 0.0))) if j > 0 else None
        next_y_min = float(sorted_objs[j+1][1].get("y_min", 0.0)) if j < len(sorted_objs)-1 else None
        gaps_above[i] = (y_min - prev_y_max) if prev_y_max is not None else 0.0
        gaps_below[i] = (next_y_min - y_max) if next_y_min is not None else 0.0

    TOP_TITLE_FRAC = 0.35      # unnumbered headings only in top 35%
    FOOTER_FRAC = 0.08         # ignore bottom 8% for headings (page number, footers)

    LEFT_MARGIN_FRAC = 0.10    # headings should usually start near the left margin
    SINGLE_LEVEL_TOP_FRAC = 0.30  # single-level headings like '4.' must be near top

    footer_y = (1.0 - FOOTER_FRAC) * page_h
    top_title_y = TOP_TITLE_FRAC * page_h

    # Single-word headings are common across domains (e.g., "Introduction", "Overview").
    # Only allow them in the top band and require centered-ish layout to avoid false positives.
    SINGLE_WORD_HEADINGS = {
        "introduction","overview","background","purpose","scope","summary","methods","methodology",
        "findings","results","discussion","conclusion","recommendations","references","appendix","annex",
        "glossary","definitions","acronyms","abbreviations"
    }

    def single_word_title_like(s: str, obj: Dict[str, float]) -> bool:
        s2 = normalize_spaces(s)
        if not s2 or len(s2) < 4 or len(s2) > 32:
            return False
        if not re.fullmatch(r"[A-Za-z][A-Za-z\-']+", s2):
            return False
        if s2.lower() not in SINGLE_WORD_HEADINGS:
            return False
        y_min = float(obj.get("y_min", 0.0))
        if y_min > top_title_y:
            return False
        # centered-ish: heading text block should be around the page center
        if page_w > 0:
            x_min = float(obj.get("x_min", 0.0))
            x_max = float(obj.get("x_max", x_min))
            cx = 0.5 * (x_min + x_max)
            if not (0.35 * page_w <= cx <= 0.65 * page_w):
                return False
        return True


    def overlaps_table(ymin: float, ymax: float) -> bool:
        if table_y0 is None or table_y1 is None:
            return False
        return not (ymax < table_y0 or ymin > table_y1)

    def digit_ratio(s: str) -> float:
        s2 = re.sub(r"\s+", "", s)
        if not s2:
            return 0.0
        return sum(ch.isdigit() for ch in s2) / len(s2)

    def looks_like_table_row(s: str) -> bool:
        # strong reject: typical table row signals
        if '$' in s:
            return True
        if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", s):
            return True
        if len(re.findall(r"\d+", s)) >= 4:
            return True
        if digit_ratio(s) >= 0.30:
            return True
        if re.search(r"\b\$?\d{1,3}(,\d{3})*(\.\d+)?\b", s) and digit_ratio(s) > 0.22:
            return True
        return False

    def looks_like_running_header(s: str) -> bool:
        s_norm = normalize_spaces(s).lower()
        return bool(s_norm) and (s_norm in header_blacklist)

    def looks_like_citation_or_list_item(s: str) -> bool:
        # Avoid "AZ Division..., 2013, ..." or sources lists becoming headings
        if "http://" in s.lower() or "https://" in s.lower():
            return True
        if s.count(",") >= 2:
            return True
        if ";" in s:
            return True
        if re.search(r"\b(19|20)\d{2}\b", s) and s.count(",") >= 1:
            return True
        return False

    # Stop words that should never be the last word in a heading
    _HEADING_END_STOP = {
        'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'with', 'by',
        'for', 'from', 'that', 'which', 'include', 'includes', 'are', 'is',
        'was', 'were', 'has', 'have', 'be', 'been', 'as', 'at', 'on',
    }
    # First words that indicate a sentence rather than a heading (applied for >6-word phrases)
    _SENTENCE_STARTERS = {
        'the', 'a', 'an', 'it', 'in', 'as', 'when', 'while', 'although',
        'since', 'because', 'this', 'these', 'those', 'there', 'by', 'for',
        'implementing', 'following', 'based', 'due', 'such', 'each', 'under',
    }

    def title_like(s: str) -> bool:
        # Conservative: mostly letters, title-ish case, no heavy punctuation
        if len(s) < 4 or len(s) > 120:
            return False
        if s.endswith("."):
            return False
        if ":" in s:
            colon_idx = s.index(":")
            before = s[:colon_idx].strip()
            _LABEL_PREFIX = re.compile(
                r"^(Criterion|Goal|Objective|Priority|Action|Strategy|Step|Phase|Task|Item|Policy)\s*#?\s*\w*$",
                re.IGNORECASE,
            )
            if _LABEL_PREFIX.match(before):
                # "Goal SES1: ...", "Criterion #5: ..." → confirmed labeled heading
                return True
            # ALL-CAPS "TITLE: SUBTITLE" format
            # e.g. "THE PATH FORWARD: GREENHOUSE GAS EMISSIONS REDUCTIONS GOALS"
            #      "EXECUTIVE SUMMARY: Key Findings"
            before_words = before.split()
            if (1 <= len(before_words) <= 5
                    and before.replace(" ", "").replace("-", "").replace("'", "").isupper()
                    and len(before) >= 3):
                # Let normal word-count / end-stop checks run on the full string,
                # but bypass the sentence-starter filter (all-caps titles can start with "THE")
                tokens_all = s.split()
                n_all = len(tokens_all)
                if n_all < 2 or n_all > 12:
                    return False
                last_tok = tokens_all[-1].rstrip(".,;:!?").lower()
                if last_tok in _HEADING_END_STOP:
                    return False
                return True
            # Colon with non-label, non-all-caps prefix → reject
            return False
        if looks_like_citation_or_list_item(s):
            return False
        if TOC_LINE_WITH_DOTS.search(s):
            return False
        if digit_ratio(s) >= 0.18:
            return False

        # Footnote references appended to end of line (e.g. "Iron and Steel Production8")
        if s[-1].isdigit():
            return False

        tokens = s.split()
        n = len(tokens)
        if n < 2 or n > 12:
            return False

        # Prose sentences ending with a stop word are not headings
        last_tok = tokens[-1].rstrip(".,;:!?").lower()
        if last_tok in _HEADING_END_STOP:
            return False

        # For longer phrases (>6 words), reject sentence-like openers
        if n > 6 and tokens[0].lower() in _SENTENCE_STARTERS:
            return False

        # All caps: short phrases OK (e.g. "EXECUTIVE SUMMARY")
        if s.isupper():
            return n <= 8

        # Title-case-ish: at least 70% of tokens capitalized.
        # This allows 2-3 lowercase prepositions/conjunctions in a longer heading.
        caps = sum(1 for w in tokens if w[:1].isupper())
        return (caps / max(1, n)) >= 0.70

    cands: List[Dict[str, Any]] = []

    for _i, obj in enumerate(line_objs):
        obj["_line_index"] = _i
        ln = obj["text"].strip()
        if not ln:
            continue

        # ignore footer zone entirely
        if obj["y_min"] >= footer_y:
            continue

        # Exclude table band + running headers + obvious noise
        if overlaps_table(obj["y_min"], obj["y_max"]):
            continue
        if looks_like_running_header(ln):
            continue
        if looks_like_table_row(ln):
            continue
        if len(ln) > 140:
            continue

        m1 = HEADING_WORDY.match(ln)
        m2 = HEADING_NUMERIC.match(ln)

        score = 0.0
        hid = None
        title = None
        kind = None

        if m1:
            kind = "wordy"
            hid = f"{m1.group(1).upper()} {m1.group(2)}"
            title = normalize_spaces(m1.group(3) or "")
            score = 2.2
            # Validate title for CHAPTER/SECTION/PART headings when a title is present.
            # Prevents "SECTION 33: 83 of City Code provides..." from becoming a heading.
            if title:
                tok_n = len(title.split())
                if tok_n > 2 and not title_like(title):
                    kind = None
        elif m2:
            kind = "numeric"
            hid = m2.group(1)
            title = normalize_spaces(m2.group(2) or "")
            score = 2.0

            if re.fullmatch(r"\d+", hid or ""):
                # Single-level numeric: must be near top-left and title-like
                x_min = float(obj.get("x_min", 0.0))
                y_min = float(obj.get("y_min", 0.0))
                near_left = (page_w > 0) and (x_min <= (LEFT_MARGIN_FRAC * page_w))
                near_top = (y_min <= (SINGLE_LEVEL_TOP_FRAC * page_h))
                tl = title_like(title) if title else False
                tok_n = len(title.split()) if title else 0
                if not (near_left and near_top and tl and (2 <= tok_n <= 10)):
                    kind = None
            elif re.search(r"\.", hid or ""):
                # Multi-level numeric (e.g. "1.7", "4.2.4"): apply title_like to prevent
                # body text like "1.7 million customers..." from becoming a heading.
                if title:
                    tok_n = len(title.split())
                    if tok_n > 2 and not title_like(title):
                        kind = None
        else:
            # Unnumbered title-like headings:
            # - Allowed in top band, OR
            # - Allowed mid-page when visually separated (whitespace above & below).
            y_min = float(obj.get("y_min", 0.0))
            i0 = int(obj.get("_line_index", -1))
            gap_above = gaps_above.get(i0, 0.0)
            gap_below = gaps_below.get(i0, 0.0)
            standalone = (gap_above >= (0.012 * page_h)) and (gap_below >= (0.010 * page_h))
            if (y_min <= top_title_y or standalone) and (title_like(ln) or single_word_title_like(ln, obj)):
                kind = "title"
                hid = None
                title = ln
                score = 1.4

        if kind is None:
            continue

        # extra rejection for "Sources" etc becoming headings unless numeric/wordy
        if kind == "title":
            low = ln.lower()
            if low in {"sources", "source", "references", "reference"}:
                # keep as title heading only if it's very near top (like a real section break)
                if obj["y_min"] > (0.22 * page_h):
                    continue

        cands.append({
            "hid": hid,
            "title": title,
            "text": (f"{hid}: {title}" if hid and title else (hid or title or ln)),
            "y_min": float(obj["y_min"]),
            "y_max": float(obj["y_max"]),
            "x_min": float(obj.get("x_min", 0.0)),
            "x_max": float(obj.get("x_max", 0.0)),
            "score": float(score),
            "kind": kind,
        })

    if not cands:
        return []

    # Sort and de-dup near-duplicates
    cands.sort(key=lambda d: (d["y_min"], -d["score"]))
    dedup = []
    seen = set()
    for c in cands:
        key = (normalize_spaces((c["hid"] or "") + " " + (c["title"] or "")).lower(), int(c["y_min"] // 6))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    cands = dedup

    # If too many headings found, keep only strong ones (numeric/wordy) to avoid fragmentation
    if len(cands) > 6:
        strong = [c for c in cands if c["kind"] in ("numeric", "wordy")]
        if strong:
            cands = strong
        # still too many? keep top 4 by score/y
        cands = cands[:4]

    return cands

def _heading_level_from_obj(h: Dict[str, Any], page_w: float) -> int:
    """Infer heading level (1/2/3) from heading object in a format-agnostic way."""
    kind = (h.get("kind") or "").lower()
    hid = (h.get("hid") or "")
    x_min = float(h.get("x_min", 0.0))
    x_max = float(h.get("x_max", 0.0))
    center = (x_min + x_max) / 2.0 if page_w > 0 else 0.0
    centered = (page_w > 0) and (abs(center - (page_w / 2.0)) <= (0.14 * page_w))

    if kind in ("numeric", "wordy") and hid:
        dot_n = hid.count(".")
        return min(3, 1 + dot_n)

    if kind == "title":
        if centered:
            return 1
        if x_min <= 0.20 * max(1.0, page_w):
            return 2
        return 2

    return 2


def section_label(h: Dict[str, Any]) -> str:
    """Create a stable, human-readable label for a heading object.

    Kept global because both page-splitting and cross-page outline tracking
    require the same labeling logic.
    """
    hid = (h.get("hid") or "").strip()
    title = (h.get("title") or "").strip()
    if hid and title:
        return f"{hid}: {title}"
    if hid:
        return hid
    return (h.get("text") or "").strip() or "(no heading)"


def _update_outline_stack(outline: List[str], heading_text: str, level: int) -> List[str]:
    """Update outline stack to reflect new heading at given level."""
    heading_text = normalize_spaces(heading_text or "")
    if not heading_text:
        return outline
    level = max(1, min(3, int(level)))
    if len(outline) >= level:
        outline = outline[: level - 1]
    while len(outline) < level - 1:
        outline.append("")
    outline.append(heading_text)
    outline = [s for s in outline if s]
    return outline

def split_page_into_heading_regions(
    line_objs: List[Dict[str, float]],
    headings: List[Dict[str, Any]],
    page_h: float,
    header_blacklist: Optional[set] = None,
    table_y0: Optional[float] = None,
    table_y1: Optional[float] = None,
    table_block_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Split a single page into multiple regions by heading y positions.

    Returns a list of regions:
      [
        {
          "section": "<section label>",
          "y_start": float,
          "y_end": float,
          "lines": [str, ...]   # region text lines (table band removed; table_block_text injected)
        },
        ...
      ]

    Key behaviors:
    - Heading line itself is NOT included in region body (prevents duplicate headings inside text).
    - Running headers in header_blacklist are excluded.
    - If a table band exists (table_y0-table_y1), lines in that band are removed and
      table_block_text is injected once at the correct vertical position inside the region.
    """

    header_blacklist = header_blacklist or set()

    if not line_objs or page_h <= 0:
        return []

    # Sort lines and headings by y
    lines_sorted = sorted(line_objs, key=lambda o: o["y_min"])
    heads_sorted = sorted(headings, key=lambda h: h["y_min"])

    # Helper: make a stable section label
    def section_label(h: Dict[str, Any]) -> str:
        hid = (h.get("hid") or "").strip()
        title = (h.get("title") or "").strip()
        if hid and title:
            return f"{hid}: {title}"
        if hid:
            return hid
        # fallback to raw heading text
        return (h.get("text") or "").strip() or "(no heading)"

    # Helper: check running header
    def is_running_header_line(s: str) -> bool:
        return normalize_spaces(s).lower() in header_blacklist

    # Helper: check overlaps table band
    def overlaps_table(ymin: float, ymax: float) -> bool:
        if table_y0 is None or table_y1 is None:
            return False
        return not (ymax < table_y0 or ymin > table_y1)

    # Build region boundaries based on headings
    boundaries: List[Tuple[float, float, Dict[str, Any]]] = []
    for i, h in enumerate(heads_sorted):
        y_start = float(h["y_min"])
        y_end = float(heads_sorted[i + 1]["y_min"]) if i + 1 < len(heads_sorted) else float(page_h)
        boundaries.append((y_start, y_end, h))

    regions: List[Dict[str, Any]] = []

    for (y_start, y_end, h) in boundaries:
        sec = section_label(h)
        reg_lines: List[str] = []

        # Track where to inject table block (once)
        injected_table = False

        # Collect all lines whose y overlaps [y_start, y_end)
        for obj in lines_sorted:
            ymid = 0.5 * (float(obj["y_min"]) + float(obj["y_max"]))

            if ymid < y_start:
                continue
            if ymid >= y_end:
                break

            txt = normalize_spaces(obj["text"])
            if not txt:
                continue

            # Skip running header lines
            if is_running_header_line(txt):
                continue

            # Skip the heading line itself (avoid duplicates)
            # Use y overlap + exact/near match
            if abs(float(obj["y_min"]) - float(h["y_min"])) <= 2.0:
                if txt == normalize_spaces(h.get("text", "")):
                    continue

            # Remove table band lines and inject the reconstructed table once
            if overlaps_table(float(obj["y_min"]), float(obj["y_max"])):
                if (not injected_table) and table_block_text:
                    # Inject at the first time we hit the table band inside this region
                    reg_lines.append(table_block_text)
                    injected_table = True
                continue

            reg_lines.append(txt)

        # If table exists but the table band is entirely within this region AND we never hit a line
        # inside the band (rare OCR ordering edge), inject at end of region
        if (not injected_table) and table_block_text and (table_y0 is not None and table_y1 is not None):
            # If region overlaps table area
            if not (y_end < table_y0 or y_start > table_y1):
                reg_lines.append(table_block_text)
                injected_table = True

        # Drop empty/noise regions (but keep if it has table block)
        clean = [ln for ln in reg_lines if ln and ln.strip()]
        if not clean:
            continue

        regions.append(
            {
                "section": sec,
                "section_heading": (h if isinstance(h, dict) else None),
                "y_start": float(y_start),
                "y_end": float(y_end),
                "lines": clean,
            }
        )

    return regions

def detect_appendix_on_page(lines: List[str]) -> Tuple[bool, Optional[str], bool]:
    """Detect appendix transition on a page.

    Returns:
      (trigger, label, strong)

    strong=True when the page looks like an appendix section divider heading (early lines, short, standalone).
    Cross-domain; does not rely on page position.
    """
    if not lines:
        return False, None, False

    # Guard: if page has ≥2 lines that start with "Annex/Appendix X:" it is a TOC listing,
    # NOT an actual appendix section page.  Avoid false early-appendix triggers from TOC pages.
    _ANNEX_ENTRY_RE = re.compile(
        r"^\s*(APPENDIX|ANNEX|EXHIBIT|ATTACHMENT)\s*[A-Z0-9]+\s*:", re.IGNORECASE
    )
    toc_annex_count = sum(1 for ln in lines if _ANNEX_ENTRY_RE.match((ln or "").strip()))
    if toc_annex_count >= 2:
        return False, None, False

    head = [ln.strip() for ln in lines if (ln or '').strip()][:12]
    for i, ln in enumerate(head):
        m = re.match(r"^\s*(APPENDIX|ANNEX|EXHIBIT|ATTACHMENT)\s*([A-Z0-9]+)?\b", ln, re.IGNORECASE)
        if m:
            label = f"{m.group(1).upper()} {m.group(2).upper()}" if m.group(2) else m.group(1).upper()
            strong = (i <= 3 and len(ln) <= 55)
            return True, label, strong

    for ln in [x.strip() for x in lines[:40] if (x or '').strip()]:
        if re.match(r"^(APPENDIX|ANNEX|EXHIBIT|ATTACHMENT)\b", ln, re.IGNORECASE):
            m = re.match(r"^\s*(APPENDIX|ANNEX|EXHIBIT|ATTACHMENT)\s*([A-Z0-9]+)?\b", ln, re.IGNORECASE)
            if m:
                label = f"{m.group(1).upper()} {m.group(2).upper()}" if m.group(2) else m.group(1).upper()
                return True, label, False
            return True, "APPENDIX", False

    return False, None, False



def toc_scores(lines: List[str]) -> Dict[str, float]:
    """Return normalized scores for TOC-likeness based on cross-domain patterns.

    We avoid document-specific rules by combining several weak signals:
    - explicit keywords ("Table of Contents", "Contents", "Index", ...)
    - dot leaders + trailing page numbers
    - line endings that look like page numbers
    - entry-like prefixes (section numbers, roman numerals, "Chapter", etc.)
    - short/fragmentary line style common in TOCs
    """
    if not lines:
        return {
            "kw": 0.0, "dots_ratio": 0.0, "endnum_ratio": 0.0, "entry_ratio": 0.0,
            "short_ratio": 0.0, "toc_score": 0.0
        }

    # Keep only non-trivial lines for scoring
    lines2 = [ln.strip() for ln in lines if (ln or "").strip()]
    if not lines2:
        return {
            "kw": 0.0, "dots_ratio": 0.0, "endnum_ratio": 0.0, "entry_ratio": 0.0,
            "short_ratio": 0.0, "toc_score": 0.0
        }

    text_join = " ".join(lines2)
    kw = 1.0 if TOC_KEYWORDS.search(text_join) else 0.0

    n = max(1, len(lines2))
    dots = sum(1 for ln in lines2 if TOC_LINE_WITH_DOTS.search(ln))
    endnum = sum(1 for ln in lines2 if TOC_LINE_ENDNUM.match(ln) and len(ln.strip()) >= 8)
    entry = sum(1 for ln in lines2 if TOC_ENTRY_LIKE.match(ln))

    # Many TOCs have lots of short-ish lines (titles are fragments)
    short = sum(1 for ln in lines2 if 8 <= len(ln) <= 70)

    dots_ratio = dots / n
    endnum_ratio = endnum / n
    entry_ratio = entry / n
    short_ratio = short / n

    score = (
        1.2 * kw
        + 1.0 * min(1.0, dots_ratio / 0.12)
        + 0.9 * min(1.0, endnum_ratio / 0.35)
        + 0.9 * min(1.0, entry_ratio / 0.30)
        + 0.6 * min(1.0, short_ratio / 0.65)
    )

    return {
        "kw": float(kw),
        "dots_ratio": float(dots_ratio),
        "endnum_ratio": float(endnum_ratio),
        "entry_ratio": float(entry_ratio),
        "short_ratio": float(short_ratio),
        "toc_score": float(score),
    }


def is_toc_like_page_strong(lines: List[str], prev_was_toc: bool = False) -> Tuple[bool, Dict[str, float]]:
    """High-precision TOC detection, with a *generic* continuation rule.

    Continuation rule (for pages after the first TOC page):
    - If the previous page is already classified as TOC, we relax the need for
      explicit keywords and rely more on TOC-style line structure.
    This helps with multi-page TOCs where only the first page says 'Contents'.
    """
    s = toc_scores(lines)

    # Hard triggers (keyword present + some structure)
    if s["kw"] >= 1.0 and (s["endnum_ratio"] >= 0.12 or s["entry_ratio"] >= 0.10 or s["dots_ratio"] >= 0.05):
        return True, s

    # Continuation trigger (previous page was TOC)
    if prev_was_toc:
        # More tolerant but still requires strong structural evidence
        if (s["endnum_ratio"] >= 0.25 and s["short_ratio"] >= 0.55) or (s["dots_ratio"] >= 0.06 and s["entry_ratio"] >= 0.12):
            s["toc_score"] = max(float(s["toc_score"]), 1.7)
            return True, s

    # Soft trigger
    if s["toc_score"] >= 1.65:
        return True, s

    return False, s

def _box_area(box) -> float:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def compute_ocr_text_cover(ocr_items, img_w: int, img_h: int) -> float:
    if not ocr_items or img_w <= 0 or img_h <= 0:
        return 0.0
    text_area = sum(_box_area(box) for box, _, _ in ocr_items)
    return float(text_area / (img_w * img_h))


def compute_page_ink_ratio(pil_img: Image.Image) -> float:
    """
    Cheap ink proxy: downsample + grayscale + count non-white pixels.
    Figure/scan pages tend to have higher ink ratio even if OCR text cover is low.
    """
    g = pil_img.convert("L").resize((300, 300))
    arr = np.array(g)
    ink = (arr < 245).mean()
    return float(ink)


def detect_table_grid_lines(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Image-based table detector using line/grid evidence (OpenCV).
    Works well for Word-exported tables with borders (like your examples).

    Returns dict:
      found: bool
      bbox: (x0,y0,x1,y1) in pixel coords (if found)
      debug: metrics
    """
    if cv2 is None:
        return {"found": False, "debug": {"reason": "cv2_not_available"}}

    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # binarize (invert so lines/text are white)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )

    # remove small noise
    thr = cv2.medianBlur(thr, 3)

    # Extract horizontal and vertical line candidates
    # kernel sizes are relative to page size (robust across DPI)
    h_ksz = max(10, w // 35)
    v_ksz = max(10, h // 35)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_ksz, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_ksz))

    h_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Strengthen lines a bit
    h_lines = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)
    v_lines = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)

    grid = cv2.bitwise_or(h_lines, v_lines)
    inter = cv2.bitwise_and(h_lines, v_lines)

    total = float(w * h)
    h_ratio = float(np.count_nonzero(h_lines)) / total
    v_ratio = float(np.count_nonzero(v_lines)) / total
    grid_ratio = float(np.count_nonzero(grid)) / total
    inter_cnt = int(np.count_nonzero(inter))

    # Bounding box of grid pixels
    ys, xs = np.where(grid > 0)
    if len(xs) == 0:
        return {"found": False, "debug": {"h_ratio": h_ratio, "v_ratio": v_ratio, "grid_ratio": grid_ratio, "inter_cnt": inter_cnt, "reason": "no_grid_pixels"}}

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    bbox_area_frac = float(bw * bh) / total

    # Heuristic thresholds (tuned for bordered tables; keep conservative to avoid false positives)
    # - grid pixels should not be extremely sparse
    # - intersections indicate "cell grid" rather than just underlines
    found = (
        (grid_ratio > 0.0025 and inter_cnt > 120 and bbox_area_frac > 0.08)
        or (grid_ratio > 0.0040 and inter_cnt > 60 and bbox_area_frac > 0.06)
    )

    return {
        "found": bool(found),
        "bbox": (x0, y0, x1, y1),
        "debug": {
            "h_ratio": h_ratio,
            "v_ratio": v_ratio,
            "grid_ratio": grid_ratio,
            "inter_cnt": inter_cnt,
            "bbox_area_frac": bbox_area_frac,
            "w": w,
            "h": h,
        },
    }


def extract_captions(lines: List[str], max_lines: int = 8) -> List[str]:
    caps = []
    for ln in lines:
        if FIGURE_CAPTION_RE.search(ln.strip()):
            caps.append(normalize_spaces(ln))
            if len(caps) >= max_lines:
                break
    return caps



def form_scores(lines: List[str]) -> Dict[str, float]:
    """
    Detect survey/questionnaire/form pages using cross-domain textual patterns:
    - Many field lines: "Date: ____", "Time: ____", "Address:" etc.
    - Many underscores / long blank lines
    - Many checkbox-like tokens
    - Strong FORM/QUESTIONNAIRE keywords
    """
    if not lines:
        return {"kw": 0.0, "field_ratio": 0.0, "blank_ratio": 0.0, "checkbox_ratio": 0.0, "form_score": 0.0}

    n = max(1, len(lines))
    joined = " ".join(lines)

    kw = 1.0 if FORM_KEYWORDS.search(joined) else 0.0
    field = sum(1 for ln in lines if FORM_FIELD_LINE.match(ln))
    blankish = sum(1 for ln in lines if FORM_UNDERSCORE.search(ln))
    checkbox = sum(1 for ln in lines if FORM_CHECKBOX.search(ln))

    field_ratio = field / n
    blank_ratio = blankish / n
    checkbox_ratio = checkbox / n

    # conservative score; we want high precision
    score = (
        1.0 * kw
        + 0.9 * min(1.0, field_ratio / 0.18)
        + 0.7 * min(1.0, blank_ratio / 0.22)
        + 0.7 * min(1.0, checkbox_ratio / 0.18)
    )
    return {
        "kw": float(kw),
        "field_ratio": float(field_ratio),
        "blank_ratio": float(blank_ratio),
        "checkbox_ratio": float(checkbox_ratio),
        "form_score": float(score),
    }


def is_form_like_page_strong(lines: List[str]) -> Tuple[bool, Dict[str, float]]:
    s = form_scores(lines)

    # Hard triggers: explicit keyword + some structure
    if s["kw"] >= 1.0 and (s["field_ratio"] >= 0.08 or s["checkbox_ratio"] >= 0.06 or s["blank_ratio"] >= 0.10):
        return True, s

    # Soft trigger: high overall score
    if s["form_score"] >= 1.55:
        return True, s

    return False, s


def extract_form_header(lines: List[str], max_lines: int = 12) -> List[str]:
    """
    Keep only the header/title-ish lines of a form, to avoid polluting chunks.
    Strategy: keep first N lines that are NOT mostly blanks/underscores,
    and also keep any line with FORM_KEYWORDS.
    """
    kept: List[str] = []
    for ln in lines[:max_lines * 2]:
        ln2 = normalize_spaces(ln)
        if not ln2:
            continue
        if FORM_UNDERSCORE.search(ln2) and len(ln2) > 10:
            # skip pure blanks
            continue
        if FORM_FIELD_LINE.match(ln2):
            # skip field-only lines
            continue
        # keep keyword lines and other header lines
        if FORM_KEYWORDS.search(ln2) or len(kept) < max_lines:
            kept.append(ln2)
        if len(kept) >= max_lines:
            break
    return kept

def is_figure_like_strong(ocr_items, img_w: int, img_h: int, ink_ratio: float) -> Tuple[bool, Dict[str, float]]:
    """
    Multi-signal figure-like detection.

    v5 -> v6 fix:
    Some "image pages" (maps/figures) still contain plenty of OCR text (labels),
    so OCR cover may not be low. We add a secondary path:
      - high ink ratio
      - many short OCR tokens / low median token length
      - modest OCR cover (not full-page paragraphs)
    """
    debug: Dict[str, float] = {
        "ocr_items": float(len(ocr_items) if ocr_items else 0),
        "ocr_cover": 0.0,
        "short_frac": 1.0,
        "median_len": 0.0,
        "ink_ratio": float(ink_ratio),
        "fig_score": 0.0,
        "alt_path": 0.0,
    }

    if not ocr_items:
        debug["fig_score"] = 1.8 if ink_ratio > 0.10 else 1.0
        return True, debug

    cover = compute_ocr_text_cover(ocr_items, img_w, img_h)
    debug["ocr_cover"] = float(cover)

    lengths = [len(t.strip()) for _, t, _ in ocr_items if t and t.strip()]
    if not lengths:
        debug["fig_score"] = 1.8 if ink_ratio > 0.10 else 1.0
        return True, debug

    lengths_sorted = sorted(lengths)
    median_len = float(lengths_sorted[len(lengths_sorted) // 2])
    short_frac = float(sum(1 for L in lengths if L <= 12) / len(lengths))

    debug["short_frac"] = short_frac
    debug["median_len"] = median_len

    fig_score = 0.0
    # classic path: low OCR cover + high ink
    if cover <= 0.05:
        fig_score += 1.0
    if cover <= 0.03:
        fig_score += 0.6
    if ink_ratio >= 0.18:
        fig_score += 0.6
    if short_frac >= 0.65:
        fig_score += 0.5
    if median_len <= 18:
        fig_score += 0.3

    # NEW: map/figure pages with labels (moderate OCR cover but still figure-like)
    alt = 0.0
    if ink_ratio >= 0.28 and 0.05 < cover <= 0.22:
        alt += 1.0
    if ink_ratio >= 0.33 and cover <= 0.28:
        alt += 0.6
    if short_frac >= 0.75:
        alt += 0.5
    if median_len <= 10:
        alt += 0.4
    if len(ocr_items) >= 220 and median_len <= 12:
        alt += 0.4

    debug["alt_path"] = float(alt)
    debug["fig_score"] = float(max(fig_score, alt))

    return (max(fig_score, alt) >= 1.6), debug



# =============================================================================
# Chunking structures
# =============================================================================
@dataclass
class PageChunk:
    doc_title: str
    section: str
    pages: str                 # "start - end"
    is_appendix: bool
    appendix_label: Optional[str]
    tokens: int
    text: str

    # hierarchical headings
    section_path: Optional[List[str]] = None

    # optional debug / trace
    pdf_path: Optional[str] = None
    page_image_path: Optional[str] = None

    # Page classification
    page_type: Optional[str] = None   # "text" | "toc" | "figure" | "form" | "uncertain"
    is_toc: Optional[bool] = None
    is_figure: Optional[bool] = None
    is_form: Optional[bool] = None
    is_table: Optional[bool] = None
    is_cover: Optional[bool] = None
    page_debug: Optional[Dict[str, Any]] = None

    # Document-level policy metadata (from policy_metadata_2.csv)
    policy_id: Optional[str] = None
    source_url: Optional[str] = None
    policy_level: Optional[str] = None
    policy_type: Optional[str] = None

    # Geo metadata (lists)
    state_list: Optional[List[str]] = None
    county_list: Optional[List[str]] = None
    city_list: Optional[List[str]] = None
    tribe_list: Optional[List[str]] = None




# =============================================================================
# Chunk post-processing helpers
# =============================================================================

def _parse_pages_range(pages: str) -> Tuple[int, int]:
    """
    Parse pages string like "10 - 11" or "10-11" or "10".
    Returns (start, end) as ints.
    """
    s = (pages or "").strip()
    if not s:
        return (0, 0)
    m = re.match(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$", s.replace("–", "-"))
    if not m:
        # fallback: try to extract the first/last integers
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if not nums:
            return (0, 0)
        return (min(nums), max(nums))
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    return (min(a, b), max(a, b))


def _format_pages_range(a: int, b: int) -> str:
    return f"{a} - {b}" if a != b else f"{a}"


def _is_mergeable_text_chunk(ch: PageChunk) -> bool:
    # Never merge non-text structural pages/chunks
    if ch.page_type in {"toc", "figure", "form"}:
        return False
    if ch.is_toc or ch.is_figure or ch.is_form or ch.is_table or ch.is_cover:
        return False
    # Keep very short / empty chunks mergeable
    return True


def _section_path_tuple(ch: PageChunk) -> Tuple[str, ...]:
    if ch.section_path:
        return tuple([x for x in ch.section_path if (x or "").strip()])
    sec = (ch.section or "").strip()
    return (sec,) if sec else tuple()


def _parent_path_tuple(ch: PageChunk) -> Tuple[str, ...]:
    sp = _section_path_tuple(ch)
    return sp[:-1] if len(sp) > 1 else sp


def _compatible_for_merge(a: PageChunk, b: PageChunk) -> bool:
    # Must be from same document instance
    if (a.pdf_path and b.pdf_path and a.pdf_path != b.pdf_path):
        return False
    if a.doc_title != b.doc_title:
        return False
    if (a.policy_id or "") != (b.policy_id or ""):
        # metadata mismatch: avoid cross-doc merges
        return False

    # Appendix state should match
    if bool(a.is_appendix) != bool(b.is_appendix):
        return False

    # Prefer same exact path; otherwise allow same parent path (siblings)
    spa = _section_path_tuple(a)
    spb = _section_path_tuple(b)
    if spa and spb and spa == spb:
        return True
    if spa and spb and _parent_path_tuple(a) == _parent_path_tuple(b) and len(_parent_path_tuple(a)) >= 1:
        return True

    # Fallback: same flat section string
    if (a.section or "").strip() and (a.section or "").strip() == (b.section or "").strip():
        return True

    return False


def _merge_two_chunks(a: PageChunk, b: PageChunk) -> PageChunk:
    a0, a1 = _parse_pages_range(a.pages)
    b0, b1 = _parse_pages_range(b.pages)
    pages = _format_pages_range(min(a0, b0), max(a1, b1))

    merged_text = (a.text or "").rstrip() + "\n\n" + (b.text or "").lstrip()
    merged_tokens = estimate_tokens_rough(merged_text)

    spa = _section_path_tuple(a)
    spb = _section_path_tuple(b)
    new_sp: Optional[List[str]] = None
    if spa and spb:
        if spa == spb:
            new_sp = list(spa)
        elif _parent_path_tuple(a) == _parent_path_tuple(b) and len(_parent_path_tuple(a)) >= 1:
            new_sp = list(_parent_path_tuple(a))
        else:
            new_sp = list(spa)
    elif spa:
        new_sp = list(spa)
    elif spb:
        new_sp = list(spb)

    new_section = " > ".join(new_sp) if new_sp else (a.section or b.section or "")

    out = PageChunk(
        doc_title=a.doc_title,
        section=new_section,
        pages=pages,
        is_appendix=a.is_appendix,
        appendix_label=a.appendix_label or b.appendix_label,
        tokens=merged_tokens,
        text=merged_text,
        section_path=new_sp,
        pdf_path=a.pdf_path or b.pdf_path,
        page_image_path=a.page_image_path or b.page_image_path,
        page_type=a.page_type or b.page_type,
        is_toc=bool(a.is_toc or b.is_toc) if (a.is_toc is not None or b.is_toc is not None) else None,
        is_figure=bool(a.is_figure or b.is_figure) if (a.is_figure is not None or b.is_figure is not None) else None,
        is_form=bool(a.is_form or b.is_form) if (a.is_form is not None or b.is_form is not None) else None,
        is_table=bool(a.is_table or b.is_table) if (a.is_table is not None or b.is_table is not None) else None,
        is_cover=bool(a.is_cover or b.is_cover) if (a.is_cover is not None or b.is_cover is not None) else None,
        page_debug=a.page_debug or b.page_debug,
        policy_id=a.policy_id or b.policy_id,
        source_url=a.source_url or b.source_url,
        policy_level=a.policy_level or b.policy_level,
        policy_type=a.policy_type or b.policy_type,
        state_list=(a.state_list or b.state_list or []),
        county_list=(a.county_list or b.county_list or []),
        city_list=(a.city_list or b.city_list or []),
        tribe_list=(a.tribe_list or b.tribe_list or []),
    )
    return out


def merge_small_chunks(
    chunks: List[PageChunk],
    min_tokens: int = MIN_CHUNK_TOKENS,
    target_tokens: int = MERGE_TARGET_TOKENS,
    max_tokens: int = MERGE_MAX_TOKENS,
) -> List[PageChunk]:
    """
    Merge overly small adjacent chunks while preserving document structure.

    Strategy:
      - Never merge TOC/figure/form/table/cover chunks.
      - Prefer merging within the same exact section_path.
      - Otherwise allow merging sibling headings (same parent section_path).
      - Stop merging once we reach target_tokens or would exceed max_tokens.
    """
    if not chunks:
        return chunks

    out: List[PageChunk] = []
    i = 0

    def try_merge_forward(seed: PageChunk, j: int) -> Tuple[PageChunk, int]:
        cur = seed
        k = j
        while cur.tokens < target_tokens and k < len(chunks):
            nxt = chunks[k]
            if not _is_mergeable_text_chunk(nxt):
                break
            if not _compatible_for_merge(cur, nxt):
                break
            if cur.tokens + nxt.tokens > max_tokens:
                break
            cur = _merge_two_chunks(cur, nxt)
            k += 1
        return cur, k

    while i < len(chunks):
        ch = chunks[i]

        # If previous output chunk is tiny, try to absorb current chunk if compatible.
        if out and _is_mergeable_text_chunk(out[-1]) and out[-1].tokens < min_tokens and _is_mergeable_text_chunk(ch) and _compatible_for_merge(out[-1], ch):
            if out[-1].tokens + ch.tokens <= max_tokens:
                out[-1] = _merge_two_chunks(out[-1], ch)
                i += 1
                continue

        if (not MERGE_SMALL_CHUNKS) or (not _is_mergeable_text_chunk(ch)) or ch.tokens >= min_tokens:
            out.append(ch)
            i += 1
            continue

        # Small chunk: try merge forward greedily
        merged, next_i = try_merge_forward(ch, i + 1)
        out.append(merged)
        i = next_i

    return out




def format_print(chunk: PageChunk) -> str:
    header = [
        "---",
        f"doc_title: {chunk.doc_title}",
        f"section: {chunk.section}",
        f"pages: {chunk.pages}",
        f"is_appendix: {chunk.is_appendix} appendix_label: {chunk.appendix_label}",
        f"page_type: {chunk.page_type} is_toc: {chunk.is_toc} is_figure: {chunk.is_figure} is_form: {getattr(chunk, 'is_form', None)}",
        f"tokens: {chunk.tokens}",
        chunk.text,
    ]
    return "\n".join(header)


# =============================================================================
# Page-level pass (OCR + structure tagging)
# ==========================================
# =============================================================================
# Table detection & extraction (embedded tables inside normal text pages)
# =============================================================================

TABLE_CAPTION_RE = re.compile(r"^\s*(TABLE|TAB\.|EXHIBIT)\b", re.IGNORECASE)

def _box_bounds(box):
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    x0, x1 = float(min(xs)), float(max(xs))
    y0, y1 = float(min(ys)), float(max(ys))
    return x0, y0, x1, y1

def _prep_items_with_centers(ocr_items):
    """
    Convert PaddleOCR items -> enriched tuples:
    (box, text, conf, x0, y0, x1, y1, xc, yc)
    """
    items = []
    for box, txt, conf in (ocr_items or []):
        t = (txt or "").strip()
        if not t:
            continue
        x0, y0, x1, y1 = _box_bounds(box)
        xc = (x0 + x1) / 2.0
        yc = (y0 + y1) / 2.0
        items.append((box, t, float(conf) if conf is not None else 0.0, x0, y0, x1, y1, xc, yc))
    return items

def _cluster_sorted(values, tol):
    """Greedy clustering for 1D sorted values -> cluster centers."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]

def _assign_to_nearest_cluster(v, centers):
    if not centers:
        return None
    return min(range(len(centers)), key=lambda i: abs(v - centers[i]))

def _group_items_to_rows(items, y_tol=10.0):
    """Cluster items into rows by y-center; each row sorted by x-center."""
    if not items:
        return []
    items_sorted = sorted(items, key=lambda it: it[8])  # yc
    rows = []
    cur = [items_sorted[0]]
    cur_y = items_sorted[0][8]
    for it in items_sorted[1:]:
        if abs(it[8] - cur_y) <= y_tol:
            cur.append(it)
            cur_y = (cur_y * (len(cur) - 1) + it[8]) / len(cur)
        else:
            rows.append(sorted(cur, key=lambda x: x[7]))
            cur = [it]
            cur_y = it[8]
    rows.append(sorted(cur, key=lambda x: x[7]))
    return rows

def _row_column_centers(row_items, x_tol=35.0):
    xcs = [it[7] for it in row_items]
    return _cluster_sorted(xcs, tol=x_tol)

def detect_table_block_from_ocr(
    ocr_items,
    img_w: int,
    img_h: int,
    y_tol: float = 10.0,
    x_tol: float = 35.0,
    min_cols: int = 2,
    min_consecutive_rows: int = 4,
):
    """
    Detect the largest consecutive run of table-like rows.

    Table-like row heuristic:
      - >= min_cols column clusters
      - >= min_cols tokens in the row
    """
    items = _prep_items_with_centers(ocr_items)
    if not items:
        return {"found": False, "debug": {"reason": "no_items"}}

    rows = _group_items_to_rows(items, y_tol=y_tol)
    if len(rows) < min_consecutive_rows:
        return {"found": False, "debug": {"reason": "too_few_rows", "rows": len(rows)}}

    is_table_row = []
    row_cols = []
    for r in rows:
        centers = _row_column_centers(r, x_tol=x_tol)
        col_n = len(centers)
        row_cols.append(col_n)
        token_n = len(r)
        is_table_row.append((col_n >= min_cols) and (token_n >= min_cols))

    # longest consecutive run of True
    best_s, best_e, best_len = 0, -1, 0
    cur_s = None
    for i, flag in enumerate(is_table_row):
        if flag and cur_s is None:
            cur_s = i
        if (not flag or i == len(is_table_row) - 1) and cur_s is not None:
            cur_e = i if flag and i == len(is_table_row) - 1 else i - 1
            run_len = cur_e - cur_s + 1
            if run_len > best_len:
                best_s, best_e, best_len = cur_s, cur_e, run_len
            cur_s = None

    if best_len < min_consecutive_rows:
        return {"found": False, "debug": {"reason": "no_long_run", "best_len": best_len, "row_cols": row_cols[:50]}}

    # bounds
    y0 = min(it[4] for r in rows[best_s:best_e + 1] for it in r)
    y1 = max(it[6] for r in rows[best_s:best_e + 1] for it in r)
    x0 = min(it[3] for r in rows[best_s:best_e + 1] for it in r)
    x1 = max(it[5] for r in rows[best_s:best_e + 1] for it in r)

    height_frac = (y1 - y0) / max(1.0, float(img_h))
    width_frac = (x1 - x0) / max(1.0, float(img_w))

    # sanity guards to avoid misclassifying scattered text as table
    if height_frac < 0.08:
        return {"found": False, "debug": {"reason": "too_short_height", "height_frac": float(height_frac)}}
    if width_frac < 0.35:
        return {"found": False, "debug": {"reason": "too_narrow", "width_frac": float(width_frac)}}

    # Alignment guard (v11.2):
    # Icon grids / multi-column layouts can mimic "table rows" in OCR.
    # A real table usually has consistent column centers across the run.
    run_rows = rows[best_s:best_e + 1]
    centers_per_row = [_row_column_centers(r, x_tol=x_tol) for r in run_rows]
    counts = [len(c) for c in centers_per_row if len(c) >= min_cols]
    target_k = 0
    if counts:
        try:
            target_k = max(set(counts), key=counts.count)
        except Exception:
            target_k = max(counts)

    if target_k >= min_cols:
        kept = [c for c in centers_per_row if abs(len(c) - target_k) <= 1]
        if len(kept) >= min_consecutive_rows:
            aligned_lists = [sorted(c)[:target_k] for c in kept if len(c) >= target_k]
            if len(aligned_lists) >= min_consecutive_rows:
                target = []
                for j in range(target_k):
                    vals = [lst[j] for lst in aligned_lists]
                    target.append(float(statistics.median(vals)) if vals else None)

                devs = []
                for c in aligned_lists:
                    d = [abs(c[j] - target[j]) for j in range(target_k) if target[j] is not None]
                    if d:
                        devs.append(float(statistics.mean(d)))
                if devs:
                    mean_dev = float(statistics.mean(devs))
                    norm_dev = mean_dev / max(1.0, float(img_w))
                    if norm_dev > 0.08:
                        return {"found": False, "debug": {"reason": "weak_alignment", "norm_dev": norm_dev, "target_k": target_k}}
    return {
        "found": True,
        "y0": float(y0),
        "y1": float(y1),
        "x0": float(x0),
        "x1": float(x1),
        "start_row": int(best_s),
        "end_row": int(best_e),
        # Treat as a "table page" when the detected table dominates the page.
        # This helps avoid over-splitting and prevents figure/form false positives.
        "table_page": bool((height_frac >= 0.45) and (width_frac >= 0.60) and (best_len >= max(min_consecutive_rows, 6))),
        "debug": {
            "rows": len(rows),
            "run_len": int(best_len),
            "row_cols_sample": row_cols[max(0, best_s - 3):min(len(row_cols), best_e + 4)],
            "height_frac": float(height_frac),
            "width_frac": float(width_frac),
        },
    }



# =============================================================================
# Table cell extraction (OpenCV grid -> cells) for better table reconstruction
# =============================================================================

def _group_positions(pos_list, tol=6):
    """Group near-duplicate line positions."""
    if not pos_list:
        return []
    pos_list = sorted(pos_list)
    groups = [[pos_list[0]]]
    for p in pos_list[1:]:
        if abs(p - groups[-1][-1]) <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(round(sum(g)/len(g))) for g in groups]

def _find_peaks_1d(arr, min_val, min_dist=8):
    """Simple peak finder on 1D projection."""
    peaks=[]
    last=-10**9
    for i,v in enumerate(arr):
        if v>=min_val and (i-last)>=min_dist:
            peaks.append(i); last=i
    return peaks

def detect_table_cells_from_image(rgb_img: np.ndarray, bbox=None):
    """Detect table grid lines and approximate cell boxes from a rendered page image.

    Returns dict with keys:
      x_lines, y_lines: sorted line coordinates (image pixels)
      cells: list of (x0,y0,x1,y1) cell rectangles
      grid_score: diagnostic score
      used_bbox: (x0,y0,x1,y1)
    """
    if cv2 is None:
        return None

    h, w = rgb_img.shape[:2]
    if bbox is None:
        x0,y0,x1,y1 = 0,0,w,h
    else:
        x0,y0,x1,y1 = [int(max(0, v)) for v in bbox]
        x1 = min(w, x1); y1 = min(h, y1)
        if x1-x0 < 50 or y1-y0 < 50:
            x0,y0,x1,y1 = 0,0,w,h

    crop = rgb_img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # binarize (invert so lines/text are white)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 15)

    # remove small noise
    bw = cv2.medianBlur(bw, 3)

    # kernels relative to crop size
    ch, cw = bw.shape[:2]
    v_ks = max(10, cw//60)
    h_ks = max(10, ch//80)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_ks))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_ks, 1))

    # extract vertical lines
    v = cv2.erode(bw, vertical_kernel, iterations=1)
    v = cv2.dilate(v, vertical_kernel, iterations=2)

    # extract horizontal lines
    h_img = cv2.erode(bw, horizontal_kernel, iterations=1)
    h_img = cv2.dilate(h_img, horizontal_kernel, iterations=2)

    # line projections
    v_proj = (v > 0).sum(axis=0)  # per x
    h_proj = (h_img > 0).sum(axis=1)  # per y

    # dynamic thresholds (need substantial line length)
    v_thr = max(int(0.45*ch), 25)
    h_thr = max(int(0.45*cw), 25)

    x_peaks = _find_peaks_1d(v_proj, v_thr, min_dist=max(6, cw//120))
    y_peaks = _find_peaks_1d(h_proj, h_thr, min_dist=max(6, ch//160))

    x_lines = _group_positions(x_peaks, tol=max(4, cw//200))
    y_lines = _group_positions(y_peaks, tol=max(4, ch//200))

    # if too few lines, likely not a grid table
    if len(x_lines) < 3 or len(y_lines) < 3:
        return {
            "x_lines": [x0+v for v in x_lines],
            "y_lines": [y0+v for v in y_lines],
            "cells": [],
            "grid_score": float(len(x_lines)*len(y_lines)),
            "used_bbox": (x0,y0,x1,y1),
        }

    # build cells between consecutive lines
    cells=[]
    for yi in range(len(y_lines)-1):
        yy0, yy1 = y_lines[yi], y_lines[yi+1]
        if yy1-yy0 < 12:
            continue
        for xi in range(len(x_lines)-1):
            xx0, xx1 = x_lines[xi], x_lines[xi+1]
            if xx1-xx0 < 12:
                continue
            cells.append((x0+xx0, y0+yy0, x0+xx1, y0+yy1))

    # grid score: number of intersections approximated
    grid_score = float((len(x_lines)-1)*(len(y_lines)-1))

    return {
        "x_lines": [x0+v for v in x_lines],
        "y_lines": [y0+v for v in y_lines],
        "cells": cells,
        "grid_score": grid_score,
        "used_bbox": (x0,y0,x1,y1),
    }

def _assign_items_to_cells(ocr_items, cells):
    """Assign OCR items to cell boxes by center point."""
    if not cells:
        return {}

    # build spatial index by naive scan (tables usually not huge)
    cell_map = {i: [] for i in range(len(cells))}
    for box, txt, conf in ocr_items:
        if not txt or txt.strip()=="":
            continue
        xs=[p[0] for p in box]; ys=[p[1] for p in box]
        cx = (min(xs)+max(xs))/2.0
        cy = (min(ys)+max(ys))/2.0
        for i,(x0,y0,x1,y1) in enumerate(cells):
            if (x0 <= cx <= x1) and (y0 <= cy <= y1):
                cell_map[i].append((cy, cx, txt.strip()))
                break
    return cell_map

def _merge_cell_items_to_text(items, y_tol=10):
    """Merge OCR snippets inside one cell into multi-line text."""
    if not items:
        return ""
    items = sorted(items, key=lambda t: (t[0], t[1]))  # (y,x)
    lines=[]
    cur=[]
    cur_y=None
    for y,x,txt in items:
        if cur_y is None or abs(y-cur_y) <= y_tol:
            cur.append((x,txt))
            cur_y = y if cur_y is None else (0.7*cur_y+0.3*y)
        else:
            cur_sorted = [t for _,t in sorted(cur, key=lambda a:a[0])]
            lines.append(" ".join(cur_sorted).strip())
            cur=[(x,txt)]
            cur_y=y
    if cur:
        cur_sorted = [t for _,t in sorted(cur, key=lambda a:a[0])]
        lines.append(" ".join(cur_sorted).strip())
    # de-duplicate empty lines
    lines=[ln for ln in lines if ln]
    return "\n".join(lines).strip()

def _flatten_cell(text: str, max_len: int = 300) -> str:
    """Flatten a multi-line cell to a single '; '-joined line, safe for Markdown pipes."""
    lines = [ln.strip() for ln in (text or "").replace("\r", "\n").split("\n") if ln.strip()]
    clean = []
    for ln in lines:
        ln = re.sub(r"^[-•·]\s*", "", ln).strip()
        if ln:
            clean.append(ln)
    result = "; ".join(clean)
    result = result.replace("|", "╎")   # escape pipes so they don't break the table
    return result[:max_len].rstrip("; ").strip()


def _build_markdown_table(headers: List[str], data_rows: List[List[str]]) -> str:
    """Build a standard Markdown pipe table.

    Each cell is flattened to a single line with semicolons separating
    bullet items.  This format is well-understood by LLMs and is compact
    enough for embedding.
    """
    if not headers or not data_rows:
        return ""
    flat_headers = [_flatten_cell(h) or f"Col{i+1}" for i, h in enumerate(headers)]
    n = len(flat_headers)
    header_line = "| " + " | ".join(flat_headers) + " |"
    separator   = "| " + " | ".join("---" for _ in flat_headers) + " |"
    row_lines = []
    for row in data_rows:
        padded = list(row) + [""] * max(0, n - len(row))
        flat_cells = [_flatten_cell(padded[i] if i < len(padded) else "") for i in range(n)]
        row_lines.append("| " + " | ".join(flat_cells) + " |")
    return "\n".join([header_line, separator] + row_lines)


def reconstruct_table_structured_text_from_cells(ocr_items, rgb_img: np.ndarray, bbox=None) -> Tuple[str, Dict[str,Any]]:
    """Reconstruct RAG-friendly structured table text using OpenCV-derived cell boxes."""
    info={}
    cell_det = detect_table_cells_from_image(rgb_img, bbox=bbox)
    if not cell_det:
        return "", {"mode":"no_cv2"}
    cells = cell_det.get("cells", [])
    x_lines = cell_det.get("x_lines", [])
    y_lines = cell_det.get("y_lines", [])
    info.update({"mode":"cells", "grid_score": cell_det.get("grid_score"), "n_cells": len(cells), "n_x": len(x_lines), "n_y": len(y_lines)})

    if not cells:
        return "", info

    # infer grid dims
    n_cols = max(0, len(x_lines)-1)
    n_rows = max(0, len(y_lines)-1)
    if n_cols <= 1 or n_rows <= 1:
        return "", info

    # map cell index (r,c) -> rect
    # cells list is row-major by our construction
    # rebuild to 2D index
    rects = [[None]*n_cols for _ in range(n_rows)]
    idx=0
    for r in range(n_rows):
        for c in range(n_cols):
            if idx < len(cells):
                rects[r][c]=cells[idx]
            idx+=1

    # assign OCR items to cell rects using the flat cell list
    cell_map = _assign_items_to_cells(ocr_items, cells)

    # build cell texts 2D
    cell_texts=[[""]*n_cols for _ in range(n_rows)]
    idx=0
    for r in range(n_rows):
        for c in range(n_cols):
            items = cell_map.get(idx, [])
            cell_texts[r][c]=_merge_cell_items_to_text(items, y_tol=12)
            idx+=1

    # header row: choose first non-empty row among top 3 rows
    header_r = 0
    for r in range(min(3,n_rows)):
        non_empty = sum(1 for c in range(n_cols) if cell_texts[r][c].strip())
        if non_empty >= max(2, n_cols//2):
            header_r = r
            break

    headers=[]
    for c in range(n_cols):
        htxt = cell_texts[header_r][c].replace("\n"," ").strip()
        htxt = re.sub(r"\s+", " ", htxt)
        if not htxt:
            htxt = f"Col{c+1}"
        headers.append(htxt)

    # detect label column
    label_col = None
    if headers and re.search(r"\bdrought\s*stage\b", headers[0], re.I):
        label_col = 0
    else:
        # heuristic: first col is narrow and other headers longer
        def _safe_col_widths(rects, table_bbox, n_cols):
            """Return per-column widths robustly.

            Some PDFs/table detectors yield missing cells (None) in rect grids.
            We fallback to an equal-width partition of the table bbox.
            """
            widths = [None] * n_cols

            # Try to find a row with actual rects
            row = None
            if isinstance(rects, list):
                for rr in rects:
                    if not isinstance(rr, list):
                        continue
                    if any(isinstance(cell, (tuple, list)) and len(cell) == 4 for cell in rr):
                        row = rr
                        break

            if row is not None:
                for c in range(min(n_cols, len(row))):
                    cell = row[c]
                    if isinstance(cell, (tuple, list)) and len(cell) == 4:
                        x0, y0, x1, y1 = cell
                        try:
                            widths[c] = float(x1) - float(x0)
                        except Exception:
                            widths[c] = None

            # Fallback: equal partition from table bbox
            if any(w is None for w in widths):
                if isinstance(table_bbox, (tuple, list)) and len(table_bbox) == 4:
                    tx0, ty0, tx1, ty1 = table_bbox
                    try:
                        total_w = float(tx1) - float(tx0)
                        if total_w > 0 and n_cols > 0:
                            eq = total_w / float(n_cols)
                            widths = [w if (w is not None and w > 0) else eq for w in widths]
                    except Exception:
                        pass

            # Last resort: drop None
            return [w for w in widths if (w is not None and w > 0)]

        col_widths = _safe_col_widths(rects, bbox, n_cols)
        if col_widths:
            med = float(np.median(col_widths))
            if med > 0 and (col_widths[0] <= med * 0.7):
                label_col = 0

    # Degenerate table guard: if ALL headers are auto-generated ("Col1", "Col2", …)
    # and no data rows have real content, fall back to reading-order OCR text.
    all_auto_grid = all(re.fullmatch(r"Col\d+", h) for h in headers)
    grid_data_rows = [
        r for r in range(header_r + 1, n_rows)
        if any(cell_texts[r][c].strip() for c in range(n_cols))
    ]
    if all_auto_grid and not grid_data_rows:
        ro = sorted(
            [it for it in (ocr_items or []) if it],
            key=lambda it: (it[0][0][1] if isinstance(it[0], list) else 0,
                            it[0][0][0] if isinstance(it[0], list) else 0),
        )
        return " ".join(
            str(it[1]) for it in ro if it[1]
        ).strip(), info

    # Build Markdown pipe table
    start_r = header_r + 1
    data_rows = []
    for r in range(start_r, n_rows):
        row_cells = cell_texts[r]
        if sum(1 for c in range(n_cols) if row_cells[c].strip()) == 0:
            continue
        data_rows.append([row_cells[c] for c in range(n_cols)])

    md = _build_markdown_table(headers, data_rows)
    return md, info



def reconstruct_table_structured_text(
    ocr_items,
    y0: float,
    y1: float,
    y_tol: float = 10.0,
    x_tol: float = 45.0,
    max_cols: int = 8,
    max_rows: int = 120,
) -> str:
    """
    Build a Markdown pipe table from borderless table OCR items.

    Strategy:
    - restrict OCR items to [y0, y1]
    - group items into rows (by y)
    - estimate column centers (by x clustering)
    - build per-row cells
    - detect an optional header row
    - output as Markdown pipe table: | col1 | col2 | ...
    If header can't be detected reliably, fall back to "Col1..ColN".
    """
    items = _prep_items_with_centers(ocr_items)
    items_in = [it for it in items if (y0 - 2) <= it[8] <= (y1 + 2)]
    if not items_in:
        return ""

    rows = _group_items_to_rows(items_in, y_tol=y_tol)
    rows = rows[:max_rows]

    # estimate columns from x-centers
    all_xc = [it[7] for r in rows for it in r]
    col_centers = _cluster_sorted(sorted(all_xc), tol=max(35.0, x_tol))
    col_centers = sorted(col_centers)[:max_cols]
    if len(col_centers) < 2:
        # Not enough structure; return reading-order text
        ro = sorted(items_in, key=lambda it: (it[8], it[7]))
        out_lines = []
        last_y = None
        for it in ro:
            if last_y is None or abs(it[8] - last_y) > (y_tol * 1.6):
                out_lines.append(it[1])
            else:
                out_lines[-1] = (out_lines[-1] + " " + it[1]).strip()
            last_y = it[8]
        return "\n".join(out_lines)

    # build row cells
    table_rows: List[List[str]] = []
    for r in rows:
        cells = [""] * len(col_centers)
        for it in r:
            ci = _assign_to_nearest_cluster(it[7], col_centers)
            if ci is None:
                continue
            if 0 <= ci < len(cells):
                cells[ci] = (cells[ci] + " " + it[1]).strip() if cells[ci] else it[1]
        if any(cells):
            table_rows.append([normalize_spaces(c) for c in cells])

    if not table_rows:
        return ""

    def _row_nonempty_count(row: List[str]) -> int:
        return sum(1 for c in row if (c or "").strip())

    # header row heuristic: first row with >=2 non-empty cells and relatively short average length
    header_idx = None
    for i, row in enumerate(table_rows[:6]):
        if _row_nonempty_count(row) >= 2:
            lens = [len(c) for c in row if c]
            if lens and (sum(lens) / max(1, len(lens))) <= 35:
                header_idx = i
                break

    headers = None
    start_i = 0
    if header_idx is not None:
        headers = [c if c else f"Col{j+1}" for j, c in enumerate(table_rows[header_idx])]
        start_i = header_idx + 1
    else:
        headers = [f"Col{j+1}" for j in range(len(col_centers))]
        start_i = 0

    # determine if the first column is a "row label" column
    # if first column tends to be short while other columns can be long, treat it as row label
    first_col_lens = []
    other_col_lens = []
    for row in table_rows[start_i:start_i+10]:
        if row and row[0]:
            first_col_lens.append(len(row[0]))
        for c in row[1:]:
            if c:
                other_col_lens.append(len(c))
    label_col = 0 if (first_col_lens and other_col_lens and (statistics.median(first_col_lens) < 50) and (statistics.median(other_col_lens) >= 60)) else None

    # Degenerate table guard: if ALL headers are auto-generated ("Col1", "Col2", …)
    # and there are no data rows with real content, the table extraction failed.
    # Fall back to raw reading-order text so the page text is not silently discarded.
    all_auto = all(re.fullmatch(r"Col\d+", h) for h in headers)
    data_rows_with_content = [
        row for row in table_rows[start_i:] if any((c or "").strip() for c in row)
    ]
    def _fallback_prose(items_in):
        """Return reading-order prose text from OCR items (used when table parsing fails)."""
        ro = sorted(items_in, key=lambda it: (it[8], it[7]))
        fallback_lines: List[str] = []
        last_y2 = None
        for it in ro:
            if last_y2 is None or abs(it[8] - last_y2) > (y_tol * 1.6):
                fallback_lines.append(it[1])
            else:
                fallback_lines[-1] = (fallback_lines[-1] + " " + it[1]).strip()
            last_y2 = it[8]
        return "\n".join(fallback_lines)

    if all_auto and not data_rows_with_content:
        return _fallback_prose(items_in)

    # Sparse-table / org-chart guard (all-auto headers)
    if all_auto and data_rows_with_content:
        total_cells = sum(len(row) for row in data_rows_with_content)
        filled_cells = sum(1 for row in data_rows_with_content for c in row if (c or "").strip())
        sparsity = 1.0 - (filled_cells / max(1, total_cells))
        all_text = " ".join(c for row in data_rows_with_content for c in row if (c or "").strip())
        has_numerics = bool(re.search(r"\b\d{2,}\b", all_text))
        if sparsity > 0.65 and not has_numerics:
            return _fallback_prose(items_in)

    # Wide-sparse infographic guard: catches complex layouts like info-graphic pages where
    # scattered text boxes produce many columns with mostly empty cells.
    # Triggers regardless of whether headers are auto-generated.
    # Condition: ≥6 columns AND >55% empty cells AND avg filled-cell length < 20 chars
    n_cols = len(headers)
    if n_cols >= 6 and data_rows_with_content:
        total_cells_w = sum(len(row) for row in data_rows_with_content)
        filled_cells_w = sum(1 for row in data_rows_with_content for c in row if (c or "").strip())
        sparsity_w = 1.0 - (filled_cells_w / max(1, total_cells_w))
        if sparsity_w > 0.55 and filled_cells_w > 0:
            avg_filled_len = (
                sum(len((c or "").strip()) for row in data_rows_with_content
                    for c in row if (c or "").strip())
                / filled_cells_w
            )
            if avg_filled_len < 20:
                # Wide, sparse, short-content → infographic/diagram, fall back to prose
                return _fallback_prose(items_in)

    # Build Markdown pipe table
    data_rows = [row for row in table_rows[start_i:] if any(row)]
    return _build_markdown_table(headers, data_rows)

def extract_table_caption_lines(lines: List[str], max_lines: int = 4) -> List[str]:
    caps = []
    for ln in lines:
        if TABLE_CAPTION_RE.search((ln or "").strip()):
            caps.append(normalize_spaces(ln))
            if len(caps) >= max_lines:
                break
    return caps

# ===================================

def process_pdf_to_page_chunks(pdf_path: str, ocr: PaddleOCR, dpi: int, doc_meta: Optional[Dict[str, Any]] = None, llm_client=None) -> List[PageChunk]:
    doc_title = os.path.splitext(os.path.basename(pdf_path))[0]
    safe_title = safe_filename(doc_title)

    pdf = pdfium.PdfDocument(pdf_path)
    total_pages = len(pdf)
    pdf.close()

    page_limit = total_pages if MAX_PAGES_PER_PDF is None else min(total_pages, MAX_PAGES_PER_PDF)

    current_section = "(no heading)"
    outline_stack: List[str] = []  # hierarchical section path (v11.2)
    in_appendix = False
    appendix_label = None
    appendix_weak_streak = 0  # NEW: for early appendix detection via consecutive weak signals

    doc_meta = doc_meta or {}

    # LLM state (per PDF)
    llm_calls_total = 0
    llm_calls_page = 0
    heading_norm_cache: Dict[str, Dict[str, Any]] = {}

    # NEW: LLM usage tracking for QA/cost control
    llm_pages_used: set = set()
    llm_calls_by_type: Dict[str, int] = {"boundary_repair": 0, "group_dense": 0, "normalize_heading": 0}
    llm_call_events: List[Dict[str, Any]] = []

    def _log_llm_call(call_type: str, page_idx_0: int, llm_calls_this_page_types: set):
        nonlocal llm_calls_total, llm_calls_page
        llm_calls_total += 1
        llm_calls_page += 1
        llm_pages_used.add(page_idx_0 + 1)
        llm_calls_this_page_types.add(call_type)
        if call_type in llm_calls_by_type:
            llm_calls_by_type[call_type] += 1
        llm_call_events.append({"page": page_idx_0 + 1, "type": call_type, "total": llm_calls_total})
        print(f"[LLM] {os.path.basename(pdf_path)} | page={page_idx_0+1} | type={call_type} | total={llm_calls_total}/{MAX_LLM_CALLS_PER_PDF}")


    # Track running headers (doc title repeated on every page)
    header_counts: Dict[str, int] = {}
    header_seen_pages = 0
    header_blacklist: set = set()
    HEADER_BAND_FRAC = 0.12          # consider top 12% as possible running header zone
    HEADER_MIN_PAGES = 6             # start deciding after seeing N pages
    HEADER_FREQ_FRAC = 0.40          # line appears on >= this fraction -> running header
    # 0.40 (was 0.67): catches section-specific running headers that only appear on
    # 40-60% of pages (e.g. "Governor's Drought Task Force" in a 20-page section)
    # while avoiding blacklisting legitimate headings that appear once.

    doc_meta = doc_meta or {}

    chunks: List[PageChunk] = []

    prev_page_type: Optional[str] = None

    for page_idx in range(page_limit):
        llm_calls_page = 0  # per-page cap
        llm_calls_this_page_types: set = set()  # NEW: per-page LLM call types
        # 1) Render page
        pil = render_pdf_page_to_pil(pdf_path, page_idx, dpi=dpi)

        page_img_path = None
        if SAVE_PAGE_IMAGES:
            ensure_dir(os.path.join(IMAGE_DIR, safe_title))
            page_img_path = os.path.join(IMAGE_DIR, safe_title, f"{page_idx+1:04d}.png")
            pil.save(page_img_path)

        # 2) OCR full (items + image dims)
        ocr_items, img_w, img_h = ocr_image_full(ocr, pil)

        # OCR confidence diagnostics (for triage)
        confs = [it[2] for it in (ocr_items or []) if isinstance(it[2], (int, float))]
        ocr_conf_mean = float(np.mean(confs)) if confs else None
        # "low" is a heuristic; tune if needed
        low_thr = 0.85
        ocr_low_conf_ratio = (float(np.mean([1.0 if c < low_thr else 0.0 for c in confs])) if confs else None)

        # 3) Merge OCR items -> lines + positions
        line_objs = merge_ocr_items_into_lines_with_pos(ocr_items, y_threshold=12)
        merged_lines = [normalize_spaces(o["text"]) for o in line_objs if normalize_spaces(o["text"])]

        # 3.1) Update running-header statistics from top band
        header_seen_pages += 1
        top_y = HEADER_BAND_FRAC * img_h
        top_lines = []
        for o in line_objs:
            if o["y_min"] > top_y:
                break
            t = normalize_spaces(o["text"])
            if len(t) >= 6:
                top_lines.append(t.lower())
        # count unique per page to reduce bias
        for t in set(top_lines):
            header_counts[t] = header_counts.get(t, 0) + 1

        if header_seen_pages >= HEADER_MIN_PAGES:
            header_blacklist = set(
                t for t, c in header_counts.items()
                if (c / header_seen_pages) >= HEADER_FREQ_FRAC and len(t) >= 6
            )

        # 4) Multi-signal page classification (NO ML)
        ink_ratio = compute_page_ink_ratio(pil)
        ocr_text_cover = compute_ocr_text_cover(ocr_items, img_w, img_h)
        # 4.0) Image-based table grid detection (robust for bordered tables)
        table_grid = detect_table_grid_lines(pil)
        table_grid_flag = bool(table_grid.get('found'))
        # NEW: Exclude diagram/flowchart-like pages that contain some boxed graphics but are mostly prose text.
        # We use both global text coverage and whether OCR text actually lies inside the detected grid bbox.
        if table_grid_flag:
            try:
                dbg = table_grid.get("debug") or {}
                bbox_area_frac = float(dbg.get("bbox_area_frac") or 0.0)
                inter_cnt = int(dbg.get("inter_cnt") or 0)
        
                # (A) simple exclusion: lots of prose + relatively small boxed area
                if (bbox_area_frac < 0.35) and (ocr_text_cover > 0.18) and (inter_cnt < 350):
                    table_grid_flag = False
                    table_grid["found"] = False
                    dbg["reason"] = "diagram_exclusion_A"
                    dbg["ocr_text_cover"] = float(ocr_text_cover)
        
                # (B) text-in-bbox exclusion: for true tables, most OCR text sits inside the grid bbox
                if table_grid_flag and table_grid.get("bbox") and line_objs:
                    x0, y0, x1, y1 = table_grid["bbox"]
                    def _area(o):
                        return max(0.0, float(o["x_max"] - o["x_min"])) * max(0.0, float(o["y_max"] - o["y_min"]))
                    def _inter_area(o):
                        ix0 = max(float(o["x_min"]), float(x0))
                        iy0 = max(float(o["y_min"]), float(y0))
                        ix1 = min(float(o["x_max"]), float(x1))
                        iy1 = min(float(o["y_max"]), float(y1))
                        return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        
                    total_text_area = sum(_area(o) for o in line_objs)
                    if total_text_area > 0:
                        in_bbox_area = sum(_inter_area(o) for o in line_objs)
                        text_in_bbox_ratio = float(in_bbox_area) / float(total_text_area)
        
                        # If the detected grid bbox contains only a minority of text, it is likely a diagram/flowchart.
                        if (text_in_bbox_ratio < 0.55) and (ocr_text_cover > 0.10) and (bbox_area_frac < 0.65):
                            table_grid_flag = False
                            table_grid["found"] = False
                            dbg["reason"] = "diagram_exclusion_B"
                            dbg["ocr_text_cover"] = float(ocr_text_cover)
                            dbg["text_in_bbox_ratio"] = float(text_in_bbox_ratio)
                            dbg["bbox_area_frac"] = float(bbox_area_frac)
        
                table_grid["debug"] = dbg
            except Exception:
                pass

        toc_flag, toc_dbg = is_toc_like_page_strong(merged_lines, prev_was_toc=(prev_page_type=="toc"))
        if table_grid_flag:
            # table pages may look like 'figures' to ink-based heuristics; override early
            toc_flag = False

        fig_flag, fig_dbg = (False, {})
        form_flag, form_dbg = (False, {})
        if (not toc_flag) and (not table_grid_flag):
            form_flag, form_dbg = is_form_like_page_strong(merged_lines)
            if not form_flag:
                fig_flag, fig_dbg = is_figure_like_strong(ocr_items, img_w, img_h, ink_ratio=ink_ratio)
        # 5) Appendix detection (independent)
        trig_appendix, label, strong_appendix = detect_appendix_on_page(merged_lines)
        if trig_appendix:
            if strong_appendix or ((page_idx + 1) >= int(APPENDIX_TRIGGER_PCT * total_pages)):
                in_appendix = True
                appendix_label = label
                appendix_weak_streak = 0
            else:
                appendix_weak_streak += 1
                if appendix_weak_streak >= 2:
                    in_appendix = True
                    appendix_label = label
                    appendix_weak_streak = 0
        else:
            appendix_weak_streak = 0

        # 6) Decide routing / kept text
        page_type = "text"
        kept_lines: List[str] = merged_lines

        # If strong grid evidence exists, treat as table page (overrides figure/form heuristics)
        if table_grid_flag:
            page_type = "table"
            kept_lines = merged_lines
        
        if toc_flag and page_type != "table":
            page_type = "toc"
            kept_lines = []
        elif fig_flag:
            page_type = "figure"
            kept_lines = []
        elif form_flag:
            page_type = "form"
            kept_lines = []

        # Grey-zone safeguard: avoid false positives on real text pages
        if page_type in ("toc", "figure", "form"):
            if len(merged_lines) >= 22 and estimate_tokens_rough("\n".join(merged_lines)) >= 260:
                page_type = "text"
                kept_lines = merged_lines
                toc_flag = False
                fig_flag = False
                form_flag = False


                # 6.5) Cover-page detection (avoid over-splitting on cover pages)
        # Many PDFs have a cover with only the title/agency/date. We keep a minimal record but
        # do not allow heading-based splitting on these pages.
        cover_flag = False
        if page_type == "text" and page_idx <= 1:
            rough_toks = estimate_tokens_rough("\n".join(merged_lines))
            if rough_toks <= 140 and len(merged_lines) <= 28:
                has_numbered = any(HEADING_NUMERIC.match(ln) or HEADING_WORDY.match(ln) for ln in merged_lines)
                if not has_numbered:
                    cover_flag = True
                    page_type = "cover"
                    kept_lines = []

        # Track page type for TOC continuation detection on next page
        prev_page_type = page_type

        # 7) Table handling
        table_info: Dict[str, Any] = {"found": False}
        table_md = ""
        table_caption: List[str] = []
        table_block_text: Optional[str] = None

        if page_type == "table":
            # Table page detected by grid-lines evidence
            bbox = (table_grid.get("bbox") if isinstance(table_grid, dict) else None) or (0, 0, img_w, img_h)
            x0, y0, x1, y1 = bbox
            table_info = {"found": True, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "table_page": True, "debug": (table_grid.get("debug") if isinstance(table_grid, dict) else {})}
            rgb_for_table = np.array(pil.convert("RGB"))
            table_md, table_cv_dbg = reconstruct_table_structured_text_from_cells(ocr_items, rgb_for_table, bbox=(x0,y0,x1,y1))
            if not table_md:
                table_md = reconstruct_table_structured_text(ocr_items, y0=float(y0), y1=float(y1))
            table_info["cv2_cells"] = table_cv_dbg

            table_caption = extract_table_caption_lines(merged_lines, max_lines=6)

            tb_parts: List[str] = []
            for cap in table_caption:
                cap2 = normalize_spaces(cap)
                if cap2:
                    tb_parts.append(cap2)
            tb_parts.append("[TABLE]")
            if table_md:
                tb_parts.append(table_md)
            else:
                # fallback: keep raw page text if markdown reconstruction fails
                tb_parts.append("\n".join(kept_lines).strip())
            tb_parts.append("[/TABLE]")
            table_block_text = "\n".join([p for p in tb_parts if p]).strip()

            # === TABLE EXPORT (DEBUG/QA) BEGIN ===
            # Writes one markdown file per detected table for easy inspection.
            # Disable by setting EXPORT_TABLE_DEBUG_FILES=False.
            export_table_debug(doc_title, pdf_path, page_idx+1, table_block_text)
            # === TABLE EXPORT (DEBUG/QA) END ===

        elif page_type == "text":
            # Detect embedded tables inside normal text pages
            table_info = detect_table_block_from_ocr(ocr_items, img_w=img_w, img_h=img_h)
            if table_info.get("found"):
                y0 = float(table_info["y0"])
                y1 = float(table_info["y1"])
                table_md = reconstruct_table_structured_text(ocr_items, y0=y0, y1=y1)
                table_caption = extract_table_caption_lines(merged_lines, max_lines=3)

                cap_lines: List[str] = []
                for cap in table_caption:
                    cap2 = normalize_spaces(cap)
                    if cap2:
                        cap_lines.append(cap2)

                tb_parts: List[str] = []
                tb_parts.extend(cap_lines)
                tb_parts.append("[TABLE]")
                if table_md:
                    tb_parts.append(table_md)
                tb_parts.append("[/TABLE]")
                table_block_text = "\n".join([p for p in tb_parts if p]).strip()

            # === TABLE EXPORT (DEBUG/QA) BEGIN ===
            # Writes one markdown file per detected table for easy inspection.
            # Disable by setting EXPORT_TABLE_DEBUG_FILES=False.
            export_table_debug(doc_title, pdf_path, page_idx+1, table_block_text)
            # === TABLE EXPORT (DEBUG/QA) END ===

        # 7.8) Emit non-text pages immediately (toc/figure/form/cover) and skip heading splitting
        if page_type in ("toc", "figure", "form", "cover"):
            prefix = f"[{page_type.upper()} PAGE]"
            tokens = estimate_tokens_rough(prefix)
            page_debug = {
                "page_idx": page_idx + 1,
                "page_type": page_type,
                "ink_ratio": ink_ratio,
                "toc": toc_dbg,
                "figure": fig_dbg,
                "form": form_dbg,
                "table": (table_info.get("debug") if isinstance(table_info, dict) else {}),
                "has_table": bool(table_info.get("found")),
                "ocr_conf_mean": ocr_conf_mean,
                "ocr_low_conf_ratio": ocr_low_conf_ratio,
            }
            chunks.append(PageChunk(
                doc_title=doc_title,
                section=("TOC" if page_type=="toc" else current_section),
                section_path=(["TOC"] if page_type=="toc" else outline_stack.copy()),
                pages=f"{page_idx+1}-{page_idx+1}",
                is_appendix=in_appendix,
                appendix_label=appendix_label,
                tokens=tokens,
                text=prefix,
                pdf_path=pdf_path,
                page_image_path=page_img_path,
                page_type=page_type,
                is_toc=(page_type=="toc"),
                is_figure=(page_type=="figure"),
                is_form=(page_type=="form"),
                is_table=False,
                is_cover=(page_type=="cover"),
                page_debug=page_debug if SAVE_PAGE_DEBUG_JSONL else None,
                policy_id=(doc_meta or {}).get("policy_id"),
                source_url=(doc_meta or {}).get("source_url"),
                policy_level=(doc_meta or {}).get("policy_level"),
                policy_type=(doc_meta or {}).get("policy_type"),
                state_list=(doc_meta or {}).get("state_list") or [],
                county_list=(doc_meta or {}).get("county_list") or [],
                city_list=(doc_meta or {}).get("city_list") or [],
                tribe_list=(doc_meta or {}).get("tribe_list") or [],
            ))
            continue

        # 7.9) Emit table pages immediately (grid-detected or dominant table pages)
        if page_type == "table" and table_block_text:
            tokens = estimate_tokens_rough(table_block_text)
            page_debug = {
                "page_idx": page_idx + 1,
                "page_type": "table",
                "ink_ratio": ink_ratio,
                "toc": toc_dbg,
                "figure": fig_dbg,
                "form": form_dbg,
                "table": (table_info.get("debug") if isinstance(table_info, dict) else {}),
                "ocr_conf_mean": ocr_conf_mean,
                "ocr_low_conf_ratio": ocr_low_conf_ratio,
            }
            chunks.append(PageChunk(
                doc_title=doc_title,
                section=(appendix_label if (in_appendix and appendix_label) else current_section),
                section_path=outline_stack.copy(),
                pages=f"{page_idx+1}-{page_idx+1}",
                is_appendix=in_appendix,
                appendix_label=appendix_label,
                tokens=tokens,
                text=table_block_text,
                pdf_path=pdf_path,
                page_image_path=page_img_path,
                page_type="table",
                is_toc=False,
                is_figure=False,
                is_form=False,
                is_table=True,
                is_cover=False,
                page_debug=page_debug if SAVE_PAGE_DEBUG_JSONL else None,
                policy_id=(doc_meta or {}).get("policy_id"),
                source_url=(doc_meta or {}).get("source_url"),
                policy_level=(doc_meta or {}).get("policy_level"),
                policy_type=(doc_meta or {}).get("policy_type"),
                state_list=(doc_meta or {}).get("state_list") or [],
                county_list=(doc_meta or {}).get("county_list") or [],
                city_list=(doc_meta or {}).get("city_list") or [],
                tribe_list=(doc_meta or {}).get("tribe_list") or [],
            ))
            continue

        if page_type == "text" and table_info.get("found") and table_info.get("table_page") and table_block_text:
            tokens = estimate_tokens_rough(table_block_text)
            page_debug = {
                "page_idx": page_idx + 1,
                "page_type": "table",
                "ink_ratio": ink_ratio,
                "toc": toc_dbg,
                "figure": fig_dbg,
                "form": form_dbg,
                "table": (table_info.get("debug") if isinstance(table_info, dict) else {}),
                "ocr_conf_mean": ocr_conf_mean,
                "ocr_low_conf_ratio": ocr_low_conf_ratio,
            }
            chunks.append(PageChunk(
                doc_title=doc_title,
                section=(appendix_label if (in_appendix and appendix_label) else current_section),
                section_path=outline_stack.copy(),
                pages=f"{page_idx+1}-{page_idx+1}",
                is_appendix=in_appendix,
                appendix_label=appendix_label,
                tokens=tokens,
                text=table_block_text,
                pdf_path=pdf_path,
                page_image_path=page_img_path,
                page_type="table",
                is_toc=False,
                is_figure=False,
                is_form=False,
                is_table=True,
                is_cover=False,
                page_debug=page_debug if SAVE_PAGE_DEBUG_JSONL else None,
                policy_id=(doc_meta or {}).get("policy_id"),
                source_url=(doc_meta or {}).get("source_url"),
                policy_level=(doc_meta or {}).get("policy_level"),
                policy_type=(doc_meta or {}).get("policy_type"),
                state_list=(doc_meta or {}).get("state_list") or [],
                county_list=(doc_meta or {}).get("county_list") or [],
                city_list=(doc_meta or {}).get("city_list") or [],
                tribe_list=(doc_meta or {}).get("tribe_list") or [],
            ))
            continue

# 8) Heading extraction + multi-section split for text pages
        produced_any = False
        if page_type == "text":
            # exclude running headers from both headings and body
            hb = header_blacklist

            table_y0 = float(table_info.get("y0")) if table_info.get("found") else None
            table_y1 = float(table_info.get("y1")) if table_info.get("found") else None

            headings = extract_heading_spans(
                line_objs=line_objs,
                page_h=img_h,
                page_w=img_w,
                header_blacklist=hb,
                table_y0=table_y0,
                table_y1=table_y1,
            )


            # LLM: repair missed section boundaries (when heading detector finds nothing)
            if LLM_BOUNDARY_PROBE and USE_LLM and (llm_client is not None) and (llm_calls_total < MAX_LLM_CALLS_PER_PDF) and (not headings):
                if llm_calls_page < MAX_LLM_CALLS_PER_PAGE:
                    try:
                        # candidates from rule-based detector (empty here, but kept for prompt consistency)
                        heading_cands = []
                        start_lines = [normalize_spaces(x) for x in (merged_lines or []) if normalize_spaces(x)]
                        resp = llm_boundary_repair(llm_client, outline_stack, start_lines, heading_cands)
                        _log_llm_call("boundary_repair", page_idx, llm_calls_this_page_types)
                    except Exception:
                        resp = None
                    if isinstance(resp, dict) and resp.get("new_section") and resp.get("canonical"):
                        canon = normalize_spaces(str(resp.get("canonical")))
                        try:
                            lvl2 = int(resp.get("level") or 1)
                        except Exception:
                            lvl2 = 1
                        # fabricate a heading span at top of page
                        headings = [{
                            "hid": "",
                            "title": canon,
                            "y_min": 0.0,
                            "y_max": min(float(img_h) * 0.08, 120.0),
                        }]

            # If we have multiple headings on one page, split into regions
            if headings:
                regions = split_page_into_heading_regions(
                    line_objs=line_objs,
                    headings=headings,
                    page_h=img_h,
                    header_blacklist=hb,
                    table_y0=table_y0,
                    table_y1=table_y1,
                    table_block_text=table_block_text,
                )

                
                # --- LLM de-fragmentation: group many tiny heading regions on one page ---
                regions_grouped = regions
                try:
                    reg_infos = []
                    tiny = 0
                    for rid, r in enumerate(regions):
                        lines = r.get("lines") or []
                        body_tmp = "\n".join(lines[:6])
                        tok = estimate_tokens_rough("\n".join(lines))
                        if tok <= LLM_TINY_REGION_TOKENS:
                            tiny += 1
                        reg_infos.append({
                            "id": rid,
                            "heading": str(r.get("section") or ""),
                            "tokens": int(tok),
                            "preview": normalize_spaces(body_tmp)[:240],
                        })
                    tiny_ratio = (tiny / max(1, len(reg_infos)))
                    need_group = (len(reg_infos) >= LLM_MERGE_HEADINGS_MIN) or (tiny_ratio >= LLM_TINY_REGIONS_RATIO)

                    if need_group and USE_LLM and (llm_client is not None) and (llm_calls_total < MAX_LLM_CALLS_PER_PDF) and (llm_calls_page < MAX_LLM_CALLS_PER_PAGE):
                        resp = llm_group_dense_headings(llm_client, outline_stack, reg_infos)
                        _log_llm_call("group_dense", page_idx, llm_calls_this_page_types)

                        groups = (resp or {}).get("groups") if isinstance(resp, dict) else None
                        if isinstance(groups, list) and groups:
                            new_regs = []
                            used = []
                            for g in groups:
                                ids = g.get("region_ids") if isinstance(g, dict) else None
                                if not isinstance(ids, list) or not ids:
                                    continue
                                ids = [int(x) for x in ids]
                                used.extend(ids)
                                ids_sorted = sorted(ids)
                                # enforce consecutive + in-range
                                if ids_sorted[0] < 0 or ids_sorted[-1] >= len(regions):
                                    continue
                                if any((ids_sorted[i] + 1 != ids_sorted[i+1]) for i in range(len(ids_sorted)-1)):
                                    continue

                                merged_lines = []
                                first_heading_obj = None
                                for j, rid in enumerate(ids_sorted):
                                    rr = regions[rid]
                                    if j == 0:
                                        first_heading_obj = rr.get("section_heading")
                                    merged_lines.extend(rr.get("lines") or [])
                                    merged_lines.append("")  # soft separator

                                merged_lines = [ln for ln in merged_lines if ln is not None]
                                sec_name = normalize_spaces(str(g.get("section_name") or "")) if isinstance(g, dict) else ""
                                lvl_hint = g.get("level") if isinstance(g, dict) else None
                                new_regs.append({
                                    "section": sec_name or (regions[ids_sorted[0]].get("section") or current_section),
                                    "section_heading": first_heading_obj,
                                    "lines": [ln for ln in merged_lines if str(ln).strip()],
                                    "_llm_section_name": sec_name or None,
                                    "_llm_level": int(lvl_hint) if isinstance(lvl_hint, (int, float, str)) and str(lvl_hint).isdigit() else None,
                                })

                            # accept only if it covers all regions exactly once
                            if sorted(used) == list(range(len(regions))) and new_regs:
                                regions_grouped = new_regs
                except Exception:
                    regions_grouped = regions
# Create one PageChunk per region
                for reg_i, reg in enumerate(regions_grouped):
                    # Update hierarchical outline if this region starts at a heading
                    hobj = reg.get("section_heading")
                    if isinstance(hobj, dict):
                        raw_htext = section_label(hobj)
                        lvl = _heading_level_from_obj(hobj, float(img_w))
                        htext = raw_htext

                        # LLM: normalize heading text + level (cached)
                        if USE_LLM and (llm_client is not None) and (llm_calls_total < MAX_LLM_CALLS_PER_PDF):
                            if raw_htext not in heading_norm_cache:
                                if llm_calls_page < MAX_LLM_CALLS_PER_PAGE:
                                    try:
                                        resp = llm_normalize_heading(llm_client, raw_htext, outline_stack)
                                        _log_llm_call("normalize_heading", page_idx, llm_calls_this_page_types)
                                    except Exception:
                                        resp = None
                                    if isinstance(resp, dict):
                                        heading_norm_cache[raw_htext] = resp
                            resp2 = heading_norm_cache.get(raw_htext)
                            if isinstance(resp2, dict) and not resp2.get("drop"):
                                htext = normalize_spaces(str(resp2.get("canonical") or raw_htext))
                                try:
                                    lvl = int(resp2.get("level") or lvl)
                                except Exception:
                                    pass

                        outline_stack = _update_outline_stack(outline_stack, htext, lvl)
                        current_section = outline_stack[-1] if outline_stack else current_section

                    reg_section = (outline_stack[-1] if outline_stack else (reg.get("section") or current_section))
                    # Remove running header lines from region lines
                    reg_lines = [ln for ln in reg["lines"] if ln.lower() not in hb]
                    body = normalize_linebreaks_for_rag("\n".join(reg_lines).strip())
                    full_text = body
                    tokens = estimate_tokens_rough(full_text)

                    page_debug = {
                        "page_idx": page_idx + 1,
                        "page_type": page_type,
                        "ink_ratio": ink_ratio,
                        "toc": toc_dbg,
                        "figure": fig_dbg,
                        "form": form_dbg,
                        "table": table_info.get("debug", table_info),
                        "has_table": bool(table_info.get("found")),
                        "header_blacklist_size": len(hb),
                        #"headings_found": [{"text": h["text"], "y_min": h["y_min"], "score": h["score"]} for h in headings],
                        "region_index": reg_i,
                        #"region_y": [reg["y_min"], reg["y_max"]],
                        "lines_total": len(merged_lines),
                        "lines_region": len(reg_lines),
                        "ocr_conf_mean": ocr_conf_mean,
                        "ocr_low_conf_ratio": ocr_low_conf_ratio,
                        "n_headings": len(headings),
                        "llm_used": (len(llm_calls_this_page_types) > 0),
                        "llm_calls_total_so_far": llm_calls_total,
                        "llm_calls_page": llm_calls_page,
                        "llm_call_types": sorted(list(llm_calls_this_page_types)),
                    }

                    chunk = PageChunk(
                        doc_title=doc_title,
                        section=(appendix_label if (in_appendix and appendix_label) else reg_section),
                        pages=f"{page_idx+1} - {page_idx+1}",
                        is_appendix=in_appendix,
                        appendix_label=appendix_label,
                        tokens=tokens,
                        text=full_text,
                        pdf_path=pdf_path,
                        page_image_path=page_img_path,
                        page_type=page_type,
                        is_toc=False,
                        is_figure=False,
                        is_form=False,
                        is_table=bool(table_info.get("found")),
                        is_cover=(page_type=="cover"),
                        page_debug=page_debug,
                        policy_id=doc_meta.get("policy_id"),
                        source_url=doc_meta.get("source_url"),
                        policy_level=doc_meta.get("policy_level"),
                        policy_type=doc_meta.get("policy_type"),
                        state_list=doc_meta.get("state_list") or [],
                        county_list=doc_meta.get("county_list") or [],
                        city_list=doc_meta.get("city_list") or [],
                        tribe_list=doc_meta.get("tribe_list") or [],
                    )
                    chunks.append(chunk)
                    produced_any = True

                # Safer carry-over: use normalized outline_stack (avoid running headers)
                if outline_stack:
                    current_section = outline_stack[-1]

        if produced_any:
            continue

        # 9) Fallback: single chunk per page (toc/figure/form OR text without headings)
        # For text pages, also remove running headers and insert table block (if any)
        if page_type == "text":
            hb = header_blacklist
            cleaned = []
            # remove header lines + table band lines
            table_y0 = float(table_info.get("y0")) if table_info.get("found") else None
            table_y1 = float(table_info.get("y1")) if table_info.get("found") else None

            for obj in line_objs:
                t = normalize_spaces(obj["text"])
                if not t:
                    continue
                if t.lower() in hb:
                    continue
                if table_y0 is not None and table_y1 is not None:
                    ymid = 0.5 * (obj["y_min"] + obj["y_max"])
                    if ymid >= table_y0 and ymid <= table_y1:
                        continue
                cleaned.append(t)

            if table_block_text:
                cleaned.append(table_block_text)

            kept_lines = cleaned

        body = "\n".join(kept_lines).strip()
        prefix = f"[{page_type.upper()} PAGE]" if INCLUDE_PAGE_PREFIX_IN_TEXT else ""
        full_text = ((prefix + ("\n\n" + body if body else "")).strip() if prefix else body)
        tokens = estimate_tokens_rough(full_text)

        page_debug = {
            "page_idx": page_idx + 1,
            "page_type": page_type,
            "ink_ratio": ink_ratio,
            "toc": toc_dbg,
            "figure": fig_dbg,
            "form": form_dbg,
            "table": table_info.get("debug", table_info),
            "has_table": bool(table_info.get("found")),
            "header_blacklist_size": len(header_blacklist),
            "lines": len(merged_lines),
            "kept_lines": len(kept_lines),
            "ocr_conf_mean": ocr_conf_mean,
            "ocr_low_conf_ratio": ocr_low_conf_ratio,
            "n_headings": 0,
        }

        chunk = PageChunk(
            doc_title=doc_title,
            section=(appendix_label if (in_appendix and appendix_label) else current_section),
            section_path=outline_stack.copy(),
            pages=f"{page_idx+1} - {page_idx+1}",
            is_appendix=in_appendix,
            appendix_label=appendix_label,
            tokens=tokens,
            text=full_text,
            pdf_path=pdf_path,
            page_image_path=page_img_path,
            page_type=page_type,
            is_toc=(page_type == "toc"),
            is_figure=(page_type == "figure"),
            is_form=(page_type == "form"),
            is_table=bool(table_info.get("found")) if page_type == "text" else False,
            is_cover=(page_type == "cover"),
            page_debug=page_debug,
                        policy_id=doc_meta.get("policy_id"),
                        source_url=doc_meta.get("source_url"),
                        policy_level=doc_meta.get("policy_level"),
                        policy_type=doc_meta.get("policy_type"),
                        state_list=doc_meta.get("state_list") or [],
                        county_list=doc_meta.get("county_list") or [],
                        city_list=doc_meta.get("city_list") or [],
                        tribe_list=doc_meta.get("tribe_list") or [],
        )
        chunks.append(chunk)



    # LLM summary for this PDF
    try:
        if USE_LLM and (llm_client is not None):
            print(f"[LLM-SUMMARY] {os.path.basename(pdf_path)} | total={llm_calls_total} | by_type={llm_calls_by_type} | pages={sorted(list(llm_pages_used))}")
    except Exception:
        pass

    return chunks


def merge_pages_by_section(page_chunks: List[PageChunk]) -> List[Dict]:
    """Merge consecutive pages with same (section, appendix flags).

    Skip TOC + FIGURE + FORM + COVER pages by default.

    IMPORTANT FIX:
    - Use a consistent merge key (section_path if available, else section string).
    - Allow '(no heading)' pages with zero headings to inherit the previous section when consecutive.
    """
    blocks: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for ch in page_chunks:
        if ch.is_toc or ch.is_figure or getattr(ch, "is_form", False) or getattr(ch, "is_cover", False):
            continue

        p0, _ = _parse_pages_range(ch.pages)
        _, body = split_prefix_and_body(ch.text)

        ch_section_key = _section_path_tuple(ch)
        ch_appendix_label = (ch.appendix_label or None)
        ch_is_appendix = bool(ch.is_appendix)

        # Inherit previous section if this page has no detected headings (continuation pages)
        if cur is not None:
            try:
                n_headings = int((ch.page_debug or {}).get("n_headings") or 0)
            except Exception:
                n_headings = 0
            if (ch.section == "(no heading)") and (not ch_is_appendix) and (n_headings == 0) and (int(p0) == int(cur["end_page"]) + 1):
                ch_section_key = cur["section_key"]
                ch.section = cur["section"]

        key = (tuple(ch_section_key), ch_is_appendix, ch_appendix_label)

        if cur is None:
            cur = {
                "doc_title": ch.doc_title,
                "pdf_path": ch.pdf_path,
                "policy_id": getattr(ch, "policy_id", None),
                "source_url": getattr(ch, "source_url", None),
                "policy_level": getattr(ch, "policy_level", None),
                "policy_type": getattr(ch, "policy_type", None),
                "state_list": getattr(ch, "state_list", None),
                "county_list": getattr(ch, "county_list", None),
                "city_list": getattr(ch, "city_list", None),
                "tribe_list": getattr(ch, "tribe_list", None),
                "section": ch.section,
                "section_key": tuple(ch_section_key),
                "is_appendix": ch_is_appendix,
                "appendix_label": ch_appendix_label,
                "start_page": int(p0),
                "end_page": int(p0),
                "bodies": [body],
            }
            continue

        cur_key = (tuple(cur["section_key"]), bool(cur["is_appendix"]), (cur["appendix_label"] or None))
        if key == cur_key and int(p0) == int(cur["end_page"]) + 1:
            cur["end_page"] = int(p0)
            cur["bodies"].append(body)
        else:
            blocks.append(cur)
            cur = {
                "doc_title": ch.doc_title,
                "pdf_path": ch.pdf_path,
                "policy_id": getattr(ch, "policy_id", None),
                "source_url": getattr(ch, "source_url", None),
                "policy_level": getattr(ch, "policy_level", None),
                "policy_type": getattr(ch, "policy_type", None),
                "state_list": getattr(ch, "state_list", None),
                "county_list": getattr(ch, "county_list", None),
                "city_list": getattr(ch, "city_list", None),
                "tribe_list": getattr(ch, "tribe_list", None),
                "section": ch.section,
                "section_key": tuple(ch_section_key),
                "is_appendix": ch_is_appendix,
                "appendix_label": ch_appendix_label,
                "start_page": int(p0),
                "end_page": int(p0),
                "bodies": [body],
            }

    if cur is not None:
        blocks.append(cur)

    return blocks

def split_text_into_chunks(
    doc_title: str,
    section: str,
    start_page: int,
    end_page: int,
    is_appendix: bool,
    appendix_label: Optional[str],
    full_body: str,
    target_tokens: int,
    overlap_tokens: int,
    pdf_path: Optional[str] = None,
    policy_id: Optional[str] = None,
    source_url: Optional[str] = None,
    policy_level: Optional[str] = None,
    policy_type: Optional[str] = None,
    state_list: Optional[List[str]] = None,
    county_list: Optional[List[str]] = None,
    city_list: Optional[List[str]] = None,
    tribe_list: Optional[List[str]] = None,
    section_path: Optional[List[str]] = None,
) -> List[PageChunk]:
    """
    Paragraph-first splitting + sentence fallback + overlap.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", full_body) if p.strip()]
    if not paras:
        paras = [full_body.strip()]

    chunks_text: List[str] = []
    cur_parts: List[str] = []
    cur_tok = 0

    def tail_by_tokens(text: str, k: int) -> str:
        words = re.split(r"\s+", text.strip())
        if len(words) <= k:
            return text.strip()
        return " ".join(words[-k:])

    for para in paras:
        t = estimate_tokens_rough(para)

        # If a single paragraph is too large, split by sentences
        if t > target_tokens:
            if cur_parts:
                chunks_text.append("\n\n".join(cur_parts))
                cur_parts, cur_tok = [], 0

            sents = re.split(r"(?<=[\.\?\!;:])\s+", para)
            tmp = ""
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                cand = (tmp + " " + s).strip() if tmp else s
                if estimate_tokens_rough(cand) <= target_tokens:
                    tmp = cand
                else:
                    if tmp:
                        chunks_text.append(tmp)
                    tmp = s
            if tmp:
                chunks_text.append(tmp)
            continue

        # Normal accumulation
        if cur_tok + t <= target_tokens:
            cur_parts.append(para)
            cur_tok += t
        else:
            chunks_text.append("\n\n".join(cur_parts))
            prev_tail = tail_by_tokens(chunks_text[-1], overlap_tokens)
            cur_parts = [prev_tail, para] if prev_tail else [para]
            cur_tok = estimate_tokens_rough("\n\n".join(cur_parts))

    if cur_parts:
        chunks_text.append("\n\n".join(cur_parts))

    out: List[PageChunk] = []
    for body in chunks_text:
        text = body
        has_table = bool(text and "[TABLE]" in text)
        out.append(PageChunk(
            doc_title=doc_title,
            section=section,
            section_path=section_path,
            pages=f"{start_page} - {end_page}",
            is_appendix=is_appendix,
            appendix_label=appendix_label,
            tokens=estimate_tokens_rough(text),
            text=text,
            pdf_path=pdf_path,
            policy_id=policy_id,
            source_url=source_url,
            policy_level=policy_level,
            policy_type=policy_type,
            state_list=state_list or [],
            county_list=county_list or [],
            city_list=city_list or [],
            tribe_list=tribe_list or [],
            page_type="table" if has_table else "text",
            is_table=has_table,
            is_toc=False,
            is_figure=False,
            is_form=False,
            is_cover=False,
        ))
    return out


def consolidate_tiny_blocks(blocks: List[Dict], min_block_tokens: int = 200) -> List[Dict]:
    """Merge section-blocks that are too small into the preceding block.

    Tiny blocks arise when the heading detector fires on contributor names,
    committee names, or other short text that isn't a real document section
    boundary. Merging them forward keeps the text intact under a meaningful
    parent section heading.

    Only merges *text* blocks (not table-only blocks whose bodies start with
    '[TABLE]'), so table chunks are always kept separate.
    """
    if not blocks:
        return blocks

    def _block_tokens(b: Dict) -> int:
        return sum(estimate_tokens_rough(body) for body in b["bodies"])

    def _is_table_only_block(b: Dict) -> bool:
        # If every non-empty body is a table extraction, don't merge
        non_empty = [body for body in b["bodies"] if body.strip()]
        return bool(non_empty) and all(
            body.strip().startswith("[TABLE]") for body in non_empty
        )

    result: List[Dict] = []
    for block in blocks:
        tok = _block_tokens(block)
        if tok < min_block_tokens and result and not _is_table_only_block(block):
            # Merge into the previous block (inherit previous section name)
            prev = result[-1]
            prev["bodies"].extend(block["bodies"])
            prev["end_page"] = max(prev["end_page"], block["end_page"])
        else:
            result.append(block)

    return result


def build_scientific_chunks(page_chunks: List[PageChunk]) -> List[PageChunk]:
    blocks = merge_pages_by_section(page_chunks)
    blocks = consolidate_tiny_blocks(blocks, min_block_tokens=200)
    final_chunks: List[PageChunk] = []

    for b in blocks:
        full_body = "\n\n".join(b["bodies"])
        # Build section_path list from the section_key tuple stored in the block
        sec_key = b.get("section_key") or ()
        sec_path_list = list(sec_key) if sec_key else None

        # Derive display section name with sub-section granularity for appendix
        if b.get("is_appendix") and b.get("appendix_label"):
            app_label = b["appendix_label"]
            leaf = sec_path_list[-1] if (sec_path_list and sec_path_list[-1] != app_label) else None
            section_for_chunk = f"{app_label} > {leaf}" if leaf else app_label
        else:
            section_for_chunk = b["section"]

        final_chunks.extend(
            split_text_into_chunks(
                doc_title=b["doc_title"],
                section=section_for_chunk,
                start_page=b["start_page"],
                end_page=b["end_page"],
                is_appendix=b["is_appendix"],
                appendix_label=b["appendix_label"],
                full_body=full_body,
                target_tokens=TARGET_TOKENS,
                overlap_tokens=OVERLAP_TOKENS,
                pdf_path=b.get("pdf_path"),
                policy_id=b.get("policy_id"),
                source_url=b.get("source_url"),
                policy_level=b.get("policy_level"),
                policy_type=b.get("policy_type"),
                state_list=b.get("state_list"),
                county_list=b.get("county_list"),
                city_list=b.get("city_list"),
                tribe_list=b.get("tribe_list"),
                section_path=sec_path_list,
            )
        )
    return final_chunks



# =============================================================================
# Diagnostics (per-PDF)
# =============================================================================

# =============================================================================
# Post-merge to reduce over-fragmentation
# =============================================================================
_PAGE_HEADER_RE = re.compile(
    r"^\s*\[TABLE\]\s*(page\s+\d+|.*?page\s+\d+\s*$|city of [a-z ]+|[A-Z ]{5,}\s*(plan|program|report|management)?\s*)\s*\[/TABLE\]\s*$",
    re.IGNORECASE | re.DOTALL,
)

def merge_tiny_final_chunks(final_chunks: List[PageChunk], min_tokens: int = 260) -> List[PageChunk]:
    """Merge tiny chunks to reduce over-fragmentation.

    Also drops degenerate table chunks under 30 tokens that contain only
    page headers, logos, or city names (OCR artifacts, not real content).

    Only merges within the same (doc_title, section, is_appendix, appendix_label).
    Prefers merging into the previous chunk; otherwise merges forward.
    """
    if not final_chunks:
        return final_chunks

    # Pre-filter: drop garbage table chunks (<= 30 tokens, page-header-like content)
    def _is_garbage_table(ch: PageChunk) -> bool:
        if not ch.is_table:
            return False
        tok = ch.tokens or 0
        if tok > 30:
            return False
        txt = (ch.text or "").strip()
        return bool(_PAGE_HEADER_RE.match(txt))

    final_chunks = [c for c in final_chunks if not _is_garbage_table(c)]
    if not final_chunks:
        return final_chunks

    merged: List[PageChunk] = []
    buf = final_chunks[0]

    def same_bucket(a: PageChunk, b: PageChunk) -> bool:
        return (a.doc_title == b.doc_title
                and a.section == b.section
                and bool(a.is_appendix) == bool(b.is_appendix)
                and (a.appendix_label or None) == (b.appendix_label or None))

    for nxt in final_chunks[1:]:
        if buf.tokens is None:
            buf.tokens = estimate_tokens_rough(buf.text or "")
        if nxt.tokens is None:
            nxt.tokens = estimate_tokens_rough(nxt.text or "")

        if buf.tokens < min_tokens and same_bucket(buf, nxt):
            # merge buf into nxt
            nxt.text = (buf.text.rstrip() + "\n\n" + (nxt.text or "").lstrip()).strip()
            nxt.tokens = int((buf.tokens or 0) + (nxt.tokens or 0))
            # expand page range
            try:
                a0 = int(str(buf.pages).split("-")[0])
                b1 = int(str(nxt.pages).split("-")[-1])
                nxt.pages = f"{a0}-{b1}"
            except Exception:
                pass
            buf = nxt
        else:
            merged.append(buf)
            buf = nxt

    merged.append(buf)

    # second pass: if the first chunk is still tiny, merge forward if possible
    if len(merged) >= 2 and merged[0].tokens < min_tokens and same_bucket(merged[0], merged[1]):
        merged[1].text = (merged[0].text.rstrip() + "\n\n" + merged[1].text.lstrip()).strip()
        merged[1].tokens = int((merged[0].tokens or 0) + (merged[1].tokens or 0))
        try:
            a0 = int(str(merged[0].pages).split("-")[0])
            b1 = int(str(merged[1].pages).split("-")[-1])
            merged[1].pages = f"{a0}-{b1}"
        except Exception:
            pass
        merged = merged[1:]

    return merged

def compute_pdf_diagnostics(pdf_path: str, page_chunks: List[PageChunk], final_chunks: List[PageChunk], doc_meta: Dict[str, Any]) -> Dict[str, Any]:
    # Page-type distribution
    n_pages_seen = len(page_chunks)
    n_toc = sum(1 for c in page_chunks if c.is_toc)
    n_fig = sum(1 for c in page_chunks if c.is_figure)
    n_form = sum(1 for c in page_chunks if getattr(c, "is_form", False))
    n_cover = sum(1 for c in page_chunks if getattr(c, "is_cover", False))
    n_text = n_pages_seen - n_toc - n_fig - n_form - n_cover

    # OCR confidence (if available)
    conf_vals = []
    low_conf = 0
    conf_total = 0
    for c in page_chunks:
        dbg = c.page_debug or {}
        # we store per-page OCR mean as 'ocr_conf_mean' when available
        if "ocr_conf_mean" in dbg and dbg["ocr_conf_mean"] is not None:
            conf_vals.append(float(dbg["ocr_conf_mean"]))
        if "ocr_low_conf_ratio" in dbg and dbg["ocr_low_conf_ratio"] is not None:
            # ratio of boxes below threshold, aggregate approximately by averaging ratios
            low_conf += float(dbg["ocr_low_conf_ratio"])
            conf_total += 1

    ocr_conf_mean = float(np.mean(conf_vals)) if conf_vals else None
    ocr_low_conf_ratio = (low_conf / conf_total) if conf_total else None

    # Heading density: count detected headings per page from page_debug
    heading_counts = []
    for c in page_chunks:
        dbg = c.page_debug or {}
        if "n_headings" in dbg:
            heading_counts.append(int(dbg["n_headings"]))
    headings_per_page_mean = float(np.mean(heading_counts)) if heading_counts else None

    # Chunk size quality
    chunk_tokens = [c.tokens for c in final_chunks] if final_chunks else []
    n_chunks = len(chunk_tokens)
    small_chunks = sum(1 for t in chunk_tokens if t < 220)
    median_tokens = float(np.median(chunk_tokens)) if chunk_tokens else None
    p10_tokens = float(np.percentile(chunk_tokens, 10)) if chunk_tokens else None
    p90_tokens = float(np.percentile(chunk_tokens, 90)) if chunk_tokens else None

    return {
        "pdf_path": pdf_path,
        "pdf_basename": os.path.basename(pdf_path),
        "policy_id": doc_meta.get("policy_id"),
        "policy_level": doc_meta.get("policy_level"),
        "policy_type": doc_meta.get("policy_type"),
        "state_list": json.dumps(doc_meta.get("state_list") or [], ensure_ascii=False),
        "county_list": json.dumps(doc_meta.get("county_list") or [], ensure_ascii=False),
        "city_list": json.dumps(doc_meta.get("city_list") or [], ensure_ascii=False),
        "tribe_list": json.dumps(doc_meta.get("tribe_list") or [], ensure_ascii=False),
        "pages_seen": n_pages_seen,
        "text_pages": n_text,
        "toc_pages": n_toc,
        "figure_pages": n_fig,
        "form_pages": n_form,
        "cover_pages": n_cover,
        "ocr_conf_mean": ocr_conf_mean,
        "ocr_low_conf_ratio": ocr_low_conf_ratio,
        "headings_per_page_mean": headings_per_page_mean,
        "final_chunks": n_chunks,
        "median_chunk_tokens": median_tokens,
        "p10_chunk_tokens": p10_tokens,
        "p90_chunk_tokens": p90_tokens,
        "small_chunk_ratio_lt220": (small_chunks / n_chunks) if n_chunks else None,
    }

def save_diagnostics_csv(rows: List[Dict[str, Any]], path: str):
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =============================================================================
# IO
# =============================================================================
def save_jsonl(chunks: List[PageChunk], path: str, overwrite: bool = False):
    ensure_dir(os.path.dirname(path))
    mode = "w" if overwrite else "a"
    with open(path, mode, encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")


# =============================================================================
# Runner
# =============================================================================
def run(pdf_dir: str | None = None, print_samples_per_pdf: int = 1, sample_from_metadata_n: int = 0):
    ensure_dir(OUTPUT_DIR)
    if SAVE_PAGE_IMAGES:
        ensure_dir(IMAGE_DIR)

    if EXPORT_TABLE_DEBUG_FILES:
        ensure_dir(TABLE_DEBUG_DIR)

    # reset outputs
    if os.path.exists(FINAL_JSONL_PATH):
        os.remove(FINAL_JSONL_PATH)
    if SAVE_PAGE_DEBUG_JSONL and os.path.exists(PAGE_DEBUG_JSONL_PATH):
        os.remove(PAGE_DEBUG_JSONL_PATH)

    ocr = init_paddleocr()

    # LLM client (Portkey)
    llm_client = None
    if USE_LLM:
        llm_client = get_portkey_client()

    # Load policy metadata once (path -> geo + other fields)
    by_full_path, by_basename = load_policy_metadata(METADATA_CSV_PATH)
    diagnostics_rows: List[Dict[str, Any]] = []

    # Decide which PDFs to process
    # sample_from_metadata_n semantics:
    #   None      → os.walk(pdf_dir) — scan entire directory tree
    #   0         → read ALL file paths from METADATA_CSV_PATH
    #   N (>0)    → read first N file paths from METADATA_CSV_PATH
    pdf_files: List[str] = []

    if sample_from_metadata_n is not None:
        # Read from CSV (all rows when n==0, first N rows when n>0)
        try:
            df_meta = pd.read_csv(METADATA_CSV_PATH)
            rows = df_meta.get("file_path", [])
            if sample_from_metadata_n > 0:
                rows = rows[:sample_from_metadata_n]
            for fp in rows:
                p = str(fp or "").strip()
                if not p:
                    continue
                if os.path.exists(p):
                    pdf_files.append(p)
                else:
                    # Also try normalized slashes (helps when CSV was edited on Windows)
                    p2 = p.replace("\\", "/").replace("/", os.sep).replace("\\", os.sep)
                    if os.path.exists(p2):
                        pdf_files.append(p2)
                    else:
                        print(f"[WARN] Metadata path not found on disk (skipped): {p}")
        except Exception as e:
            print(f"[WARN] Failed reading metadata CSV for sampling: {e}")

        pdf_files = sorted(list(dict.fromkeys(pdf_files)))  # de-dup, stable
        if not pdf_files:
            print("[WARN] No PDFs found from metadata CSV; falling back to pdf_dir scan.")
            sample_from_metadata_n = None

    if sample_from_metadata_n is None:
        if not pdf_dir:
            raise ValueError("pdf_dir is required when sample_from_metadata_n=None")
        for root, _, files in os.walk(pdf_dir):
            for fn in files:
                if fn.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, fn))
        pdf_files.sort()

    for pdf_path in tqdm(pdf_files, desc="PDFs"):
        try:
            doc_meta = lookup_policy_meta(pdf_path, by_full_path, by_basename)
            page_chunks = process_pdf_to_page_chunks(pdf_path, ocr=ocr, dpi=DPI, doc_meta=doc_meta, llm_client=llm_client)

            if SAVE_PAGE_DEBUG_JSONL:
                save_jsonl(page_chunks, PAGE_DEBUG_JSONL_PATH, overwrite=False)

            final_chunks = build_scientific_chunks(page_chunks)
            # Reduce over-fragmentation
            final_chunks = merge_tiny_final_chunks(final_chunks, min_tokens=260)
            save_jsonl(final_chunks, FINAL_JSONL_PATH, overwrite=False)

            diagnostics_rows.append(compute_pdf_diagnostics(pdf_path, page_chunks, final_chunks, doc_meta))

            for i in range(min(print_samples_per_pdf, len(final_chunks))):
                print(format_print(final_chunks[i]))
                print()
        except Exception as _pdf_err:
            import traceback
            print(f"\n[ERROR] Skipping {os.path.basename(pdf_path)}: {_pdf_err}")
            traceback.print_exc()
            diagnostics_rows.append({
                "pdf_path": pdf_path,
                "pdf_basename": os.path.basename(pdf_path),
                "error": str(_pdf_err),
            })

    # Write diagnostics once
    try:
        save_diagnostics_csv(diagnostics_rows, DIAGNOSTICS_CSV_PATH)
        print(f"[OK] Wrote diagnostics CSV: {DIAGNOSTICS_CSV_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to write diagnostics CSV: {e}")


if __name__ == "__main__":
    PDF_DIR = "E:/2026_capstone/policy_data/pdf_data/data"

    # --- Sanity check: first N PDFs from metadata CSV ---
    # run(pdf_dir=PDF_DIR, print_samples_per_pdf=1, sample_from_metadata_n=3)

    # --- Full batch: all PDFs listed in policy_metadata_4.csv ---
    # run(pdf_dir=PDF_DIR, print_samples_per_pdf=0, sample_from_metadata_n=0)

    # --- Full directory scan (ignores CSV, processes every PDF in pdf_dir) ---
    # run(pdf_dir=PDF_DIR, print_samples_per_pdf=0, sample_from_metadata_n=None)

    run(pdf_dir=PDF_DIR, print_samples_per_pdf=0, sample_from_metadata_n=3)