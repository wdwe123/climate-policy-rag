"""
Manual Chunking Tool
====================
Creates chunks in the exact same JSONL format as final_chunks_v11_6.jsonl.
Only requires entering section name + text content per chunk.
Document metadata is auto-loaded from pdf_data/metadata/unstructured_metadata.csv.

Run with:
    streamlit run manual_chunker.py
"""

import ast
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import tiktoken

# ── Constants ──────────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent / "pdf_data" / "metadata" / "unstructured_metadata.csv"
OUTPUT_DIR = Path(__file__).parent / "chunking_output"
PAGE_TYPES = ["text", "table", "figure", "form", "toc", "uncertain"]

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_encoder():
    return tiktoken.get_encoding("cl100k_base")


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["num_id", "policy_id", "source_url", "file_path",
               "state_list", "county_list", "city_list", "tribe_list",
               "policy_level", "policy_type"],
        dtype=str,
    )
    df["doc_title"] = df["file_path"].apply(lambda p: Path(str(p)).stem if pd.notna(p) and str(p).strip() else "")
    return df


def parse_list_col(val):
    """Parse a Python-list string like \"['Arizona']\" into a Python list."""
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        result = ast.literal_eval(str(val).strip())
        return result if isinstance(result, list) else []
    except Exception:
        return []


def count_tokens(text: str) -> int:
    enc = get_encoder()
    return len(enc.encode(text))


def make_chunk(row: pd.Series, section: str, text: str, pages: str,
               page_type: str, is_appendix: bool) -> dict:
    return {
        "doc_title": row["doc_title"],
        "section": section,
        "pages": pages,
        "is_appendix": is_appendix,
        "appendix_label": None,
        "tokens": count_tokens(text),
        "text": text,
        "section_path": [section],
        "pdf_path": row["file_path"],
        "page_image_path": None,
        "page_type": page_type,
        "is_toc": False,
        "is_figure": False,
        "is_form": False,
        "is_table": False,
        "is_cover": False,
        "page_debug": None,
        "policy_id": row["policy_id"] if pd.notna(row["policy_id"]) else None,
        "source_url": row["source_url"] if pd.notna(row["source_url"]) else None,
        "policy_level": row["policy_level"] if pd.notna(row["policy_level"]) else None,
        "policy_type": row["policy_type"] if pd.notna(row["policy_type"]) else None,
        "state_list": parse_list_col(row["state_list"]),
        "county_list": parse_list_col(row["county_list"]),
        "city_list": parse_list_col(row["city_list"]),
        "tribe_list": parse_list_col(row["tribe_list"]),
    }


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Manual Chunker", layout="wide")
st.title("Manual Chunking Tool")

# ── Session state ──────────────────────────────────────────────────────────────
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "form_key" not in st.session_state:
    st.session_state.form_key = 0

# Persistent across Add Chunk
if "page_start" not in st.session_state:
    st.session_state.page_start = None   # int or None
if "page_end" not in st.session_state:
    st.session_state.page_end = None
if "ps_input" not in st.session_state:
    st.session_state.ps_input = ""
if "pe_input" not in st.session_state:
    st.session_state.pe_input = ""
if "sticky_appendix" not in st.session_state:
    st.session_state.sticky_appendix = False


# ── Page-number helpers ─────────────────────────────────────────────────────────
def _sync_pages():
    """Auto-fill right from left; enforce left ≤ right."""
    ps, pe = st.session_state.page_start, st.session_state.page_end
    if ps is not None and pe is None:
        st.session_state.page_end = ps
        st.session_state.pe_input = str(ps)
    elif ps is not None and pe is not None and ps > pe:
        st.session_state.page_end = ps
        st.session_state.pe_input = str(ps)

def _ps_minus():
    if st.session_state.page_start and st.session_state.page_start > 1:
        st.session_state.page_start -= 1
        st.session_state.ps_input = str(st.session_state.page_start)
        _sync_pages()

def _ps_plus():
    st.session_state.page_start = (st.session_state.page_start or 0) + 1
    st.session_state.ps_input = str(st.session_state.page_start)
    _sync_pages()

def _pe_minus():
    floor = st.session_state.page_start or 1
    if st.session_state.page_end and st.session_state.page_end > floor:
        st.session_state.page_end -= 1
        st.session_state.pe_input = str(st.session_state.page_end)

def _pe_plus():
    st.session_state.page_end = (st.session_state.page_end or st.session_state.page_start or 0) + 1
    st.session_state.pe_input = str(st.session_state.page_end)

def _on_ps_change():
    val = st.session_state.ps_input.strip()
    if val == "":
        st.session_state.page_start = None
    elif val.isdigit() and int(val) >= 1:
        st.session_state.page_start = int(val)
        _sync_pages()
    else:
        st.session_state.ps_input = str(st.session_state.page_start) if st.session_state.page_start else ""

def _on_pe_change():
    val = st.session_state.pe_input.strip()
    if val == "":
        st.session_state.page_end = None
    elif val.isdigit() and int(val) >= 1:
        pe_val = int(val)
        ps = st.session_state.page_start
        if ps is not None and pe_val < ps:
            st.session_state.page_end = ps
            st.session_state.pe_input = str(ps)
        else:
            st.session_state.page_end = pe_val
    else:
        st.session_state.pe_input = str(st.session_state.page_end) if st.session_state.page_end else ""

# ── Load CSV ───────────────────────────────────────────────────────────────────
if not CSV_PATH.exists():
    st.error(f"Metadata CSV not found at: {CSV_PATH}")
    st.stop()

df = load_csv(str(CSV_PATH))

# ── Sidebar — document selection ───────────────────────────────────────────────
with st.sidebar:
    st.header("Document Selection")

    display_labels = [
        f"{row['num_id']}: {row['policy_id']}"
        for _, row in df.iterrows()
    ]
    selected_idx = st.selectbox(
        "Select document",
        options=range(len(df)),
        format_func=lambda i: display_labels[i],
    )
    row = df.iloc[selected_idx]

    st.divider()
    st.subheader("Loaded Metadata")
    st.markdown(f"**doc_title:** `{row['doc_title']}`")
    st.markdown(f"**policy_id:** `{row['policy_id']}`")
    st.markdown(f"**policy_level:** `{row['policy_level']}`")
    st.markdown(f"**policy_type:** `{row['policy_type']}`")
    st.markdown(f"**state_list:** `{parse_list_col(row['state_list'])}`")
    st.markdown(f"**county_list:** `{parse_list_col(row['county_list'])}`")
    st.markdown(f"**city_list:** `{parse_list_col(row['city_list'])}`")
    st.markdown(f"**tribe_list:** `{parse_list_col(row['tribe_list'])}`")
    with st.expander("source_url"):
        st.write(row["source_url"])

    st.divider()
    chunk_count = len(st.session_state.chunks)
    st.metric("Chunks in session", chunk_count)

# ── Main area — chunk entry ────────────────────────────────────────────────────
col_form, col_list = st.columns([1, 1], gap="large")

with col_form:
    st.subheader("Add New Chunk")

    fk = st.session_state.form_key
    section_key = f"section_{fk}"
    ptype_key   = f"page_type_{fk}"
    text_key    = f"text_{fk}"

    section = st.text_input("Section name *", key=section_key,
                             placeholder="e.g. Executive Summary")

    # ── Pages widget ────────────────────────────────────────────────────────────
    st.markdown("**Pages** (optional)")
    ps_disabled_minus = not st.session_state.page_start or st.session_state.page_start <= 1
    pe_disabled_minus = (not st.session_state.page_end or
                         st.session_state.page_end <= (st.session_state.page_start or 1))

    pc = st.columns([0.10, 0.22, 0.10, 0.06, 0.10, 0.22, 0.10, 0.10])
    with pc[0]:
        st.button("−", key="ps_minus_btn", on_click=_ps_minus, disabled=ps_disabled_minus)
    with pc[1]:
        st.text_input("start", key="ps_input", label_visibility="collapsed",
                      placeholder="from", on_change=_on_ps_change)
    with pc[2]:
        st.button("+", key="ps_plus_btn", on_click=_ps_plus)
    with pc[3]:
        st.markdown("<div style='padding-top:0.45rem;text-align:center'>–</div>",
                    unsafe_allow_html=True)
    with pc[4]:
        st.button("−", key="pe_minus_btn", on_click=_pe_minus, disabled=pe_disabled_minus)
    with pc[5]:
        st.text_input("end", key="pe_input", label_visibility="collapsed",
                      placeholder="to", on_change=_on_pe_change)
    with pc[6]:
        st.button("+", key="pe_plus_btn", on_click=_pe_plus)

    # ── Page type + Is appendix ─────────────────────────────────────────────────
    meta_cols = st.columns([1, 1])
    with meta_cols[0]:
        page_type = st.selectbox("Page type", PAGE_TYPES, key=ptype_key)
    with meta_cols[1]:
        is_appendix = st.checkbox("Is appendix", key="sticky_appendix")

    text = st.text_area(
        "Text content *",
        key=text_key,
        height=300,
        placeholder="Paste the chunk text here...",
    )

    # Live token count
    token_count = count_tokens(text) if text else 0
    color = "green" if token_count <= 1200 else ("orange" if token_count <= 1500 else "red")
    st.markdown(f"**Tokens:** :{color}[{token_count}]")

    if st.button("➕ Add Chunk", type="primary", use_container_width=True):
        if not section.strip():
            st.error("Section name is required.")
        elif not text.strip():
            st.error("Text content is required.")
        else:
            ps = st.session_state.page_start
            pe = st.session_state.page_end
            if ps is None:
                pages_str = ""
            elif pe is None or ps == pe:
                pages_str = str(ps)
            else:
                pages_str = f"{ps} - {pe}"

            chunk = make_chunk(row, section.strip(), text.strip(),
                               pages_str, page_type, is_appendix)
            st.session_state.chunks.append(chunk)
            st.success(f"Chunk #{len(st.session_state.chunks)} added  ({chunk['tokens']} tokens)")
            # Clear only section + text; pages and appendix persist
            st.session_state.form_key += 1
            st.rerun()

# ── Right column — chunk list & export ────────────────────────────────────────
with col_list:
    st.subheader(f"Chunks ({len(st.session_state.chunks)})")

    if not st.session_state.chunks:
        st.info("No chunks yet. Add some using the form on the left.")
    else:
        # Summary table
        summary = [
            {
                "#": i + 1,
                "Section": c["section"],
                "Tokens": c["tokens"],
                "Pages": c["pages"],
                "Doc": c["doc_title"],
            }
            for i, c in enumerate(st.session_state.chunks)
        ]
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Individual chunk preview + edit
        with st.expander("Preview chunk content"):
            preview_idx = st.number_input(
                "Chunk #", min_value=1, max_value=len(st.session_state.chunks), value=1
            ) - 1
            c = st.session_state.chunks[preview_idx]

            e_section = st.text_input("Section", value=c["section"], key=f"e_section_{preview_idx}")
            e_text    = st.text_area("Text",    value=c["text"],    height=200, key=f"e_text_{preview_idx}")

            ecols = st.columns([1, 1, 1])
            with ecols[0]:
                e_pages = st.text_input("Pages", value=c["pages"], key=f"e_pages_{preview_idx}")
            with ecols[1]:
                cur_type_idx = PAGE_TYPES.index(c["page_type"]) if c["page_type"] in PAGE_TYPES else 0
                e_page_type  = st.selectbox("Page type", PAGE_TYPES, index=cur_type_idx, key=f"e_ptype_{preview_idx}")
            with ecols[2]:
                e_is_appendix = st.checkbox("Is appendix", value=c["is_appendix"], key=f"e_appendix_{preview_idx}")

            if st.button("💾 Save changes", key=f"e_save_{preview_idx}", use_container_width=True):
                st.session_state.chunks[preview_idx].update({
                    "section":      e_section.strip(),
                    "text":         e_text.strip(),
                    "tokens":       count_tokens(e_text.strip()),
                    "pages":        e_pages.strip(),
                    "page_type":    e_page_type,
                    "is_appendix":  e_is_appendix,
                    "section_path": [e_section.strip()],
                })
                st.success("Chunk updated.")
                st.rerun()

        st.divider()

        # Delete controls
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            if st.button("🗑️ Delete last chunk", use_container_width=True):
                removed = st.session_state.chunks.pop()
                st.warning(f"Deleted: \"{removed['section']}\"")
                st.rerun()
        with dcol2:
            if st.button("🗑️ Clear all chunks", use_container_width=True):
                st.session_state.chunks = []
                st.rerun()

        # Export
        st.divider()
        jsonl_str = "\n".join(
            json.dumps(c, ensure_ascii=False) for c in st.session_state.chunks
        )
        default_filename = f"manual_chunks_{row['doc_title'][:30]}.jsonl"
        st.download_button(
            label="⬇️ Export JSONL",
            data=jsonl_str.encode("utf-8"),
            file_name=default_filename,
            mime="application/jsonl",
            type="primary",
            use_container_width=True,
        )
        st.caption(f"Will save as: `{default_filename}`")
