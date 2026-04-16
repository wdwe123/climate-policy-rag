"""
Climate_Policy_RAG_Test.py — Climate Policy RAG Test Interface (Streamlit)

Run:
    $env:PORTKEY_API_KEY="..."
    streamlit run E:/2026_capstone/policy_data/Climate_Policy_RAG_Test.py
"""
import os, re, sys
import pandas as pd
import streamlit as st
from urllib.parse import unquote

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

POLICY_CSV = os.path.join(_BASE_DIR, "pdf_data", "metadata", "Policy Data Sheet - structured_text.csv")
_IS_CLOUD  = bool(os.environ.get("QDRANT_URL", ""))  # True when running on Streamlit Cloud

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Policy RAG",
    page_icon="🌿",
    layout="wide",
)

# ─── Session state init ───────────────────────────────────────────────────────
if "map_geo" not in st.session_state:
    st.session_state.map_geo    = None
if "map_marker" not in st.session_state:
    st.session_state.map_marker = None
for _k in ("dd_state", "dd_tribe", "dd_county", "dd_city"):
    if _k not in st.session_state:
        st.session_state[_k] = "(Auto)"

# ─── URL → Policy Name lookup ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_url_map() -> dict:
    def norm(url):
        return unquote(url or "").rstrip("/").lower()
    try:
        df = pd.read_csv(POLICY_CSV)
        return {norm(r["Links to PDF"]): r["Policy Name"]
                for _, r in df.iterrows()
                if pd.notna(r.get("Links to PDF")) and pd.notna(r.get("Policy Name"))}
    except Exception:
        return {}

def get_display_name(doc_title, source_url, url_map):
    def norm(url):
        return unquote(url or "").rstrip("/").lower()
    name = url_map.get(norm(source_url))
    if name:
        return name
    clean = re.sub(r'^[a-f0-9]{8,}__', '', doc_title or "")
    return clean.replace("-", " ").replace("_", " ").title()

# ─── Lazy-load modules ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval model...")
def load_retriever():
    import retriever; return retriever

@st.cache_resource(show_spinner="Loading generation model...")
def load_generator():
    import generator; return generator

@st.cache_resource(show_spinner="Loading map data...")
def load_map_utils():
    import map_utils
    map_utils.load_all_layers()
    return map_utils

# ─── Geo helpers ──────────────────────────────────────────────────────────────
def _has_geo(geo):
    return bool(geo and any(geo.get(k) for k in ("tribes", "counties", "cities", "states")))

def geo_summary(geo):
    parts = []
    for k in ("tribes", "counties", "cities", "states"):
        if geo.get(k): parts.append(", ".join(geo[k]))
    return " / ".join(parts) if parts else "none"

# ─── Header ───────────────────────────────────────────────────────────────────
hdr_left, hdr_right = st.columns([3, 1])
with hdr_left:
    st.title("🌿 Climate Policy RAG")
    st.caption("Native American Climate Policy Assistant · AZ / NM / OK")
with hdr_right:
    st.write("")
    api_key = st.text_input("Portkey API Key",
                            value=os.environ.get("PORTKEY_API_KEY", ""),
                            type="password", label_visibility="collapsed",
                            placeholder="Portkey API Key")
    if api_key:
        os.environ["PORTKEY_API_KEY"] = api_key

st.divider()

# ─── Pre-load vocab for dropdowns ─────────────────────────────────────────────
retriever_mod = load_retriever()
mu_mod        = load_map_utils()

state_options  = ["(Auto)", "Arizona", "New Mexico", "Oklahoma"]
tribe_options  = ["(Auto)"] + sorted(retriever_mod.TRIBES)
county_options = ["(Auto)"] + sorted(retriever_mod.COUNTIES)
city_options   = ["(Auto)"] + mu_mod.get_all_cities()

# ══════════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([5, 6], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN: Map + Geo filter
# ─────────────────────────────────────────────────────────────────────────────
with col_left:
    try:
        from streamlit_folium import st_folium
        _folium_ok = True
    except ImportError:
        st.error("streamlit-folium not installed.")
        _folium_ok = False

    if _folium_ok:
        mu = mu_mod

        # Map marker status
        marker_info, marker_clear = st.columns([5, 1])
        with marker_info:
            if st.session_state.map_geo and _has_geo(st.session_state.map_geo):
                st.info(f"📍 **{geo_summary(st.session_state.map_geo)}**  "
                        f"(lat={st.session_state.map_marker[0]:.3f}, "
                        f"lon={st.session_state.map_marker[1]:.3f})")
            else:
                st.caption("Click the map to place a location marker.")
        with marker_clear:
            if st.button("✕ Clear", use_container_width=True):
                st.session_state.map_geo    = None
                st.session_state.map_marker = None
                for k in ("dd_state", "dd_tribe", "dd_county", "dd_city"):
                    st.session_state[k] = "(Auto)"
                st.rerun()

        # Map fragment (only reruns on click, not zoom/pan)
        @st.fragment
        def _map_fragment():
            fmap = mu.build_folium_map(marker_latlon=st.session_state.map_marker)
            result = st_folium(fmap, height=480, use_container_width=True,
                               key="folium_map", returned_objects=["last_clicked"])
            if result and result.get("last_clicked"):
                lat = result["last_clicked"]["lat"]
                lon = result["last_clicked"]["lng"]
                if st.session_state.map_marker != (lat, lon):
                    st.session_state.map_marker = (lat, lon)
                    region = mu.identify_region(lat, lon)
                    st.session_state.map_geo = region
                    _s = ["(Auto)", "Arizona", "New Mexico", "Oklahoma"]
                    st.session_state.dd_state  = region["states"][0]  if region["states"]  and region["states"][0]  in _s else "(Auto)"
                    st.session_state.dd_tribe  = region["tribes"][0]  if region["tribes"]  else "(Auto)"
                    st.session_state.dd_county = region["counties"][0] if region["counties"] else "(Auto)"
                    st.session_state.dd_city   = region["cities"][0]  if region["cities"]  else "(Auto)"
                    if not _has_geo(region):
                        st.warning("Outside coverage area (AZ / NM / OK). Filter not applied.")
                    st.rerun(scope="app")

        _map_fragment()

    # Geo filter dropdowns (2×2 grid)
    st.markdown("**Geographic Filter**")
    st.caption("Syncs with map marker · overrides auto-detection from query")
    g1, g2 = st.columns(2)
    with g1:
        sel_state  = st.selectbox("State",  state_options,  key="dd_state")
        sel_county = st.selectbox("County", county_options, key="dd_county")
    with g2:
        sel_tribe  = st.selectbox("Tribe",  tribe_options,  key="dd_tribe")
        sel_city   = st.selectbox("City",   city_options,   key="dd_city")

    if st.button("Reset All Filters", use_container_width=True):
        for k in ("dd_state", "dd_tribe", "dd_county", "dd_city"):
            st.session_state[k] = "(Auto)"
        st.session_state.map_geo    = None
        st.session_state.map_marker = None
        st.rerun()

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        top_k          = st.slider("Candidates per level (top_k)",      5, 15, 8)
        top_n          = st.slider("Results per level (top_n)",         2, 10, 5)
        chunks_per_lvl = st.slider("Chunks per level sent to LLM",      1, top_n, min(2, top_n))
        st.caption("Models: gemini-embedding-001 · claude-haiku-4-5 (rerank) · claude-sonnet-4-6 (gen)")

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN: Search + Answer
# ─────────────────────────────────────────────────────────────────────────────
with col_right:

    # Build geo override from dropdowns
    def build_dropdown_geo():
        tribes   = [sel_tribe]  if sel_tribe  != "(Auto)" else []
        counties = [sel_county] if sel_county != "(Auto)" else []
        cities   = [sel_city]   if sel_city   != "(Auto)" else []
        states   = [sel_state]  if sel_state  != "(Auto)" else []
        if any([tribes, counties, cities, states]):
            return {"tribes": tribes, "counties": counties, "cities": cities, "states": states}
        return None

    map_geo      = st.session_state.map_geo if _has_geo(st.session_state.map_geo) else None
    dropdown_geo = build_dropdown_geo()

    # Conflict detection
    conflict_msg = None
    if map_geo and dropdown_geo:
        conflict_msg = mu_mod.detect_conflict(map_geo, dropdown_geo)

    if conflict_msg:
        st.error(f"⚠️ Geographic conflict — search disabled.\n\n{conflict_msg}\n\n"
                 "Clear the map marker (✕ Clear) or reset the dropdowns.")
        active_geo = None
        can_search = False
    elif map_geo:
        st.info(f"📍 Using map location: **{geo_summary(map_geo)}**")
        active_geo = map_geo
        can_search = True
    elif dropdown_geo:
        st.info(f"🔽 Using dropdown filter: **{geo_summary(dropdown_geo)}**")
        active_geo = dropdown_geo
        can_search = True
    else:
        st.caption("Geographic filter: auto-detected from query text.")
        active_geo = None
        can_search = True

    # Search input
    query = st.text_input("Enter your query",
                          placeholder="e.g. What flooding plans are available for Apache County?",
                          label_visibility="collapsed")

    btn_col, clr_col = st.columns([2, 1])
    with btn_col:
        search_clicked = st.button("🔍 Search", type="primary",
                                   use_container_width=True, disabled=not can_search)
    with clr_col:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    # ── Search execution ───────────────────────────────────────────────────────
    if search_clicked and query.strip():
        if len(query.strip()) < 5:
            st.warning("Please enter a more specific query.")
            st.stop()
        if not api_key:
            st.error("Please enter your Portkey API Key (top right).")
            st.stop()

        retriever = load_retriever()
        generator = load_generator()
        url_map   = load_url_map()

        with st.spinner("Retrieving and reranking across all geographic levels..."):
            level_results = retriever.search_all_levels(
                query,
                top_k_per_level=top_k,
                top_n_per_level=top_n,
                geo_override=active_geo,
            )

        if not level_results:
            st.warning("No relevant results found.")
            st.stop()

        # Inject display_name into every chunk payload
        for lvl_data in level_results.values():
            for r in lvl_data["chunks"]:
                p = r["payload"]
                p["display_name"] = get_display_name(
                    p.get("doc_title", ""), p.get("source_url", ""), url_map)

        # Debug expander: show geo entities + which levels were searched
        first_chunks = next(
            (d["chunks"] for d in level_results.values() if d["chunks"]), [])
        geo      = first_chunks[0].get("geo", {}) if first_chunks else {}
        geo_tier = first_chunks[0].get("geo_tier", "none") if first_chunks else "none"
        with st.expander("Geographic Entity Detection (debug)", expanded=False):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Start Level", geo_tier)
            c2.metric("Tribe",  ", ".join(geo.get("tribes",   [])) or "—")
            c3.metric("County", ", ".join(geo.get("counties", [])) or "—")
            c4.metric("City",   ", ".join(geo.get("cities",   [])) or "—")
            c5.metric("State",  ", ".join(geo.get("states",   [])) or "—")
            searched = [f"{retriever.LEVEL_LABEL[lv]} ({'✅' if d['has_results'] else '❌'})"
                        for lv, d in level_results.items()]
            st.caption("Levels searched: " + "  →  ".join(searched))

        # Check if any level has results
        any_relevant = any(d["has_results"] for d in level_results.values())
        if not any_relevant:
            st.warning("No relevant policy documents found across any geographic level. "
                       "Try a more specific question.")
        else:
            # Build gen_input: up to chunks_per_lvl chunks per level that has results
            gen_input = []
            for lv in retriever.LEVEL_ORDER:
                lvl_data = level_results.get(lv, {})
                if lvl_data.get("has_results"):
                    gen_input.extend(lvl_data["chunks"][:chunks_per_lvl])
            gen_input = sorted(gen_input,
                               key=lambda x: x.get("cross_score", 0), reverse=True)

            # Build level_summary for generator
            level_summary = {
                lv: {"geo_label": d["geo_label"], "has_results": d["has_results"]}
                for lv, d in level_results.items()
            }

            with st.spinner("Generating answer..."):
                gen_result = generator.generate_answer(
                    query, gen_input, level_summary=level_summary)

            st.subheader("Answer")
            st.markdown(gen_result["answer"])
            st.caption(f"Model: {gen_result['model']} · {gen_result['input_chunks']} chunks")

    elif search_clicked:
        st.warning("Please enter a query.")

# ══════════════════════════════════════════════════════════════════════════════
# FULL-WIDTH: Retrieved Chunks (grouped by geographic level)
# ══════════════════════════════════════════════════════════════════════════════
if search_clicked and query.strip() and "level_results" in dir() and level_results:

    def render_chunk(r, i):
        p      = r["payload"]
        cross  = r.get("cross_score", 0)
        tier   = p.get("retrieval_tier", "—")
        tag    = p.get("primary_tag", "—")
        doc    = p.get("display_name") or p.get("doc_title", "—")
        sec    = p.get("section", "—")
        pages  = p.get("pages", "—")
        url    = p.get("source_url", "")
        path   = (p.get("pdf_path") or p.get("file_path", "—")).replace(
                     "/scratch/xt2284/pdf_data", "E:/2026_capstone/policy_data/pdf_data")
        text   = (p.get("text") or "").strip()
        pol_sc = p.get("policy_score", None)
        tier_color = {"primary":"🟢","secondary":"🟡","low_priority":"🟠","exclude":"🔴"}.get(tier,"⚪")

        with st.expander(f"{tier_color} [{i}] score={cross:.2f}  {tag}  ·  {doc[:70]}",
                         expanded=(i <= 2)):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Relevance Score", f"{cross:.2f}")
            c2.metric("Retrieval Tier",  tier)
            c3.metric("Primary Tag",     tag)
            c4.metric("Policy Score",    f"{pol_sc:.2f}" if pol_sc is not None else "—")
            st.markdown(f"**Document:** {doc}")
            st.markdown(f"**Section:** {sec}　　**Pages:** {pages}")
            if url: st.markdown(f"**URL:** [{url}]({url})")
            if not _IS_CLOUD:
                st.caption(f"Local path: {path}")
            st.markdown("**Text Preview:**")
            st.text(text[:600] + ("..." if len(text) > 600 else ""))

    LEVEL_ICONS = {
        "tribe": "🏘️", "city": "🏙️", "county": "🗺️",
        "state": "📍", "federal": "🏛️",
    }
    total_chunks = sum(len(d["chunks"]) for d in level_results.values())
    st.divider()
    st.subheader(f"Retrieved Chunks ({total_chunks} total across {len(level_results)} levels)")

    chunk_counter = 1
    for lv in retriever.LEVEL_ORDER:
        lvl_data = level_results.get(lv)
        if lvl_data is None:
            continue
        icon      = LEVEL_ICONS.get(lv, "📄")
        label     = retriever.LEVEL_LABEL[lv]
        geo_label = lvl_data["geo_label"]
        chunks    = lvl_data["chunks"]

        if lvl_data["has_results"]:
            st.markdown(f"**{icon} {label} Level — {geo_label}** ✅  ({len(chunks)} results)")
            for r in chunks:
                render_chunk(r, chunk_counter)
                chunk_counter += 1
        else:
            st.markdown(f"**{icon} {label} Level — {geo_label}** — *No relevant policies found at this level.*")