"""
download_unstructured.py
========================
Reads unstructured_pdf.csv, attempts to download any direct-PDF links,
saves files under the same directory structure as policy_metadata_3.csv,
and writes unstructured_metadata.csv with the results.

For links that are NOT downloadable PDFs (web pages, Scribd, etc.),
the row is still written to unstructured_metadata.csv with file_path=''
so you can manually download and fill in the path later.

Usage:
    python download_unstructured.py

Output:
    pdf_data/metadata/unstructured_metadata.csv
    pdf_data/data/{State}/{Level}/{filename}.pdf  (for successful downloads)
"""

import os
import re
import csv
import hashlib
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse, unquote
from typing import Optional

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DATA_DIR = "E:/2026_capstone/policy_data/pdf_data/data"
INPUT_CSV     = "E:/2026_capstone/policy_data/pdf_data/metadata/unstructured_pdf.csv"
OUTPUT_CSV    = "E:/2026_capstone/policy_data/pdf_data/metadata/unstructured_metadata.csv"

# ── download settings ──────────────────────────────────────────────────────
TIMEOUT_SEC    = 30
RETRY_COUNT    = 2
DELAY_BETWEEN  = 1.5   # seconds between requests (be polite)

# Domains that are known to NOT serve raw PDFs even with .pdf in URL
# (redirect to viewer, require login, etc.)
NON_DIRECT_DOMAINS = {
    "scribd.com",
    "drive.google.com",
    "docs.google.com",
    "dropbox.com",
    "onedrive.live.com",
    "sharepoint.com",
    "nafws.org",          # news article page
    "fernandez.house.gov",
    "vasquez.house.gov",
    "peoriatribe.com",
    "shawnee-nsn.gov",
    "toolkit.climate.gov",
    "taosunited.org",
    "cochiti.org",
    "nambepueblo.org",
    "sfpueblo.com",
    "lagunapueblo-nsn.gov",
    "fortsillapache-nsn.gov",
    "lcecnet.com",
    "sanidecp.org",
    "santafenm.gov",       # may redirect
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


# ── helpers ─────────────────────────────────────────────────────────────────
def policy_id_from_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def safe_filename(name: str, max_len: int = 80) -> str:
    name = unquote(name)
    name = re.sub(r"[^\w\-\. ]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_. ")
    return name[:max_len]


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


def looks_like_direct_pdf(url: str) -> bool:
    """Heuristic: URL ends with .pdf (after stripping query/fragment) and domain is not blocked."""
    if not url or not url.startswith("http"):
        return False
    dom = domain_of(url)
    if any(dom == d or dom.endswith("." + d) for d in NON_DIRECT_DOMAINS):
        return False
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def level_str(policy_level: str) -> str:
    """Normalise policy level to directory name."""
    lvl = (policy_level or "").strip().title()
    mapping = {
        "Federal": "Federal",
        "State": "State",
        "County": "County",
        "City": "City",
        "Tribe": "Tribe",
        "Federal-Tribal Collab": "Tribe",
        "Federal-Tribal": "Tribe",
    }
    return mapping.get(lvl, lvl or "Other")


def state_dir(state: str) -> str:
    """Map state name to directory name."""
    s = (state or "").strip().title()
    mapping = {
        "New Mexico": "New Mexico",
        "Arizona": "Arizona",
        "Oklahoma": "Oklahoma",
    }
    return mapping.get(s, s or "Other")


def download_pdf(url: str, dest_path: str) -> bool:
    """Download url to dest_path. Returns True on success."""
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                content_type = resp.headers.get("Content-Type", "")
                # Accept only if content-type hints at PDF or is generic binary
                if "text/html" in content_type.lower():
                    print(f"  [SKIP] HTML response (not a PDF): {url}")
                    return False
                data = resp.read()
            # Verify PDF magic bytes
            if not data[:4] == b"%PDF":
                print(f"  [SKIP] Response is not a PDF (no %PDF header): {url}")
                return False
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except urllib.error.HTTPError as e:
            print(f"  [HTTP {e.code}] attempt {attempt}/{RETRY_COUNT}: {url}")
        except urllib.error.URLError as e:
            print(f"  [URLError] attempt {attempt}/{RETRY_COUNT}: {e.reason} — {url}")
        except Exception as e:
            print(f"  [Error] attempt {attempt}/{RETRY_COUNT}: {e} — {url}")
        if attempt < RETRY_COUNT:
            time.sleep(DELAY_BETWEEN)
    return False


def extract_urls(raw_link_field: str) -> list[str]:
    """A single cell may contain multiple newline-separated URLs."""
    urls = []
    for part in raw_link_field.splitlines():
        part = part.strip()
        if part.startswith("http"):
            urls.append(part)
    return urls if urls else ([raw_link_field.strip()] if raw_link_field.strip().startswith("http") else [])


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    # Read input CSV
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows from {INPUT_CSV}")

    output_rows = []

    for row in rows:
        state      = row.get("State", "").strip()
        county     = row.get("County", "").strip()
        city       = row.get("City", "").strip()
        tribe      = row.get("Tribe Name", "").strip()
        policy_name = row.get("Policy Name", "").strip()
        policy_level = row.get("Policy Level (Federal/State/County/City)", "").strip()
        policy_type  = row.get("Policy Type (Climate/Energy/Weather/etc.)", "").strip()
        raw_links  = row.get("Links to PDF", "").strip()

        urls = extract_urls(raw_links)

        # Try each URL for this row
        downloaded_path = ""
        attempted_url   = ""
        download_status = "web_only"  # default: no downloadable PDF found

        for url in urls:
            if not url:
                continue
            attempted_url = url

            if not looks_like_direct_pdf(url):
                print(f"[WEB]  {url}")
                download_status = "web_only"
                continue

            # Build destination path
            lvl = level_str(policy_level or ("Tribe" if tribe else "County" if county else "City" if city else "State"))
            sdir = state_dir(state)
            pid  = policy_id_from_url(url)[:10]

            # Derive filename from URL
            url_path = urlparse(url).path
            url_fname = os.path.basename(url_path)
            url_fname = safe_filename(url_fname) or "document"
            if not url_fname.lower().endswith(".pdf"):
                url_fname += ".pdf"
            dest_filename = f"{pid}__{url_fname}"
            dest_path = os.path.join(BASE_DATA_DIR, sdir, lvl, dest_filename)
            dest_path = dest_path.replace("\\", "/")

            print(f"[DL]   {url}")
            print(f"       → {dest_path}")

            if os.path.exists(dest_path):
                print(f"       (already exists, skipping download)")
                downloaded_path = dest_path
                download_status = "already_exists"
                break

            success = download_pdf(url, dest_path)
            time.sleep(DELAY_BETWEEN)

            if success:
                downloaded_path = dest_path
                download_status = "downloaded"
                print(f"       ✓ saved ({os.path.getsize(dest_path):,} bytes)")
                break
            else:
                download_status = "failed"

        # Build output metadata row (mirrors policy_metadata_3.csv schema)
        state_list   = f"['{state}']"   if state  else "[]"
        county_list  = f"['{county}']"  if county else "[]"
        city_list    = f"['{city}']"    if city   else "[]"
        tribe_list   = f"['{tribe}']"   if tribe  else "[]"

        # policy_id: use hash of the first URL
        first_url = urls[0] if urls else ""
        pid_full  = policy_id_from_url(first_url) if first_url else ""

        output_rows.append({
            "row_index":      len(output_rows),
            "policy_id":      pid_full,
            "source_url":     attempted_url or first_url,
            "all_urls":       " | ".join(urls),
            "file_path":      downloaded_path,   # blank = needs manual download
            "state_list":     state_list,
            "county_list":    county_list,
            "city_list":      city_list,
            "tribe_list":     tribe_list,
            "policy_level":   policy_level,
            "policy_type":    policy_type,
            "policy_name":    policy_name,
            "download_status": download_status,  # downloaded / already_exists / failed / web_only
        })

    # Write output CSV
    fieldnames = [
        "row_index", "policy_id", "source_url", "all_urls", "file_path",
        "state_list", "county_list", "city_list", "tribe_list",
        "policy_level", "policy_type", "policy_name", "download_status",
    ]
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # Summary
    downloaded   = sum(1 for r in output_rows if r["download_status"] == "downloaded")
    already      = sum(1 for r in output_rows if r["download_status"] == "already_exists")
    failed       = sum(1 for r in output_rows if r["download_status"] == "failed")
    web_only     = sum(1 for r in output_rows if r["download_status"] == "web_only")
    need_manual  = sum(1 for r in output_rows if not r["file_path"])

    print("\n" + "=" * 60)
    print(f"Done. Results written to: {OUTPUT_CSV}")
    print(f"  Downloaded:      {downloaded}")
    print(f"  Already existed: {already}")
    print(f"  Failed:          {failed}")
    print(f"  Web only (no PDF link): {web_only}")
    print(f"  Need manual download:   {need_manual}  ← fill file_path manually")
    print("=" * 60)


if __name__ == "__main__":
    main()
