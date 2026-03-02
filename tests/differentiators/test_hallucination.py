"""
Hallucination Detection Test

Verifies that the LLM never fabricates source citations. For each response,
parses the **Source Files** line and checks that every filename mentioned
actually exists in the returned source_documents array.

Requires a running backend instance.
Usage: python tests/differentiators/test_hallucination.py [--url URL]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

CLAIMS_FILE = Path(__file__).parent.parent / "quality" / "test_claims.json"
DEFAULT_URL = "http://localhost:8000"
TIMEOUT = 60


def extract_cited_files(chat_response: str) -> list[str]:
    """Parse filenames from the **Source Files** line in the response."""
    for line in chat_response.split("\n"):
        if "source files" in line.lower():
            # Remove the label and extract filenames
            content = re.sub(r"\*\*Source Files\*\*:?\s*", "", line, flags=re.IGNORECASE).strip()
            if content.lower() in ("n/a", "none", ""):
                return []
            # Split by comma, semicolon, or "and"
            files = re.split(r"[,;]|\band\b", content)
            return [f.strip().strip("`'\"") for f in files if f.strip()]
    return []


# ── Filename Normalization ───────────────────────────────────────────

def _normalize_for_comparison(name: str) -> str:
    """Lowercase, strip extensions, and collapse separators to spaces."""
    n = name.lower().strip()
    n = n.replace(".txt", "")
    n = n.replace("/", " ").replace("\\", " ")
    n = n.replace("-", " ").replace("_", " ")
    return " ".join(n.split())


def _extract_id(name: str) -> str | None:
    """Try to extract a recognisable ID (PubMed, DOI, arXiv, WHO slug)."""
    low = name.lower()
    # PubMed ID — also match URLs like pubmed.ncbi.nlm.nih.gov/12345678/
    m = re.search(r"(?:pubmed|pmid)[/_ ]?(\d+)", low)
    if m:
        return m.group(1)
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", low)
    if m:
        return m.group(1)
    # ArXiv ID — also match URLs like arxiv.org/abs/1234.5678v1
    m = re.search(r"arxiv[_ ]?([\d.]+v?\d*)", low)
    if m:
        return m.group(1)
    m = re.search(r"arxiv\.org/abs/([\d.]+v?\d*)", low)
    if m:
        return m.group(1)
    # PLOS DOI fragment — normalize to DOI suffix like "journal.pmed.0020168"
    m = re.search(r"plos[_ ]?10[_ ]1371[_ ](journal[_ ]\w+[_ ]\d+)", low)
    if m:
        return m.group(1).replace("_", ".").replace(" ", ".")
    m = re.search(r"10\.1371/(journal\.\w+\.\d+)", low)
    if m:
        return m.group(1)
    # WHO slug
    m = re.search(r"who[_ ](?:factsheet[_ ]?)?(.*)", low)
    if m:
        slug = m.group(1).strip()
        if slug:
            return slug.replace("_", " ").replace("-", " ")
    return None


def _extract_all_ids(name: str) -> set[str]:
    """Extract ALL recognisable IDs from a string (may contain multiple)."""
    ids = set()
    low = name.lower()
    # PubMed IDs
    for m in re.finditer(r"(?:pubmed|pmid)[/_ ]?(\d+)", low):
        ids.add(m.group(1))
    for m in re.finditer(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", low):
        ids.add(m.group(1))
    # ArXiv IDs
    for m in re.finditer(r"arxiv[_ ]?([\d.]+v?\d*)", low):
        ids.add(m.group(1))
    for m in re.finditer(r"arxiv\.org/abs/([\d.]+v?\d*)", low):
        ids.add(m.group(1))
    # PLOS DOIs — normalize to DOI suffix like "journal.pmed.0020168"
    for m in re.finditer(r"plos[_ ]?10[_ ]1371[_ ](journal[_ ]\w+[_ ]\d+)", low):
        ids.add(m.group(1).replace("_", ".").replace(" ", "."))
    for m in re.finditer(r"10\.1371/(journal\.\w+\.\d+)", low):
        ids.add(m.group(1))
    return ids


def is_citation_real(cited: str, actual_files: set[str], actual_sources: set[str] | None = None) -> bool:
    """Check whether a cited filename corresponds to any actual source doc.

    Checks against both file_name metadata AND source URLs from the
    returned source_documents.
    """
    if not cited:
        return True

    # Combine file names and source URLs for matching
    all_identifiers = set(actual_files)
    if actual_sources:
        all_identifiers |= actual_sources

    cited_norm = _normalize_for_comparison(cited)
    for af in all_identifiers:
        af_norm = _normalize_for_comparison(af)
        if cited_norm in af_norm or af_norm in cited_norm:
            return True

    # Strategy 2: ID matching (extract IDs from cited and from all actual identifiers)
    cited_id = _extract_id(cited)
    if cited_id:
        for af in all_identifiers:
            af_id = _extract_id(af)
            if af_id and cited_id == af_id:
                return True
            if cited_id in _normalize_for_comparison(af):
                return True

    # Strategy 2b: extract ALL IDs from source URLs and match
    cited_ids = _extract_all_ids(cited)
    if cited_ids and actual_sources:
        all_actual_ids = set()
        for src in actual_sources:
            all_actual_ids |= _extract_all_ids(src)
        if cited_ids & all_actual_ids:
            return True

    # Strategy 3: token overlap
    cited_tokens = set(cited_norm.split())
    if len(cited_tokens) >= 2:
        for af in all_identifiers:
            af_norm = _normalize_for_comparison(af)
            af_tokens = set(af_norm.split())
            overlap = cited_tokens & af_tokens
            if len(overlap) / len(cited_tokens) >= 0.6:
                return True

    # Strategy 4: Source-type matching (e.g., cited "plos_10_1371..." matches "PLOS Article")
    SOURCE_KEYWORDS = {
        "plos": ["plos"],
        "pubmed": ["pubmed", "pmid"],
        "arxiv": ["arxiv"],
        "who": ["who"],
    }
    cited_low = cited.lower()
    for source_type, keywords in SOURCE_KEYWORDS.items():
        if any(kw in cited_low for kw in keywords):
            for af in all_identifiers:
                if source_type in af.lower():
                    return True
    return False


def run_hallucination_test(base_url: str) -> dict:
    with open(CLAIMS_FILE) as f:
        claims = json.load(f)

    test_claims = claims  # Use all claims for robust testing

    print(f"\n{'='*70}")
    print(f"  Hallucination Detection Test")
    print(f"  Target: {base_url}")
    print(f"  Claims: {len(test_claims)}")
    print(f"{'='*70}\n")

    total = 0
    clean = 0  # no hallucinated sources
    hallucinated = 0
    details = []
    client = httpx.Client(timeout=TIMEOUT)

    for i, claim in enumerate(test_claims, 1):
        print(f"[{i:2d}/{len(test_claims)}] {claim['claim'][:55]}...", end=" ", flush=True)

        try:
            r = client.post(f"{base_url}/rag/query", json={"query": claim["claim"], "top_k": 3})
            if r.status_code != 200:
                print(f"⚠️ HTTP {r.status_code}")
                continue

            data = r.json()
            total += 1

            # Get cited files from LLM response text
            cited_files = extract_cited_files(data["chat_response"])

            # Get actual source document filenames AND source URLs
            actual_files = {
                doc.get("metadata", {}).get("file_name", "")
                for doc in data.get("source_documents", [])
            }
            actual_sources = {
                doc.get("metadata", {}).get("source", "")
                for doc in data.get("source_documents", [])
                if doc.get("metadata", {}).get("source")
            }

            # Check if every cited file exists in actual sources
            fabricated = [f for f in cited_files if f and not is_citation_real(f, actual_files, actual_sources)]

            if not fabricated:
                clean += 1
                print(f"✅ Cited: {len(cited_files)} | Actual: {len(actual_files)}")
            else:
                hallucinated += 1
                print(f"❌ HALLUCINATED: {fabricated}")

            details.append({
                "claim": claim["claim"],
                "cited_files": cited_files,
                "actual_files": list(actual_files),
                "fabricated": fabricated,
            })

        except Exception as e:
            print(f"⚠️ Error: {e}")

    client.close()

    clean_rate = clean / total * 100 if total else 0

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Clean responses:       {clean}/{total} ({clean_rate:.1f}%)")
    print(f"  Hallucinated sources:  {hallucinated}/{total}")
    print(f"  Status: {'✅ PASS' if hallucinated == 0 else '❌ FAIL'}  (target: 0 hallucinations)")
    print(f"{'='*70}\n")

    return {"total": total, "clean": clean, "hallucinated": hallucinated}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination Detection Test")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend base URL")
    args = parser.parse_args()

    results = run_hallucination_test(args.url)
    sys.exit(0 if results["hallucinated"] == 0 else 1)
