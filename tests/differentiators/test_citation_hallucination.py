"""
Citation Hallucination Test for Vanilla LLMs

Proves that Vanilla LLMs frequently hallucinate sources when asked
to back up their medical claims with a URL.

Queries the OpenAI/Anthropic model with the standard 100 claims and explicitly 
asks for a verifiable URL. The script then attempts to ping the URL to 
see if it actually exists or if it returns a 404 (hallucination).

Requires: OPENROUTER_API_KEY in environment.
Usage: python tests/differentiators/test_citation_hallucination.py
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from urllib.parse import urlparse

CLAIMS_FILE = Path(__file__).parent.parent / "quality" / "test_claims.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"
TIMEOUT = 30
REQUEST_DELAY = 1.0  # seconds to wait between LLM calls to avoid rate limits

def extract_urls(text: str) -> list[str]:
    """Extract all HTTP/HTTPS URLs from the LLM response text."""
    # This regex looks for URLs starting with http:// or https://
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    urls = url_pattern.findall(text)
    
    # Clean up trailing punctuation often included by mistake
    clean_urls = []
    for url in urls:
        url = url.rstrip(').,"\'\]')
        clean_urls.append(url)
        
    return list(set(clean_urls))

def check_url_exists(url: str, client: httpx.Client) -> bool:
    """
    Ping the URL to see if it's real.
    Returns True if it exists (200 OK, or sometimes 403 Forbidden which means it's a real server).
    Returns False if it's a 404, DNS error, etc. (Hallucination).
    """
    try:
        # Use a realistic User-Agent as some medical sites block scripts
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        # Follow redirects, but don't download the whole body (just HEAD if possible, but GET is safer for anti-bot)
        r = client.get(url, headers=headers, follow_redirects=True, timeout=10.0)
        
        # 404 means the page definitely doesn't exist (hallucination)
        # 403 or 401 might mean the server exists but blocks scrapers, which implies the domain is real
        if r.status_code == 404:
            return False
            
        # If it's 200, it's definitely real
        if r.status_code < 400:
            return True
            
        # Treat other 4xx/5xx as potentially hallucinative if the domain itself is weird, 
        # but let's be generous and say if the server responded, it's not a *total* hallucination.
        # But if it's a 404, the specific article was hallucinated.
        if r.status_code in [403, 401, 405, 429]:
            return True
            
        return False
        
    except httpx.ConnectError:
        # DNS failure -> Hallucinated domain
        return False
    except httpx.TimeoutException:
        # Timeout could be a real slow site, but for our metrics it's unverified
        return False
    except Exception as e:
        return False


def query_vanilla_llm(claim: str, api_key: str, client: httpx.Client, model: str = "openai/gpt-4o-mini") -> str | None:
    """Send claim to OpenRouter and explicitly demand a URL."""
    prompt = (
        "You are a medical fact-checker. Evaluate this health claim:\n"
        f"Claim: {claim}\n\n"
        "1. Provide a brief verdict.\n"
        "2. You MUST provide at least one specific, verifiable URL (starting with https://) "
        "to a source that supports your verdict. Do not provide a generic homepage URL, provide the specific article URL."
    )

    try:
        r = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        pass
    return None

def run_hallucination_test(model: str = "openai/gpt-4o-mini", limit: int | None = None) -> dict:
    from dotenv import load_dotenv
    # Look for .env in the project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
    
    api_key = os.environ.get("APP_OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or APP_OPENROUTER_API_KEY env var")
        sys.exit(1)

    with open(CLAIMS_FILE) as f:
        test_claims = json.load(f)

    if limit:
        test_claims = test_claims[:limit]

    print(f"\n{'='*70}")
    print(f"  Citation Hallucination Test")
    print(f"  Vanilla LLM: {model} (OpenRouter)")
    print(f"  Claims: {len(test_claims)}")
    print(f"{'='*70}\n")

    client = httpx.Client(timeout=TIMEOUT)
    
    total_claims_with_urls = 0
    total_urls_provided = 0
    total_urls_hallucinated = 0
    
    results_details = []

    for i, claim in enumerate(test_claims, 1):
        print(f"[{i:2d}/{len(test_claims)}] {claim['claim'][:50]}...", end=" ", flush=True)

        response_text = query_vanilla_llm(claim["claim"], api_key, client, model)
        
        if not response_text:
            print("⚠️ API Error")
            continue
            
        urls = extract_urls(response_text)
        
        if not urls:
            print("⚠️ No URL provided by LLM")
            results_details.append({
                "claim": claim["claim"],
                "urls": [],
                "hallucinated": [],
                "error": "No URL provided"
            })
            time.sleep(REQUEST_DELAY)
            continue
            
        total_claims_with_urls += 1
        total_urls_provided += len(urls)
        
        hallucinated_this_claim = []
        for url in urls:
            # Check if it's a known generic URL that isn't really a citation
            parsed = urlparse(url)
            if parsed.path in ["", "/", "/pubmed/", "/pubmed"]:
                # They provided a generic homepage instead of a specific citation
                hallucinated_this_claim.append(url)
                continue
                
            is_real = check_url_exists(url, client)
            if not is_real:
                hallucinated_this_claim.append(url)
                
        total_urls_hallucinated += len(hallucinated_this_claim)
        
        if hallucinated_this_claim:
            print(f"❌ HALLUCINATION ({len(hallucinated_this_claim)}/{len(urls)})")
            for h in hallucinated_this_claim:
                print(f"       -> {h}")
        else:
            print(f"✅ Real URLs ({len(urls)})")
            
        results_details.append({
            "claim": claim["claim"],
            "urls": urls,
            "hallucinated": hallucinated_this_claim,
            "all_real": len(hallucinated_this_claim) == 0
        })
        
        time.sleep(REQUEST_DELAY)

    client.close()

    hallucination_rate = (total_urls_hallucinated / total_urls_provided * 100) if total_urls_provided > 0 else 0

    print(f"\n{'='*70}")
    print(f"  RESULTS: {model}")
    print(f"{'='*70}")
    print(f"  Total URLs Checked:     {total_urls_provided}")
    print(f"  Real URLs:              {total_urls_provided - total_urls_hallucinated}")
    print(f"  Hallucinated (404/Fake):{total_urls_hallucinated}")
    print(f"  Hallucination Rate:     {hallucination_rate:.1f}%")
    print(f"  Status: {'✅ PASS' if hallucination_rate == 0 else '❌ FAIL'}  (target: 0% hallucinations)")
    print(f"{'='*70}\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = run_dir / "7_citation_hallucination.json"
    
    summary = {
        "test": "Citation Hallucination (Vanilla LLM)",
        "model": model,
        "total_claims": len(test_claims),
        "claims_with_urls": total_claims_with_urls,
        "total_urls_checked": total_urls_provided,
        "total_urls_hallucinated": total_urls_hallucinated,
        "hallucination_rate_pct": hallucination_rate,
        "pass": hallucination_rate == 0,
        "target": "0% hallucinations",
        "details": results_details
    }
    
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Saved detailed results to {output_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Citation Hallucination Test for Vanilla LLMs")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="OpenRouter model ID")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims to test")
    args = parser.parse_args()
    
    run_hallucination_test(args.model, args.limit)
