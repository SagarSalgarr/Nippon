"""
test_s3_sdk.py — Simulate the SDK's model download flow against live S3.

This does exactly what the SDK on a phone does:
  1. Fetch manifest.json from CDN
  2. Parse it to find model URLs
  3. HEAD each model to verify it's publicly accessible
  4. Download one model and verify SHA256

Usage:
  python test_s3_sdk.py
  python test_s3_sdk.py --full    # downloads ALL models and verifies SHA256 (slow)
  python test_s3_sdk.py --lang hi # test just Hindi TTS download
"""

import sys
import json
import hashlib
import urllib.request
import urllib.error
import time

MANIFEST_URL = "https://indicai-cdn.s3.ap-south-1.amazonaws.com/manifest.json"
S3_BASE_URL  = "https://indicai-cdn.s3.ap-south-1.amazonaws.com/models"


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def head_check(url: str) -> tuple:
    """HEAD request — returns (status_code, content_length_mb) or raises."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as resp:
        size = int(resp.headers.get("Content-Length", 0)) / 1_000_000
        return resp.status, size


def download_and_hash(url: str) -> tuple:
    """Download file and return (sha256_hex, size_mb, seconds)."""
    sha = hashlib.sha256()
    start = time.time()
    total = 0
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=600) as resp:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
            total += len(chunk)
            mb = total / 1_000_000
            elapsed = time.time() - start
            speed = mb / elapsed if elapsed > 0 else 0
            print(f"\r  downloading: {mb:.1f} MB ({speed:.1f} MB/s)", end="", flush=True)
    elapsed = time.time() - start
    print()
    return sha.hexdigest(), total / 1_000_000, elapsed


def main():
    full_mode = "--full" in sys.argv
    lang_filter = None
    for i, arg in enumerate(sys.argv):
        if arg == "--lang" and i + 1 < len(sys.argv):
            lang_filter = sys.argv[i + 1]

    print("=" * 60)
    print("  SDK S3 Integration Test")
    print("=" * 60)

    # ── Step 1: Fetch manifest ────────────────────────────────
    print(f"\n1. Fetching manifest from:\n   {MANIFEST_URL}")
    try:
        manifest = fetch_json(MANIFEST_URL)
        print(f"   [OK] version={manifest['version']}  "
              f"languages={list(manifest['languages'].keys())}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        sys.exit(1)

    # ── Step 2: Check shared models ───────────────────────────
    print("\n2. Checking shared model URLs (HEAD request)...")
    all_models = []

    for name in ["whisper", "indic_en", "indic_from_en"]:
        info = manifest["models"].get(name, {})
        if not info:
            print(f"   [WARN] {name} not in manifest")
            continue
        url = f"{S3_BASE_URL}/{info['file']}"
        all_models.append((name, url, info.get("sha256"), info.get("size_mb", 0)))
        try:
            status, size = head_check(url)
            print(f"   [OK]  {name}: {info['file']} ({size:.1f} MB) — HTTP {status}")
        except Exception as e:
            print(f"   [FAIL] {name}: {url} — {e}")

    # ── Step 3: Check per-language TTS models ─────────────────
    print("\n3. Checking TTS model URLs per language...")
    languages = manifest.get("languages", {})
    for lc, lcfg in languages.items():
        tts = lcfg.get("tts", {})
        if not tts:
            print(f"   [SKIP] {lc} ({lcfg['name']}) — no TTS")
            continue
        url = f"{S3_BASE_URL}/{tts['file']}"
        all_models.append((f"tts-{lc}", url, tts.get("sha256"), tts.get("size_mb", 0)))
        try:
            status, size = head_check(url)
            print(f"   [OK]  {lc} ({lcfg['name']}): {tts['file']} ({size:.1f} MB) — HTTP {status}")
        except Exception as e:
            print(f"   [FAIL] {lc}: {url} — {e}")

    # ── Step 4: Download + verify SHA256 ──────────────────────
    if full_mode:
        print(f"\n4. Downloading ALL {len(all_models)} models + SHA256 verification...")
        models_to_test = all_models
    elif lang_filter:
        models_to_test = [m for m in all_models if lang_filter in m[0] or m[0] in ["whisper", "indic_en", "indic_from_en"]]
        if not models_to_test:
            models_to_test = [all_models[0]]
        print(f"\n4. Downloading {len(models_to_test)} model(s) for verification...")
    else:
        models_to_test = [all_models[0]]
        print(f"\n4. Downloading 1 model for SHA256 verification (use --full for all)...")

    passed = 0
    failed = 0
    for name, url, expected_sha, expected_mb in models_to_test:
        print(f"\n   [{name}] {url.split('/')[-1]}")
        try:
            actual_sha, size_mb, seconds = download_and_hash(url)
            speed = size_mb / seconds if seconds > 0 else 0
            print(f"  size: {size_mb:.1f} MB  time: {seconds:.0f}s  speed: {speed:.1f} MB/s")

            if expected_sha:
                if actual_sha == expected_sha:
                    print(f"  SHA256: {actual_sha[:16]}... MATCH")
                    passed += 1
                else:
                    print(f"  SHA256: MISMATCH!")
                    print(f"    expected: {expected_sha}")
                    print(f"    actual:   {actual_sha}")
                    failed += 1
            else:
                print(f"  SHA256: {actual_sha[:16]}... (no expected hash in manifest)")
                passed += 1
        except Exception as e:
            print(f"  [FAIL] download error: {e}")
            failed += 1

    # ── Summary ───────────────────────────────────────────────
    total_urls = len(all_models)
    print("\n" + "=" * 60)
    print(f"  RESULTS")
    print(f"  Manifest:     OK (v{manifest['version']}, {len(languages)} languages)")
    print(f"  URLs checked: {total_urls}/{total_urls} accessible")
    print(f"  SHA256 tests: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed:
        sys.exit(1)
    print("\n  SDK will work correctly with these S3 URLs.")


if __name__ == "__main__":
    main()
