"""
step4_upload.py — Build final manifest.json and upload models + manifest to S3.

Run:  python step4_upload.py
What: 1. Reads draft manifest from ./dist/manifest_draft.json
      2. Adds language metadata from config.LANGUAGES
      3. Uploads all .ort files to S3 (models are immutable — never overwrite)
      4. Atomically updates manifest.json on CDN bucket LAST
         (so no device ever sees a manifest pointing to a file not yet uploaded)

Agnostic: manifest.json is the only thing SDK reads.
          Adding a language → add to config.LANGUAGES, re-run pipeline.
          SDK + APK change nothing.
"""

import sys
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from config import (
    LANGUAGES,
    MODEL_VERSION,
    S3_BUCKET,
    CDN_BUCKET,
    MANIFEST_KEY,
    S3_MODELS_PREFIX,
    DIST_DIR,
)

TRANSFER_CFG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,   # 8 MB — use multipart above this
    multipart_chunksize=8 * 1024 * 1024,   # 8 MB chunks (small enough to survive slow links)
    max_concurrency=2,                      # keep bandwidth manageable
    use_threads=True,
)

MAX_RETRIES = 5


# ── S3 helpers ────────────────────────────────────────────────────────────────

def s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def upload_file_if_new(s3_client, local_path: Path, bucket: str, key: str) -> bool:
    """Upload only if key does not already exist (immutable model files). Retries on timeout."""
    if s3_object_exists(s3_client, bucket, key):
        print(f"  [skip]   s3://{bucket}/{key} — already exists")
        return False

    size_mb = local_path.stat().st_size / 1_000_000
    print(f"  [upload] {local_path.name} ({size_mb:.1f} MB) → s3://{bucket}/{key}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            s3_client.upload_file(
                str(local_path),
                bucket,
                key,
                ExtraArgs={
                    "ContentType": "application/octet-stream",
                    "ServerSideEncryption": "AES256",
                },
                Config=TRANSFER_CFG,
            )
            print(f"  [ok]     uploaded {size_mb:.1f} MB")
            return True
        except Exception as e:
            print(f"  [retry]  attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                wait = min(30, 5 * attempt)
                print(f"  [wait]   sleeping {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"  [FAIL]   giving up on {local_path.name} after {MAX_RETRIES} attempts")
                raise


# ── Manifest builder ──────────────────────────────────────────────────────────

def build_manifest(draft: dict) -> dict:
    """
    Build the final manifest.json that the SDK reads.
    Structure:
    {
      "version": 1,
      "updated_at": "2025-01-01T00:00:00Z",
      "models": {
        "whisper": { "file": "...", "sha256": "...", "size_mb": 38 },
        "indic_en": { ... },
        "indic_from_en": { ... },
      },
      "languages": {
        "hi": { "name": "Hindi", "tts": { "file": "...", "sha256": "..." } },
        ...
      }
    }

    SDK uses:
      manifest["models"]["whisper"]  → single model for all STT
      manifest["models"]["indic_en"] → single model for all Indic→EN MT
      manifest["languages"]["hi"]["tts"] → per-language TTS model (lazy load)
    """
    manifest = {
        "version":    MODEL_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "whisper":       draft.get("whisper", {}),
            "indic_en":      draft.get("indic_en", {}),
            "indic_from_en": draft.get("indic_from_en", {}),
        },
        "languages": {},
    }

    for lang_code, lang_cfg in LANGUAGES.items():
        lang_entry = {
            "name":         lang_cfg["name"],
            "whisper_lang": lang_cfg["whisper_lang"],
            "flores":       lang_cfg["flores"],
        }
        # attach TTS model if available
        tts = draft.get("tts", {}).get(lang_code)
        if tts:
            lang_entry["tts"] = tts
        manifest["languages"][lang_code] = lang_entry

    return manifest


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Step 4 — Upload to S3")
    print("=" * 60)

    dist = Path(DIST_DIR)
    draft_path = dist / "manifest_draft.json"

    if not draft_path.exists():
        print(f"[FATAL] {draft_path} not found — run step2_quantize.py first")
        sys.exit(1)

    draft = json.loads(draft_path.read_text())
    manifest = build_manifest(draft)

    # ── S3 client (generous timeouts for large uploads on slow links) ────────
    s3 = boto3.client("s3", config=BotoConfig(
        connect_timeout=30,
        read_timeout=300,
        retries={"max_attempts": 3, "mode": "adaptive"},
    ))

    # ── Step 1: upload model files (immutable) ────────────────────────────────
    print("\n── Uploading model files ─────────────────────────────────────")

    ort_files = list(dist.glob("*.ort")) + list(dist.glob("*.zip"))
    if not ort_files:
        print("[FATAL] No .ort files found in ./dist/ — run step2_quantize.py first")
        sys.exit(1)

    for f in sorted(ort_files):
        key = f"{S3_MODELS_PREFIX}{f.name}"
        upload_file_if_new(s3, f, S3_BUCKET, key)
        # also upload sha256 sidecar
        sha_file = f.with_suffix(f.suffix + ".sha256")
        if sha_file.exists():
            upload_file_if_new(s3, sha_file, S3_BUCKET, key + ".sha256")

    # ── Step 2: write + upload manifest LAST (atomic) ────────────────────────
    print("\n── Uploading manifest ────────────────────────────────────────")

    manifest_local = dist / "manifest.json"
    manifest_local.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  [write]  manifest.json written locally")

    s3.put_object(
        Bucket=CDN_BUCKET,
        Key=MANIFEST_KEY,
        Body=json.dumps(manifest, ensure_ascii=False),
        ContentType="application/json",
        CacheControl="max-age=3600, stale-while-revalidate=86400",
    )
    print(f"  [ok]     s3://{CDN_BUCKET}/{MANIFEST_KEY} updated")
    print(f"           version={MODEL_VERSION}  languages={list(manifest['languages'].keys())}")

    print("\n" + "=" * 60)
    print("  Upload complete.")
    print(f"  Manifest: https://{CDN_BUCKET}.s3.amazonaws.com/{MANIFEST_KEY}")
    print("  SDK will pick up new models on next init() — no APK change needed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
