"""
step1_download.py — Download all models from HuggingFace.

Run:  python step1_download.py

Per-model skip logic (no re-download if weights already exist):
  For each ./models/raw/{model}/, scan the tree for any file ending in
  .bin, .safetensors, or .pt
  → If found: print [skip] and continue
  → If missing: snapshot_download from HuggingFace

Re-runs are safe: Whisper / IndicTrans2 / MMTTS are each decided independently.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login
from config import (
    LANGUAGES, WHISPER_MODEL_ID, INDIC_EN_MODEL_ID,
    INDIC_FROM_MODEL_ID, MMS_TTS_MODEL_PATTERN, RAW_DIR,
    HF_TOKEN,
)



def hf_login() -> None:
    """Authenticate with HuggingFace. Required for gated models (IndicTrans2, MMTTS)."""
    if not HF_TOKEN or HF_TOKEN.startswith("hf_xxx"):
        print("[warn] HF_TOKEN not set — only public models will download.")
        print("       Set it in config.py or: export HF_TOKEN=hf_your_token")
        print("       Get yours at: https://huggingface.co/settings/tokens")
        return
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("[auth] HuggingFace login OK")


# Weight file extensions that indicate a real model snapshot (recursive check).
_WEIGHT_SUFFIXES = frozenset({".bin", ".safetensors", ".pt"})


def is_downloaded(local_dir: str) -> bool:
    """
    True if local_dir exists and contains at least one weight file anywhere
    under it (.bin, .safetensors, or .pt). HuggingFace often nests files in
    subfolders; we use rglob so partial top-level-only layouts still count.
    """
    root = Path(local_dir)
    if not root.is_dir():
        return False
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES:
            return True
    return False


def download_model(repo_id: str, local_dir: str, description: str, required: bool = True) -> bool:
    """
    Download a model only if not already present.
    Returns True if downloaded or already present, False if skipped due to error.
    """
    if is_downloaded(local_dir):
        print(f"\n[skip] {description}")
        print(f"       weights already present under {local_dir}")
        return True

    print(f"\n[download] {description}")
    print(f"           repo  : {repo_id}")
    print(f"           target: {local_dir}")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=HF_TOKEN or None,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"           [OK]")
        return True
    except Exception as e:
        print(f"           [FAIL] {e}")
        if required:
            sys.exit(1)
        return False


def main():
    print("=" * 60)
    print("  Step 1 — Model Download  (skips already-downloaded models)")
    print("=" * 60)

    hf_login()

    # 1. Whisper Small — single multilingual model, covers all 10 languages
    download_model(
        repo_id=WHISPER_MODEL_ID,
        local_dir=f"{RAW_DIR}/whisper-small",
        description="Whisper Small (multilingual STT)",
        required=True,
    )

    # 2. IndicTrans2 Indic→EN
    download_model(
        repo_id=INDIC_EN_MODEL_ID,
        local_dir=f"{RAW_DIR}/indic-trans2-indic-en",
        description="IndicTrans2 Indic→EN translation",
        required=True,
    )

    # 3. IndicTrans2 EN→Indic
    download_model(
        repo_id=INDIC_FROM_MODEL_ID,
        local_dir=f"{RAW_DIR}/indic-trans2-en-indic",
        description="IndicTrans2 EN→Indic translation",
        required=True,
    )

    # 4. Meta MMS-TTS — one small model per language (~150-300MB each)
    #    Uses ISO 639-3 lang codes (mms_lang field in LANGUAGES config)
    #    All repos are public — no HF token needed for these
    langs_with_tts = []
    for lang_code, lang_cfg in LANGUAGES.items():
        mms_lang = lang_cfg.get("mms_lang")
        if not mms_lang:
            print(f"\n[skip]     MMS-TTS {lang_cfg['name']} — no mms_lang defined in config")
            continue
        repo_id   = MMS_TTS_MODEL_PATTERN.format(mms_lang=mms_lang)
        local_dir = f"{RAW_DIR}/mms-tts-{lang_code}"
        ok = download_model(
            repo_id=repo_id,
            local_dir=local_dir,
            description=f"MMS-TTS {lang_cfg['name']} ({mms_lang})",
            required=False,
        )
        if ok:
            langs_with_tts.append(lang_code)


    print("\n" + "=" * 60)
    print("  Download complete.")
    print(f"  Whisper : 1 model (all languages)")
    print(f"  MT      : 2 models (Indic→EN, EN→Indic)")
    print(f"  TTS     : {len(langs_with_tts)} languages ({', '.join(langs_with_tts)})")
    print("=" * 60)


if __name__ == "__main__":
    main()