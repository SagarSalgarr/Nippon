"""
fix_pipeline.py — Run this once from your pipeline/ folder to patch both issues:

  1. mms_lang keys missing from LANGUAGES in config.py
  2. MMTTS_GENDER import error in step2_quantize.py

Run:
  cd pipeline
  python fix_pipeline.py
"""

import re

# ── Fix 1: config.py — add mms_lang to every language entry ──────────────────

MMS_LANG_MAP = {
    "hi": "hin",
    "mr": "mar",
    "ta": "tam",
    "te": "tel",
    "kn": "kan",
    "ml": "mal",
    "bn": "ben",
    "gu": "guj",
    "pa": "pan",
    "ur": "urd",
}

with open("config.py", "r") as f:
    cfg = f.read()

# Fix MMS_TTS_MODEL_PATTERN if still using old MMTTS pattern
if "MMTTS_MODEL_PATTERN" in cfg and "MMS_TTS_MODEL_PATTERN" not in cfg:
    cfg = re.sub(
        r'MMTTS_MODEL_PATTERN\s*=\s*"[^"]*"\s*.*\nMMTTS_GENDER\s*=\s*"[^"]*"\s*.*',
        'MMS_TTS_MODEL_PATTERN = "facebook/mms-tts-{mms_lang}"  # public, no token needed',
        cfg,
    )
    print("[config.py] fixed MMS_TTS_MODEL_PATTERN")
elif "MMS_TTS_MODEL_PATTERN" not in cfg:
    # Add it before quality gate section
    cfg = cfg.replace(
        "# ── Quality gate thresholds",
        'MMS_TTS_MODEL_PATTERN = "facebook/mms-tts-{mms_lang}"  # public, no token needed\n\n# ── Quality gate thresholds',
    )
    print("[config.py] added MMS_TTS_MODEL_PATTERN")

# Add mms_lang to each language entry if missing
for lang_code, mms_code in MMS_LANG_MAP.items():
    mms_line = f'"mms_lang":     "{mms_code}"'
    if mms_line not in cfg:
        # Insert after the "name" line for this language
        pattern = rf'("name":\s*"[^"]+",\n)(\s*\}},)'
        # More targeted: find the block for this lang_code and add mms_lang before closing brace
        # Find the language block and add mms_lang before its closing },
        block_pattern = rf'("{lang_code}":\s*\{{[^}}]+)"name":\s*"([^"]+)",\s*\n(\s*\}})'
        def add_mms(m):
            return m.group(0).replace(
                f'"name":',
                f'"mms_lang":     "{mms_code}",\n        "name":'
            )
        new_cfg = re.sub(block_pattern, add_mms, cfg, count=1, flags=re.DOTALL)
        if new_cfg != cfg:
            cfg = new_cfg
            print(f"[config.py] added mms_lang={mms_code} for {lang_code}")
        else:
            print(f"[config.py] {lang_code} already has mms_lang or pattern not found — check manually")

with open("config.py", "w") as f:
    f.write(cfg)

print("[config.py] done\n")

# ── Fix 2: step2_quantize.py — remove MMTTS_GENDER import, fix function names ─

with open("step2_quantize.py", "r") as f:
    s2 = f.read()

# Fix import — remove MMTTS_GENDER and MMTTS_MODEL_PATTERN, add MMS_TTS_MODEL_PATTERN
s2 = re.sub(r",?\s*MMTTS_GENDER", "", s2)
s2 = re.sub(r",?\s*MMTTS_MODEL_PATTERN", "", s2)

if "MMS_TTS_MODEL_PATTERN" not in s2:
    s2 = s2.replace(
        "MODEL_VERSION,",
        "MMS_TTS_MODEL_PATTERN, MODEL_VERSION,"
    )
    print("[step2_quantize.py] added MMS_TTS_MODEL_PATTERN to import")

# Fix quantize_mmtts → quantize_mms_tts
if "quantize_mmtts" in s2:
    s2 = s2.replace("def quantize_mmtts(", "def quantize_mms_tts(")
    s2 = s2.replace("result = quantize_mmtts(", "result = quantize_mms_tts(")
    print("[step2_quantize.py] renamed quantize_mmtts → quantize_mms_tts")

# Fix raw dir path mmtts- → mms-tts-
if 'f"{RAW_DIR}/mmtts-{lang_code}"' in s2:
    s2 = s2.replace(
        'f"{RAW_DIR}/mmtts-{lang_code}"',
        'f"{RAW_DIR}/mms-tts-{lang_code}"'
    )
    print("[step2_quantize.py] fixed raw dir path")

# Fix dist filename mmtts- → mms-tts-
if '"mmtts-{lang_code}-int8' in s2:
    s2 = s2.replace(
        'f"mmtts-{lang_code}-int8-v{MODEL_VERSION}.ort"',
        'f"mms-tts-{lang_code}-int8-v{MODEL_VERSION}.ort"'
    )
    print("[step2_quantize.py] fixed dist filename")

with open("step2_quantize.py", "w") as f:
    f.write(s2)

print("[step2_quantize.py] done\n")

# ── Fix 3: step1_download.py — fix the MMS loop if still using old pattern ────

with open("step1_download.py", "r") as f:
    s1 = f.read()

if "MMTTS_MODEL_PATTERN" in s1 or "MMTTS_GENDER" in s1:
    s1 = re.sub(r",?\s*MMTTS_MODEL_PATTERN", "", s1)
    s1 = re.sub(r",?\s*MMTTS_GENDER", "", s1)
    if "MMS_TTS_MODEL_PATTERN" not in s1:
        s1 = s1.replace(
            "RAW_DIR,",
            "MMS_TTS_MODEL_PATTERN, RAW_DIR,"
        )
    print("[step1_download.py] fixed MMTTS imports")

# Fix the download loop if still using old MMTTS pattern
if "MMTTS_MODEL_PATTERN.format(gender=MMTTS_GENDER" in s1:
    s1 = s1.replace(
        """    langs_with_tts = []
    for lang_code, lang_cfg in LANGUAGES.items():
        repo_id   = MMTTS_MODEL_PATTERN.format(gender=MMTTS_GENDER, lang=lang_code)
        local_dir = f\"{RAW_DIR}/mmtts-{lang_code}\"""",
        """    langs_with_tts = []
    for lang_code, lang_cfg in LANGUAGES.items():
        mms_lang = lang_cfg.get("mms_lang")
        if not mms_lang:
            print(f"\\n[skip]     MMS-TTS {lang_cfg['name']} — no mms_lang defined in config")
            continue
        repo_id   = MMS_TTS_MODEL_PATTERN.format(mms_lang=mms_lang)
        local_dir = f\"{RAW_DIR}/mms-tts-{lang_code}\""""
    )
    print("[step1_download.py] fixed MMS-TTS download loop")

with open("step1_download.py", "w") as f:
    f.write(s1)

print("[step1_download.py] done\n")
print("=" * 50)
print("All fixes applied. Now run:")
print("  bash run_pipeline.sh")
print("=" * 50)