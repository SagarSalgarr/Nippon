#!/usr/bin/env bash
# run_pipeline.sh — Run the full quantization pipeline end-to-end.
#
# Usage:
#   ./run_pipeline.sh                    # steps 1–3 only (local: download, quantize, eval)
#   ./run_pipeline.sh --skip-download    # quantize + eval (models already present)
#   ./run_pipeline.sh --upload           # also run step 4 — S3/CDN upload (needs AWS creds)
#   ./run_pipeline.sh --dry-run          # same as default (no upload); kept for scripts/README
#
# Step 4 (upload) is OFF by default so quantization + testing stay local-only.
#
# Requirements (from repo root: pip install -r pipeline/requirements.txt):
#   cd pipeline && pip install -r requirements.txt
#   pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
#   For --upload only: export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=...

set -euo pipefail

SKIP_DOWNLOAD=false
DO_UPLOAD=false

for arg in "$@"; do
  case $arg in
    --skip-download) SKIP_DOWNLOAD=true ;;
    --upload)        DO_UPLOAD=true ;;
    --dry-run)       ;; # no-op: upload already disabled unless --upload
  esac
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Indic AI SDK — Quantization Pipeline           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Download ──────────────────────────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = false ]; then
  echo "▶ Step 1 — Download models from HuggingFace"
  python step1_download.py
  echo ""
else
  echo "▶ Step 1 — Skipped (--skip-download)"
  echo ""
fi

# ── Step 2: Quantize ──────────────────────────────────────────────────────────
echo "▶ Step 2 — Export to ONNX + quantize INT8"
python step2_quantize.py
echo ""

# ── Step 3: Eval ─────────────────────────────────────────────────────────────
echo "▶ Step 3 — Evaluate all languages"
python step3_eval.py
echo ""

# ── Step 4: Upload to S3/CDN (opt-in; local dev skips this) ───────────────────
if [ "$DO_UPLOAD" = true ]; then
  echo "▶ Step 4 — Upload to S3"
  python step4_upload.py
  echo ""
else
  echo "▶ Step 4 — Skipped (local-only: use --upload when ready to publish to S3)"
  echo ""
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║   Pipeline complete.                             ║"
echo "╚══════════════════════════════════════════════════╝"
