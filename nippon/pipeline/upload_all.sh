#!/bin/bash
# Usage:
#   export AWS_ACCESS_KEY_ID=your_key
#   export AWS_SECRET_ACCESS_KEY=your_secret
#   export AWS_DEFAULT_REGION=ap-south-1   # optional, defaults below
#   bash upload_all.sh
#
# Or set credentials via ~/.aws/credentials / AWS SSO / IAM role.
# Never hardcode credentials in this file.

: "${AWS_DEFAULT_REGION:=ap-south-1}"

# Resolve paths relative to this script's location (works from any working dir)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST="${SCRIPT_DIR}/dist"
BUCKET="indicai-models"
PREFIX="models"
CDN_BUCKET="indicai-cdn"
MAX_RETRIES=5
LOG_FILE="${SCRIPT_DIR}/upload_log.txt"

# Validate AWS credentials are set
if [ -z "${AWS_ACCESS_KEY_ID}" ] && ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "ERROR: AWS credentials not configured."
    echo "Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or configure ~/.aws/credentials"
    exit 1
fi

# Count total files
TOTAL_ZIP=$(ls "$DIST"/*.zip 2>/dev/null | wc -l)
TOTAL_SHA=$(ls "$DIST"/*.sha256 2>/dev/null | wc -l)
TOTAL_FILES=$(( TOTAL_ZIP + TOTAL_SHA ))

UPLOADED=0
SKIPPED=0
FAILED=0
START_TIME=$(date +%s)

log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

print_progress() {
    local done=$(( UPLOADED + SKIPPED ))
    local elapsed=$(( $(date +%s) - START_TIME ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PROGRESS: ${done}/${TOTAL_FILES} done | ✓ ${UPLOADED} uploaded | ⏭ ${SKIPPED} skipped | ✗ ${FAILED} failed"
    echo "  ELAPSED:  ${mins}m ${secs}s"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

upload_file() {
    local file="$1"
    local fname=$(basename "$file")
    local key="${PREFIX}/${fname}"
    local size_bytes=$(stat -c%s "$file")
    local size_mb=$(( size_bytes / 1000000 ))

    # Check if already exists on S3
    log "Checking s3://${BUCKET}/${key} ..."
    if aws s3api head-object --bucket "$BUCKET" --key "$key" --cli-read-timeout 30 --cli-connect-timeout 15 >/dev/null 2>&1; then
        log "[SKIP]   ${fname} (${size_mb} MB) — already on S3 ✓"
        SKIPPED=$(( SKIPPED + 1 ))
        return 0
    fi

    log "[START]  ${fname} (${size_mb} MB) → s3://${BUCKET}/${key}"

    for attempt in $(seq 1 $MAX_RETRIES); do
        local attempt_start=$(date +%s)
        log "         attempt ${attempt}/${MAX_RETRIES} starting..."

        if aws s3 cp "$file" "s3://${BUCKET}/${key}" \
            --cli-read-timeout 600 \
            --cli-connect-timeout 60 2>&1; then

            local attempt_elapsed=$(( $(date +%s) - attempt_start ))
            local speed_mbps=0
            if [ $attempt_elapsed -gt 0 ]; then
                speed_mbps=$(( size_mb / attempt_elapsed ))
            fi
            log "[OK]     ${fname} uploaded in ${attempt_elapsed}s (~${speed_mbps} MB/s) ✓"
            UPLOADED=$(( UPLOADED + 1 ))
            return 0
        fi

        local wait_secs=$(( attempt * 15 ))
        log "[FAIL]   attempt ${attempt}/${MAX_RETRIES} failed for ${fname}"

        if [ $attempt -lt $MAX_RETRIES ]; then
            log "         waiting ${wait_secs}s before retry..."
            sleep $wait_secs
        fi
    done

    log "[ERROR]  ${fname} — GAVE UP after ${MAX_RETRIES} attempts ✗"
    FAILED=$(( FAILED + 1 ))
    return 1
}

# ── Start ────────────────────────────────────────────────────
echo "" > "$LOG_FILE"

echo "╔══════════════════════════════════════════════╗"
echo "║     Indic AI — S3 Model Upload               ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Bucket:  ${BUCKET}                          "
echo "║  CDN:     ${CDN_BUCKET}                      "
echo "║  Files:   ${TOTAL_FILES} (${TOTAL_ZIP} zips + ${TOTAL_SHA} checksums)"
echo "║  Region:  ${AWS_DEFAULT_REGION}              "
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Upload zip files first (large ones) ──────────────────────
log "── Phase 1: Uploading model .zip files ──────────"
for f in "$DIST"/*.zip; do
    [ -f "$f" ] || continue
    upload_file "$f"
    print_progress
done

# ── Then sha256 files (tiny) ─────────────────────────────────
log "── Phase 2: Uploading .sha256 checksums ─────────"
for f in "$DIST"/*.sha256; do
    [ -f "$f" ] || continue
    upload_file "$f"
done
print_progress

# ── Upload manifest to CDN bucket ────────────────────────────
log "── Phase 3: Building & uploading manifest.json ──"

cd "${SCRIPT_DIR}"
source .venv/bin/activate
python -c "
import json, sys
from pathlib import Path
from datetime import datetime, timezone
from config import LANGUAGES, MODEL_VERSION

draft = json.loads(Path('dist/manifest_draft.json').read_text())
manifest = {
    'version': MODEL_VERSION,
    'updated_at': datetime.now(timezone.utc).isoformat(),
    'models': {
        'whisper': draft.get('whisper', {}),
        'indic_en': draft.get('indic_en', {}),
        'indic_from_en': draft.get('indic_from_en', {}),
    },
    'languages': {},
}
for lc, cfg in LANGUAGES.items():
    entry = {'name': cfg['name'], 'whisper_lang': cfg['whisper_lang'], 'flores': cfg['flores']}
    tts = draft.get('tts', {}).get(lc)
    if tts:
        entry['tts'] = tts
    manifest['languages'][lc] = entry

out = Path('dist/manifest.json')
out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f'manifest.json written ({out.stat().st_size} bytes)')
print(f'  version={MODEL_VERSION}  languages={list(manifest[\"languages\"].keys())}')
"

log "Uploading manifest.json to CDN bucket..."
if aws s3 cp "$DIST/manifest.json" "s3://${CDN_BUCKET}/manifest.json" \
    --content-type "application/json" \
    --cache-control "max-age=3600, stale-while-revalidate=86400" \
    --cli-read-timeout 60 \
    --cli-connect-timeout 30 2>&1; then
    log "[OK]     manifest.json → s3://${CDN_BUCKET}/manifest.json ✓"
    UPLOADED=$(( UPLOADED + 1 ))
else
    log "[ERROR]  manifest.json upload failed ✗"
    FAILED=$(( FAILED + 1 ))
fi

# ── Final summary ────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║             UPLOAD COMPLETE                   ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  ✓ Uploaded:  ${UPLOADED} files              "
echo "║  ⏭ Skipped:   ${SKIPPED} files (already on S3)"
echo "║  ✗ Failed:    ${FAILED} files                "
echo "║  ⏱ Time:      ${TOTAL_MINS}m ${TOTAL_SECS}s "
echo "╠══════════════════════════════════════════════╣"
echo "║  Model URL:    https://${BUCKET}.s3.${AWS_DEFAULT_REGION}.amazonaws.com/models/"
echo "║  Manifest URL: https://${CDN_BUCKET}.s3.${AWS_DEFAULT_REGION}.amazonaws.com/manifest.json"
echo "║  CDN Website:  http://${CDN_BUCKET}.s3-website.${AWS_DEFAULT_REGION}.amazonaws.com"
echo "╚══════════════════════════════════════════════╝"

if [ $FAILED -gt 0 ]; then
    echo ""
    log "WARNING: ${FAILED} file(s) failed. Re-run this script to retry (already-uploaded files will be skipped)."
    exit 1
fi

log "Full log saved to: ${LOG_FILE}"
