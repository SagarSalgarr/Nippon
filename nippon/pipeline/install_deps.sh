#!/usr/bin/env bash
# Install pipeline dependencies with long timeouts (PyTorch wheel is ~900MB+).
# Use this if: ReadTimeoutError / HTTPS read timed out from files.pythonhosted.org
set -euo pipefail
cd "$(dirname "$0")"

TIMEOUT="${PIP_DEFAULT_TIMEOUT:-1000}"
RETRIES="${PIP_RETRIES:-15}"

echo "pip install with --default-timeout=${TIMEOUT} --retries=${RETRIES}"
echo "(set PIP_DEFAULT_TIMEOUT / PIP_RETRIES to override)"
echo ""

pip install \
  --upgrade pip \
  --retries "${RETRIES}" \
  --default-timeout "${TIMEOUT}" \
  -r requirements.txt

echo ""
echo "Optional (IndicTrans eval): pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
