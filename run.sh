#!/bin/bash
# ─────────────────────────────────────────
#  GPS Mapper — quick start (macOS / Linux)
# ─────────────────────────────────────────

echo ""
echo "  GPS Mapper – Setup"
echo "  ────────────────────"

# 1. Tesseract
if ! command -v tesseract &> /dev/null; then
  echo ""
  echo "  [!] Tesseract not found."
  echo "      Install it first:"
  echo ""
  echo "  macOS:   brew install tesseract"
  echo "  Ubuntu:  sudo apt install tesseract-ocr"
  echo "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
  echo ""
  exit 1
fi

echo "  [✓] Tesseract: $(tesseract --version 2>&1 | head -1)"

# 2. Python deps
echo "  [→] Installing Python packages…"
pip install -r requirements.txt -q

echo "  [✓] All packages installed"
echo ""
echo "  [→] Starting server at http://localhost:5000"
echo ""

python app.py
