#!/bin/bash
# =============================================================================
# CT052-3-M-NLP  —  Convenience launcher
# Activates the nlp conda environment, starts the Spelling Correction GUI
# in the background, waits briefly, then starts the Sentiment Web App in
# the foreground.
#
# Usage:
#     bash Code/run_all.sh
# =============================================================================

# Activate conda (works even if conda is not on PATH yet)
eval "$(conda shell.bash hook)"
conda activate nlp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  CT052-3-M-NLP — Starting both systems"
echo "============================================================"

# --- Spelling Correction GUI (background) ---
echo ""
echo "=== Starting Spelling Correction GUI ==="
python "${SCRIPT_DIR}/spelling_correction/gui.py" &
GUI_PID=$!
sleep 3

# --- Sentiment Web App (foreground) ---
echo ""
echo "=== Starting Sentiment Web App ==="
echo "    Open http://localhost:8000 in your browser"
python "${SCRIPT_DIR}/text_classification/deploy.py"

# If deploy.py exits, also kill the GUI
kill $GUI_PID 2>/dev/null
