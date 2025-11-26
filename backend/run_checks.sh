#!/usr/bin/env sh
set -e

echo "Running regression test..."
python backend/regression_test.py

echo "Byte-compiling backend modules..."
python -m compileall backend/generate_news_scores.py backend/generate_pick.py

echo "Done."
