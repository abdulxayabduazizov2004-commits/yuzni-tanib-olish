#!/bin/bash
# Ikki marta bosib ishga tushirish (macOS)
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
fi
[ -f models/person_reid_youtu_2021nov.onnx ] || .venv/bin/python download_models.py
exec .venv/bin/python server.py "$@"
