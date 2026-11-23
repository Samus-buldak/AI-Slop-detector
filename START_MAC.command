#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1
trap 'result=$?; if [ "$result" -ne 0 ]; then echo "Ошибка запуска. См. START_HERE_RU.txt"; fi; read -r -p "Нажми Enter, чтобы закрыть окно..."' EXIT
if [ ! -x .venv-local/bin/python ]; then
    python3 -m venv .venv-local
fi
.venv-local/bin/python -m pip install -r requirements-local-lock.txt
.venv-local/bin/python quick_start.py
