#!/usr/bin/env bash
set -euo pipefail

# --- 1) Crear y activar venv (Python 3.11 recomendado) ---
if [ ! -d ".venv" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv .venv
  else
    echo "Instalando Python 3.11 con Homebrew..."
    brew install python@3.11
    python3.11 -m venv .venv
  fi
fi

# Activar venv
# shellcheck disable=SC1091
source .venv/bin/activate

# --- 2) Actualizar instaladores ---
python -m pip install -U pip setuptools wheel

# --- 3) Instalar dependencias si faltan (versiones estables) ---
REQUIRED=("streamlit<2" "pandas==2.2.3" "numpy==2.1.3" "networkx==3.3" "geopy==2.4.1" "pyarrow==21.0.0" "pydeck==0.9.1")
pip install "${REQUIRED[@]}"

# --- 4) Exportar PYTHONPATH para que 'core/' se resuelva siempre ---
export PYTHONPATH=".:$PYTHONPATH"

# --- 5) Lanzar Streamlit ---
python -m streamlit run app/main_streamlit.py
