#!/usr/bin/env bash
# Anchor — the single command. First run creates .venv and installs dependencies;
# every run then starts the menu-bar dot. Extra arguments go to `python -m anchor`
# (e.g. ./run.sh doctor, ./run.sh headless).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

install_deps() {
  # Editable install of this project plus the test extra; falls back to the bare
  # dependency list from pyproject.toml if the editable build is not possible.
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python -e ".[test]" && return 0
    uv pip install --python .venv/bin/python $(deps_from_pyproject)
  else
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install -e ".[test]" && return 0
    .venv/bin/python -m pip install $(deps_from_pyproject)
  fi
}

deps_from_pyproject() {
  .venv/bin/python - <<'PY'
import tomllib
with open("pyproject.toml", "rb") as fh:
    data = tomllib.load(fh)
deps = list(data["project"].get("dependencies", []))
for extra in data["project"].get("optional-dependencies", {}).values():
    deps += extra
print(" ".join(f'"{d}"' for d in deps))
PY
}

if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv ..."
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python 3.12
  else
    python3 -m venv .venv
  fi
  install_deps
fi

if [ ! -f .env ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${ANCHOR_ENV:-}" ]; then
  echo "Create .env with OPENAI_API_KEY=... (setup.md #1)" >&2
  exit 1
fi

exec .venv/bin/python -m anchor "$@"
