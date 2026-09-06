#!/usr/bin/env bash
# Anchor — the single command. First run creates .venv and installs dependencies;
# every run then starts the menu-bar dot. Extra arguments go to `python -m anchor`
# (e.g. ./run.sh doctor, ./run.sh headless).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

install_deps() {
  # Editable install of this project plus the test extra; falls back to the bare
  # dependency list from pyproject.toml if the editable build is not possible.
  # The fallback goes through a requirements file: word-splitting a quoted list
  # would hand pip literal quote characters, which it rejects.
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python -e ".[test]" && return 0
    deps_from_pyproject > .venv/requirements-fallback.txt
    uv pip install --python .venv/bin/python -r .venv/requirements-fallback.txt
  else
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install -e ".[test]" && return 0
    deps_from_pyproject > .venv/requirements-fallback.txt
    .venv/bin/python -m pip install -r .venv/requirements-fallback.txt
  fi
}

deps_from_pyproject() {
  # One requirement per line (pip -r format), so specifiers with spaces or markers survive.
  .venv/bin/python - <<'PY'
import tomllib
with open("pyproject.toml", "rb") as fh:
    data = tomllib.load(fh)
deps = list(data["project"].get("dependencies", []))
for extra in data["project"].get("optional-dependencies", {}).values():
    deps += extra
print("\n".join(deps))
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

# `doctor` and `version` must run without a key: the doctor is how a missing key gets diagnosed.
case "${1:-app}" in
  doctor|version) ;;
  *)
    if [ ! -f .env ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${ANCHOR_ENV:-}" ]; then
      echo "Create .env with OPENAI_API_KEY=... (setup.md #1)" >&2
      exit 1
    fi
    ;;
esac

exec .venv/bin/python -m anchor "$@"
