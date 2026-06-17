#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/root/autodl-tmp/datasets/calvin/lerobot/task_ABC_D}"
MANIFEST="${MANIFEST:-/root/autodl-tmp/calvin_abc_d_actual_parquets.txt}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/miniforge3/envs/Evo1/bin/python}"
PARALLEL="${PARALLEL:-128}"
PASSES="${PASSES:-5}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-8}"
MAX_TIME="${MAX_TIME:-120}"
RETRY="${RETRY:-3}"

BASES=(
  "https://hf-mirror.com/datasets/CollisionCode/calvin_abc_d_lerobot_v2.1/resolve/main"
  "https://huggingface.co/datasets/CollisionCode/calvin_abc_d_lerobot_v2.1/resolve/main"
)

source /etc/network_turbo >/dev/null 2>&1 || true

echo "START date=$(date -Is) root=${ROOT} manifest=${MANIFEST} parallel=${PARALLEL} passes=${PASSES}"
find "${ROOT}/data" -name '*.tmp.*' -type f -delete 2>/dev/null || true

for pass in $(seq 1 "${PASSES}"); do
  REM="/root/autodl-tmp/calvin_abc_d_remaining_pass_${pass}.txt"
  FAIL="/root/autodl-tmp/calvin_abc_d_fail_pass_${pass}.txt"
  export ROOT MANIFEST REM

  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
manifest = Path(os.environ["MANIFEST"])
remaining = Path(os.environ["REM"])

missing = []
for line in manifest.read_text().splitlines():
    rel = line.strip()
    path = root / rel
    if rel and not (path.is_file() and path.stat().st_size > 0):
        missing.append(rel)

remaining.write_text("\n".join(missing) + ("\n" if missing else ""))
print(len(missing))
PY

  total="$(wc -l < "${MANIFEST}")"
  remaining="$(wc -l < "${REM}" 2>/dev/null || echo 0)"
  done_count="$(( total - remaining ))"
  echo "PASS ${pass} begin date=$(date -Is) done=${done_count}/${total} remaining=${remaining}"
  [ "${remaining}" -eq 0 ] && break

  : > "${FAIL}"
  export ROOT FAIL CONNECT_TIMEOUT MAX_TIME RETRY
  export BASE_0="${BASES[0]}"
  export BASE_1="${BASES[1]}"

  xargs -r -n 1 -P "${PARALLEL}" bash -c '
    rel="$1"
    file="${ROOT}/${rel}"
    if [ -s "${file}" ]; then
      exit 0
    fi

    mkdir -p "$(dirname "${file}")"
    tmp="${file}.tmp.$$"
    for base in "${BASE_0}" "${BASE_1}"; do
      if curl -L --fail --retry "${RETRY}" --retry-all-errors --retry-delay 0 \
          --connect-timeout "${CONNECT_TIMEOUT}" --max-time "${MAX_TIME}" \
          -sS -o "${tmp}" "${base}/${rel}"; then
        mv "${tmp}" "${file}"
        exit 0
      fi
    done

    rm -f "${tmp}"
    echo "${rel}" >> "${FAIL}"
    exit 1
  ' _ < "${REM}"

  code="$?"
  fail="$(wc -l < "${FAIL}" 2>/dev/null || echo 0)"
  current="$(while read -r rel; do [ -s "${ROOT}/${rel}" ] && echo 1; done < "${MANIFEST}" | wc -l)"
  echo "PASS ${pass} end date=$(date -Is) code=${code} done=${current}/${total} fail=${fail}"
done

find "${ROOT}/data" -name '*.tmp.*' -type f -delete 2>/dev/null || true
final="$(while read -r rel; do [ -s "${ROOT}/${rel}" ] && echo 1; done < "${MANIFEST}" | wc -l)"
echo "DONE date=$(date -Is) done=${final}/$(wc -l < "${MANIFEST}")"
