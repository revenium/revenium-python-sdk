#!/usr/bin/env bash
# load-demo.sh — load the agentic-outcomes examples into Revenium with one command.
#
#   REVENIUM_API_KEY=<your_key> ./load-demo.sh            # all three: sales + coding + support
#   REVENIUM_API_KEY=<your_key> ./load-demo.sh sales      # just one
#   REVENIUM_API_KEY=<your_key> ./load-demo.sh coding support
#   ./load-demo.sh list                                   # show available examples
#
# Required env:  REVENIUM_API_KEY
# Optional env:  REVENIUM_TEAM_ID   REVENIUM_API_BASE_URL   COUNT=<n>
set -uo pipefail
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- credentials: caller environment only. Your key is never printed. ---
if [ -z "${REVENIUM_API_KEY:-}" ]; then
  echo "FATAL: REVENIUM_API_KEY is not set. Export your Revenium API key and re-run:" >&2
  echo "       REVENIUM_API_KEY=<your_key> ./load-demo.sh" >&2
  exit 1
fi
export REVENIUM_API_KEY REVENIUM_METERING_API_KEY="$REVENIUM_API_KEY"
[ -n "${REVENIUM_TEAM_ID:-}" ] && export REVENIUM_TEAM_ID
[ -n "${REVENIUM_API_BASE_URL:-}" ] && export REVENIUM_API_BASE_URL

# --- venv: reuse existing, else build one alongside this script ---
VENV="$HERE/.venv"
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import revenium_middleware" 2>/dev/null; then
  PY="$VENV/bin/python"
else
  echo "Building venv at $VENV (one-time, ~1-2 min)..."
  python3 -m venv "$VENV" || { echo "FATAL: venv create failed" >&2; exit 1; }
  "$VENV/bin/pip" install -q revenium-python-sdk || { echo "FATAL: pip install failed" >&2; exit 1; }
  PY="$VENV/bin/python"
fi

cd "$HERE" || { echo "FATAL: cannot cd to $HERE" >&2; exit 1; }

# watchdog: metering lands before the SDK's at-exit hang on some platforms (notably macOS);
# wait for the run's done-sentinel, then reap the process so the script always returns.
run() {  # $1 label  $2 sentinel  rest: python script + args
  local label="$1" sentinel="$2"; shift 2
  local log; log="$(mktemp -t "revenium-load-$label.XXXXXX")"
  echo ">> $label"
  "$PY" "$@" >"$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || break
    grep -q "$sentinel" "$log" 2>/dev/null && { sleep 2; break; }
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  grep -E "Jobs:|Tickets:|CONVERTED:|Runtime:|Emitted|jobs \(SUCCESS" "$log" | sed 's/^/   /'
  rm -f "$log"
}

do_sales()   { run sales   "Runtime:" sales.py   --count "${COUNT:-18}"; }
do_coding()  { run coding  "Runtime:" coding.py  --count "${COUNT:-28}"; }
do_support() { run support "Runtime:" support.py --count "${COUNT:-28}"; }

examples=("$@"); [ ${#examples[@]} -eq 0 ] && examples=("all")
[ "${examples[0]}" = "list" ] && { echo "examples: sales · coding · support | all (default = all three)"; exit 0; }

echo "Loading [${examples[*]}] into Revenium ..."
for e in "${examples[@]}"; do
  case "$e" in
    all)     do_sales; do_coding; do_support ;;
    sales)   do_sales ;;
    coding)  do_coding ;;
    support) do_support ;;
    *) echo "  ?? unknown example '$e' — try: sales coding support all list" >&2 ;;
  esac
done
echo "DONE. Open your Revenium dashboard and refresh to see the metered jobs."
