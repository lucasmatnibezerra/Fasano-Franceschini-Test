#!/usr/bin/env bash
# Run the whole synthetic programme unattended on the lab server.
#
#   1. frozen-configuration campaign  (4 scenarios x 10 seeds x 8 detectors)
#   2. FFWIN window sweep             (W = 500 and 1000, 3 seeds)
#   3. oracle-aligned power curve     (runs alongside, single core)
#   4. Table V rendered from (1)
#
# Every stage is resumable, so re-running this script after an interruption
# picks up where it stopped. Writes logs/ and finishes by creating DONE.
set -u

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE" || exit 1
mkdir -p logs
rm -f DONE FAILED

CORES="$(nproc 2>/dev/null || echo 8)"
MAIN_WORKERS=$(( CORES > 40 ? 40 : CORES ))
SWEEP_WORKERS=$(( CORES > 12 ? 12 : CORES ))

log() { echo "[$(date '+%F %T')] $*"; }

# An interpreter that already carries the stack, preferring the environment the
# September campaign ran in. Falls back to a --user install.
pick_python() {
  local cand
  for cand in "$HOME/codes/ffwin_revision/venv/bin/python" \
              "$HOME/codes/ffwin_revision/.venv/bin/python" \
              "$HOME/venv/bin/python" \
              python3.12 python3; do
    if [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c 'import numpy,pandas,scipy,sklearn,joblib,numba,rtree' >/dev/null 2>&1; then
        echo "$cand"
        return 0
      fi
    fi
  done
  return 1
}

PY="$(pick_python)" || {
  log "no interpreter carries the full stack; attempting a --user install"
  for cand in python3.12 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      "$cand" -m pip install --user numpy pandas scipy scikit-learn joblib numba rtree \
        >> logs/install.log 2>&1
    fi
  done
  PY="$(pick_python)" || {
    log "FAILED: still no usable interpreter, see logs/install.log"
    touch FAILED
    exit 1
  }
}

log "interpreter: $PY"
"$PY" -V 2>&1 | sed 's/^/  /'
log "cores=${CORES} main_workers=${MAIN_WORKERS} sweep_workers=${SWEEP_WORKERS}"

# (3) alongside everything else: one core, and nothing depends on it.
log "starting power curve in the background"
"$PY" scripts/power_curve.py > logs/power_curve.log 2>&1 &
POWER_PID=$!

log "stage 1: frozen-configuration campaign"
"$PY" scripts/run_synthetic.py --workers "${MAIN_WORKERS}" > logs/main.log 2>&1
log "stage 1 done: $(ls synthetic/results/_raw 2>/dev/null | wc -l)/40 realisations"

for W in 500 1000; do
  log "stage 2: FFWIN sweep W=${W}"
  "$PY" scripts/run_synthetic.py \
    --detectors FasanoFranceschiniWindow \
    --seeds 11 22 33 \
    --window "${W}" \
    --out "synthetic/results_w${W}" \
    --workers "${SWEEP_WORKERS}" > "logs/sweep_w${W}.log" 2>&1
  log "stage 2 W=${W} done: $(ls "synthetic/results_w${W}/_raw" 2>/dev/null | wc -l)/12 realisations"
done

log "stage 4: rendering Table V"
"$PY" scripts/make_synth_table.py > logs/make_synth_table.log 2>&1

log "waiting for the power curve"
wait "$POWER_PID" 2>/dev/null

{
  echo "=== realisations ==="
  echo "frozen configuration: $(ls synthetic/results/_raw 2>/dev/null | wc -l)/40"
  for W in 500 1000; do
    echo "sweep W=${W}: $(ls "synthetic/results_w${W}/_raw" 2>/dev/null | wc -l)/12"
  done
  echo
  echo "=== FFWIN, one line per realisation ==="
  grep -h "FasanoFranceschini" logs/main.log logs/sweep_w*.log 2>/dev/null | sed 's/  */ /g'
  echo
  echo "=== power curve ==="
  grep -E "===|W=|effect size" logs/power_curve.log 2>/dev/null
} > SUMMARY.txt 2>&1

touch DONE
log "all stages finished; see SUMMARY.txt, tables/table_synth.tex, synthetic/results*/"
