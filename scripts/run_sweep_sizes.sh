#!/usr/bin/env bash
# Run the g(x) x y-bar Max-Cut sweep on SMALL / MEDIUM / LARGE instances.
#
# Designed for bhz (direct-connect GPU host, no SLURM): launch inside tmux so the
# run survives disconnects. Each size class writes its own CSV and log so they can
# be inspected / re-run independently.
#
#   tmux new -s sweep
#   cd /home/bhz/PDBO_LZJ/PDBO_test        # repo root (where main.py lives)
#   bash scripts/run_sweep_sizes.sh
#   # detach: Ctrl-b then d ;  reattach: tmux attach -t sweep
#
# Override any knob from the environment, e.g.:
#   SIZES="small large" YBAR="0 2 4 6 8" SEEDS="0 1" bash scripts/run_sweep_sizes.sh
set -euo pipefail

# ---- instance ids per size class (representative Gset graphs, paper Table 11) ----
G_SMALL="${G_SMALL:-1}"      # G1   : 800 nodes  / 19176 edges
G_MEDIUM="${G_MEDIUM:-22}"   # G22  : 2000 nodes / 19990 edges
G_LARGE="${G_LARGE:-67}"     # G67  : 10000 nodes / 20000 edges
# (G81 = 20000 nodes is the heaviest; add "xlarge" to SIZES and set G_XLARGE=81 to include)
G_XLARGE="${G_XLARGE:-81}"

SIZES="${SIZES:-small medium large}"   # which classes to run

# ---- swept variables (same across all sizes so the comparison is controlled) ----
YBAR="${YBAR:-0 1 2 4 6 8 10}"
SEEDS="${SEEDS:-0 1 2}"
# g list defaults to all 14 (11 convex + 2 partial + 1 linear); override with G_LIST="quad entropy ..."
G_LIST="${G_LIST:-}"

# ---- fixed algorithm hyper-parameters (do NOT vary within one experiment) ----
BATCH="${BATCH:-100}"
LR_X="${LR_X:-0.025}"
LR_Y="${LR_Y:-0.025}"
ITERS="${ITERS:-5000}"
NORMALIZE="${NORMALIZE:-1}"   # 1 => --g_normalize (value range [-1,0]); 0 => --no-g_normalize

RESULT_DIR="${RESULT_DIR:-results}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# Resolve repo root from this script's location, so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

norm_flag="--g_normalize"; [ "$NORMALIZE" = "0" ] && norm_flag="--no-g_normalize"
g_flag=""; [ -n "$G_LIST" ] && g_flag="--g $G_LIST"

run_one() {
    local size="$1" ids="$2"
    local out="$RESULT_DIR/sweep_${size}_${STAMP}.csv"
    local log="$RESULT_DIR/sweep_${size}_${STAMP}.log"
    echo "==================================================================="
    echo "[$(date +%H:%M:%S)] size=$size  ids=($ids)  -> $out"
    echo "  ybar=[$YBAR] seeds=[$SEEDS] batch=$BATCH iters=$ITERS normalize=$NORMALIZE"
    echo "==================================================================="
    # shellcheck disable=SC2086
    python scripts/sweep_g_y.py \
        --gset_ids $ids \
        $g_flag \
        --ybar $YBAR \
        --seeds $SEEDS \
        --batch "$BATCH" --lr_x "$LR_X" --lr_y "$LR_Y" --max_iters "$ITERS" \
        $norm_flag \
        --out "$out" 2>&1 | tee "$log"
    echo "[$(date +%H:%M:%S)] finished size=$size -> $out"
    echo
}

for size in $SIZES; do
    case "$size" in
        small)  run_one small  "$G_SMALL"  ;;
        medium) run_one medium "$G_MEDIUM" ;;
        large)  run_one large  "$G_LARGE"  ;;
        xlarge) run_one xlarge "$G_XLARGE" ;;
        *) echo "[warn] unknown size class: $size (skipped)" >&2 ;;
    esac
done

echo "All requested sweeps done. CSVs + logs under $RESULT_DIR/ (stamp $STAMP)."
