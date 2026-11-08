#!/usr/bin/env bash
# reproduce every result in the paper from the raw Alibaba v2022 traces.
#
# usage: bash scripts/reproduce_paper.sh [--n-jobs N]
# needs data/data/CallGraph/CallGraph_{84,136,297}.csv and
# data/data/MSRTMCR/MCRRTUpdate_0.csv (run scripts/download_data.sh first).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1  # show progress immediately when logging to a file

echo "== real anomaly windows (Table real-world input)"
# the paper used the first 1M MSRTMCR rows to detect the anomaly windows
uv run python scripts/analyze_real_metrics.py --nrows 1000000

echo "== comparison, compute, convergence, stats, real"
uv run python scripts/evaluate_performance.py "$@"

echo "== ablation (Table ablation)"
uv run python scripts/run_ablation.py --dataset-index 84

echo "== done: results in results/"
