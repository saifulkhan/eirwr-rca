#!/usr/bin/env bash
# download and extract the Alibaba v2022 files used in the paper into
# data/data/ (about 0.9 GB download, 13 GB extracted). safe to re-run:
# finished downloads are skipped and partial ones resume.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces
FILES=(
  "CallGraph/CallGraph_84.tar.gz CallGraph"
  "CallGraph/CallGraph_136.tar.gz CallGraph"
  "CallGraph/CallGraph_297.tar.gz CallGraph"
  "MCRRTUpdate/MCRRTUpdate_0.tar.gz MSRTMCR"
  "MSMetricsUpdate/MSMetricsUpdate_0.tar.gz MSMetrics"
)

for entry in "${FILES[@]}"; do
  read -r remote dir <<< "$entry"
  dest="data/data/$dir"
  archive="$dest/$(basename "$remote")"
  csv="${archive%.tar.gz}.csv"
  mkdir -p "$dest"
  if [[ -f "$csv" ]]; then
    echo "ok       $csv"
    continue
  fi
  echo "download $remote"
  wget -q --show-progress -c --tries=20 --timeout=50 -O "$archive" "$BASE/$remote"
  echo "extract  $archive"
  tar -xzf "$archive" -C "$dest"
done
echo "data ready in data/data/"
