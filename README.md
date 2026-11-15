# eIRWR: Enhanced Iterative Random Walk with Restart

Root cause analysis (RCA) for microservice architectures. eIRWR ranks
services by how likely they are to be the root cause of an incident,
using only the service call graph and aggregate anomaly scores.

This repository contains the implementation, the baselines, and the
scripts that produce every table and figure in the paper.

## Quick start: reproduce the paper

```sh
git clone https://github.com/saifulkhan/eirwr-rca.git && cd eirwr-rca
uv sync                              # 1. install dependencies
bash scripts/download_data.sh        # 2. download the dataset
bash scripts/reproduce_paper.sh      # 3. run all experiments
```

Step 3 overwrites the CSV files in `results/`, which are committed to
the repository. To see whether your run matches the paper, compare
against the committed versions:

```sh
git diff --stat results/
```

An empty diff for a file means an exact reproduction.
`computational_metrics.csv` (runtime and memory) is expected to differ,
because timings depend on the machine. On other platforms, floating
point values may also differ in the last digits.

To regenerate the figures in `results/figures/` from the CSV files:

```sh
uv run python scripts/make_paper_figures.py
```

The figure PDFs are written without a creation date, so the same CSV
files always give byte-identical figures.

## Requirements

- Linux or macOS with `bash`, `wget` and `tar`.
- Python >= 3.13 and [uv](https://docs.astral.sh/uv/). `uv sync`
  installs the exact package versions in `uv.lock`.
- About 8 GB of free disk space for the dataset.
- Several CPU cores. The main comparison runs in parallel. The full
  run took about 75 minutes on a 28-core machine with 196 GB of RAM.

## Step details

### 1. Install

```sh
uv sync
```

### 2. Data

The experiments use the
[Alibaba Microservice Trace Dataset v2022](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022).
The dataset is not part of this repository. The `data/` folder is
git-ignored, so the downloaded files stay on your machine.

#### Get the data

The download script fetches four archives into `data/data/` and
extracts them (about 0.8 GB download, 7 GB extracted):

```sh
bash scripts/download_data.sh
```

The script can be re-run safely: it skips finished files and resumes
partial downloads.

| File | Used for |
| --- | --- |
| `CallGraph/CallGraph_{84,136,297}.csv` | the three topologies CG84, CG136, CG297 |
| `MSRTMCR/MCRRTUpdate_297.csv` | real anomaly windows on CG297 |

Without `wget`, the same files can be downloaded with `curl`:

```sh
BASE=https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces
mkdir -p data/data/CallGraph data/data/MSRTMCR
for i in 84 136 297; do
  curl -fL -o data/data/CallGraph/CallGraph_$i.tar.gz \
    "$BASE/CallGraph/CallGraph_$i.tar.gz"
  tar -xzf data/data/CallGraph/CallGraph_$i.tar.gz -C data/data/CallGraph
done
curl -fL -o data/data/MSRTMCR/MCRRTUpdate_297.tar.gz \
  "$BASE/MCRRTUpdate/MCRRTUpdate_297.tar.gz"
tar -xzf data/data/MSRTMCR/MCRRTUpdate_297.tar.gz -C data/data/MSRTMCR
```

After extraction, the folder should look like this:

```text
data/data/
├── CallGraph/
│   ├── CallGraph_84.csv
│   ├── CallGraph_136.csv
│   └── CallGraph_297.csv
└── MSRTMCR/
    └── MCRRTUpdate_297.csv
```

The `.tar.gz` archives can be deleted after extraction. Only the
optional topology selection (below) reads them.

#### How the data are processed

There is no separate preprocessing step and no intermediate files.
The scripts read the raw CSV files and process them in memory.

1. **Dependency graph** (all experiments). `src/loader.py` reads a
   CallGraph CSV and skips malformed lines.
   `build_dependency_graph` drops rows with a missing caller (`um`),
   callee (`dm`) or response time (`rt`), and rows where either
   service is `UNKNOWN`. It then counts the calls of each
   (caller, callee) pair. `compute_weight_matrix` divides each row of
   the count matrix by its sum, which gives the weight matrix W.
2. **Simulated incidents.** The anomaly vectors are generated on W, so
   they need no further data (see [Evaluation protocol](#evaluation-protocol)).
3. **Real anomaly windows.** Each CallGraph and MSRTMCR file covers
   three minutes of the trace, and files with the same index cover the
   same three minutes. CG297 therefore uses `MCRRTUpdate_297.csv`
   (one-minute timestamps 53,460,000 to 53,580,000 ms).
   `scripts/analyze_real_metrics.py` computes, for each service, the
   mean and standard deviation of the RPC response time
   (`providerrpc_rt`) over the whole file. A reading is anomalous when
   its z-score exceeds 3. A timestamp is kept when at least three
   services are anomalous at the same time. The result is written to
   `results/real_anomaly_windows.csv`. The `real` experiment then
   builds one anomaly vector per window from the positive response
   time z-scores, with the same baselines (`src/rca/real_eval.py`).

### 3. Experiments

`scripts/reproduce_paper.sh` runs the steps below in order. Extra
arguments are passed to `evaluate_performance.py`, for example
`--n-jobs 8` to limit the number of workers.

To keep the run going after you log out or close the editor:

```sh
nohup bash scripts/reproduce_paper.sh > reproduce.log 2>&1 &
tail -f reproduce.log
```

Each step can also be run on its own (prefix with `uv run python`):

| Command | Output in `results/` | Paper |
| --- | --- | --- |
| `scripts/analyze_real_metrics.py --dataset-index 297` | `real_anomaly_windows.csv` | input to the real-window case study |
| `scripts/evaluate_performance.py --experiments comparison` | `full_comparison_metrics_CG*.csv`, `per_incident_ranks_CG*.csv`, `multi_topology_metrics.csv` | Table main-results, MRR figures |
| `scripts/evaluate_performance.py --experiments compute` | `computational_metrics.csv` | Table computational, scalability figure |
| `scripts/evaluate_performance.py --experiments convergence` | `convergence_metrics.csv` | convergence figure |
| `scripts/evaluate_performance.py --experiments stats` | `statistical_*.csv` | Table Wilcoxon, Friedman test |
| `scripts/evaluate_performance.py --experiments real` | `real_world_evaluation.csv` | real-window case study (cascade coverage) |
| `scripts/real_window_diagnostics.py --dataset-index 297` | `real_window_diagnostics.csv` | real-window case study (positive services, callers, Spearman) |
| `scripts/run_ablation.py --dataset-index 84` | `ablation_metrics.csv` | Table ablation |

The `stats` and `real` experiments read the output of `comparison` and
`analyze_real_metrics.py`, so run those first. `real_window_diagnostics.py` reads the output of `real`.

Useful options of `evaluate_performance.py` (see `--help`):

- `--quick`: a smaller configuration (4 visibility levels, 50
  incidents) to check that everything runs.
- `--resume`: skip topologies whose comparison CSV already exists.
- `--seed`: random seed (default 42). Results depend only on the seed,
  not on the number of workers.

### Topology selection (optional)

`scripts/select_topologies.py` reproduces how CG84, CG136 and CG297
were chosen and writes `results/selected_topologies.csv` (Table
dataset-stats). It picks the three most dissimilar topologies by
size and density, always including CG297. By default, it reads the
graph statistics of all 500 topologies from the committed
`results/topology_stats.csv`. To recompute that file, run it with
`--rescan`. This needs all 500 CallGraph archives in
`data/data/CallGraph/` (downloaded with `fetchData.sh` from the
Alibaba repository). The scan extracts every archive, which takes
about 800 GB of disk space for the CSV files.

## Evaluation protocol

1. **Graph construction.** Build the service dependency graph from the
   call records of each topology. Edge weights are normalized call
   counts.
2. **Anomaly injection.** For each simulated incident, pick a random
   callee as the root cause. Propagate the fault backward to its callers
   with geometric decay (0.6 per hop, 5 hops) and add half-normal noise
   (the absolute value of Gaussian noise with standard deviation 0.05).
   The visibility parameter sets the anomaly score of the root cause
   itself.
3. **RCA.** Every method ranks all services for the same incidents.
4. **Metrics.** PR@k is the fraction of incidents with the true root
   cause in the top k. MRR is the mean of 1/rank.

Trace-level methods (MicroRank, TraceRank) are evaluated on traces
simulated from the true root cause. They therefore act as an upper
bound and are not part of the aggregate-metric comparison in the paper.
The statistical tests (`statistical_*.csv`) also leave them out.
In the result files, Corr-Fusion is named `CORAL`.

## Using eIRWR on your own graph

```python
import numpy as np
from src.rca import compute_weight_matrix, irwr_enhanced

# edges_df has columns um (caller), dm (callee), request_count
W = compute_weight_matrix(edges_df, nodes)
s_obs = anomaly_scores / anomaly_scores.sum()  # one score per node
scores, _ = irwr_enhanced(W, s_obs)
ranking = [nodes[i] for i in np.argsort(-scores)]
```

## Code layout

```text
src/loader.py          dataset loaders
src/rca/graph.py       dependency graph, weight matrix W, M_R
src/rca/irwr.py        IRWR and eIRWR
src/rca/baselines.py   aggregate-metric baselines and MicroRank
src/rca/tracerank.py   TraceRank and trace simulation
src/rca/simulation.py  anomaly injection
src/rca/metrics.py     PR@k, MRR
src/rca/statistics.py  Friedman and Wilcoxon tests
src/rca/real_eval.py   real anomaly vectors, label-free metrics
src/vis/               paper figures
scripts/               experiment entry points
notebooks/             method tutorial, paper figures, and the Fig. 1-2 example
results/               result CSV files
results/figures/       paper figures (PDF)
data/                  downloaded dataset (git-ignored)
```

## Citation

If you use this code, please cite the paper:

> Saiful Khan and Afrah Farea. eIRWR: Enhanced Iterative Random Walk
> with Restart for Scalable Root Cause Analysis in Microservices.
> arXiv:2608.08073, 2026.

```bibtex
@article{khan2026eirwr,
  title         = {{eIRWR}: Enhanced Iterative Random Walk with Restart
                   for Scalable Root Cause Analysis in Microservices},
  author        = {Khan, Saiful and Farea, Afrah},
  journal       = {arXiv preprint arXiv:2608.08073},
  year          = {2026},
  eprint        = {2608.08073},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2608.08073}
}
```

## License

This project is released under the [MIT License](LICENSE). The Alibaba
trace dataset is distributed separately under its own terms; see the
[Alibaba clusterdata repository](https://github.com/alibaba/clusterdata).
