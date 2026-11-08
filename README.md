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

To regenerate the figures in `paper/figures/` from the CSV files:

```sh
uv run python scripts/make_paper_figures.py
```

## Requirements

- Linux or macOS with `bash`, `wget` and `tar`.
- Python >= 3.13 and [uv](https://docs.astral.sh/uv/). `uv sync`
  installs the exact package versions in `uv.lock`.
- About 15 GB of free disk space for the dataset.
- Several CPU cores. The main comparison runs in parallel. The full
  run took about 75 minutes on a 28-core machine with 196 GB of RAM.

## Step details

### 1. Install

```sh
uv sync
```

### 2. Dataset

The experiments use the
[Alibaba Microservice Trace Dataset v2022](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022).
`scripts/download_data.sh` downloads the five files below into
`data/data/` and extracts them (about 0.9 GB download, 13 GB
extracted). The script can be re-run safely: it skips finished files
and resumes partial downloads.

| File | Used for |
| --- | --- |
| `CallGraph/CallGraph_{84,136,297}.csv` | the three topologies CG84, CG136, CG297 |
| `MSRTMCR/MCRRTUpdate_0.csv` | real anomaly windows |
| `MSMetrics/MSMetricsUpdate_0.csv` | optional CPU/memory statistics |

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
| `scripts/analyze_real_metrics.py --nrows 1000000` | `real_anomaly_windows.csv` | input to Table real-world |
| `scripts/evaluate_performance.py --experiments comparison` | `full_comparison_metrics_CG*.csv`, `per_incident_ranks_CG*.csv`, `multi_topology_metrics.csv` | Table main-results, MRR figures |
| `scripts/evaluate_performance.py --experiments compute` | `computational_metrics.csv` | Table computational, scalability figure |
| `scripts/evaluate_performance.py --experiments convergence` | `convergence_metrics.csv` | convergence figure |
| `scripts/evaluate_performance.py --experiments stats` | `statistical_*.csv` | Table Wilcoxon, Friedman test |
| `scripts/evaluate_performance.py --experiments real` | `real_world_evaluation.csv` | Table real-world |
| `scripts/run_ablation.py --dataset-index 84` | `ablation_metrics.csv` | Table ablation |

The `stats` and `real` experiments read the output of `comparison` and
`analyze_real_metrics.py`, so run those first.

Useful options of `evaluate_performance.py` (see `--help`):

- `--quick`: a smaller configuration (4 visibility levels, 50
  incidents) to check that everything runs.
- `--resume`: skip topologies whose comparison CSV already exists.
- `--seed`: random seed (default 42). Results depend only on the seed,
  not on the number of workers.

### Topology selection (optional)

`scripts/select_topologies.py` reproduces how CG84, CG136 and CG297
were chosen and writes `results/topology_stats.csv` and
`results/selected_topologies.csv` (Table dataset-stats). It scans all
500 call-graph archives, so it needs the full CallGraph download
(`fetchData.sh` in the Alibaba repository).

## Evaluation protocol

1. **Graph construction.** Build the service dependency graph from the
   call records of each topology. Edge weights are normalized call
   counts.
2. **Anomaly injection.** For each simulated incident, pick a random
   callee as the root cause. Propagate the fault backward to its callers
   with geometric decay (0.6 per hop, 5 hops) and add Gaussian noise
   (standard deviation 0.05). The visibility parameter sets the anomaly
   score of the root cause itself.
3. **RCA.** Every method ranks all services for the same incidents.
4. **Metrics.** PR@k is the fraction of incidents with the true root
   cause in the top k. MRR is the mean of 1/rank.

Trace-level methods (MicroRank, TraceRank) are evaluated on traces
simulated from the true root cause. They therefore act as an upper
bound and are not part of the aggregate-metric comparison in the paper.
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
src/rca/statistics.py  Friedman, Wilcoxon, Cliff's delta, Nemenyi
src/rca/real_eval.py   real anomaly vectors, label-free metrics
src/vis/               paper figures
scripts/               experiment entry points
notebooks/             method tutorial and figure notebook
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
