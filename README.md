# eIRWR -- Enhanced Iterative Random Walk with Restart

Root Cause Analysis (RCA) for microservice architectures using Enhanced Iterative Random Walk with Restart.

---
## Setup

Requires Python >= 3.13.

```sh
uv sync
```

---
## Dataset

The [Alibaba Microservice Trace Dataset v2022](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022) provides real-world traces from a large-scale production microservice cluster.

To download the data:

```sh
bash data/clusterdata/cluster-trace-microservices-v2022/fetchData.sh start_date=0d0 end_date=1d1
```

One CSV file from each subfolder should be unzipped into `data/data/`. The paper experiments use `CallGraph_84.csv`, `CallGraph_136.csv`, and `CallGraph_297.csv` (referenced by index, e.g. `--dataset-index 297`). Use `scripts/select_topologies.py` to scan candidate call graphs and reproduce the topology selection.

| Dataset     | Rows (sampled) | Key columns                                                       |
| ----------- | -------------- | ----------------------------------------------------------------- |
| CallGraph   | 500K           | traceid, um (caller), dm (callee), rt (response time ms), rpctype |
| MSMetrics   | 500K           | msname, cpu_utilization, memory_utilization                       |
| MSRTMCR     | 500K           | msname, {rpc,db,mc,mq,http}\_{rt,mcr}                             |
| NodeMetrics | 500K           | nodeid, cpu_utilization, memory_utilization                       |

---
## Running experiments

The project includes both automated Python scripts and interactive Jupyter notebooks for running experiments.

### Option 1: Automated evaluation script (recommended)

Run all experiments using the consolidated evaluation script:

```sh
# Run all experiments (default: 200 incidents, 7 visibility levels, dataset 297)
uv run python scripts/evaluate_performance.py

# Quick mode (faster, fewer configurations)
uv run python scripts/evaluate_performance.py --quick

# Custom configuration
uv run python scripts/evaluate_performance.py --n-incidents 100 --dataset-index 297

# Run specific experiments (1-7)
uv run python scripts/evaluate_performance.py --experiments 1 2 3

# Multi-topology evaluation
uv run python scripts/evaluate_performance.py --dataset-indices 297 298 299
```

**Parallelized version** (for multi-core machines):

```sh
# Run all experiments in parallel using all available CPUs
uv run python scripts/evaluate_performance_parallel.py

# Control parallelism (e.g., use 64 CPUs)
uv run python scripts/evaluate_performance_parallel.py --n-jobs 64

# Quick mode with parallelization
uv run python scripts/evaluate_performance_parallel.py --quick

# Run specific experiments in parallel
uv run python scripts/evaluate_performance_parallel.py --experiments 1 2 3

# Run in background (continues even if you close terminal/editor)
nohup uv run python scripts/evaluate_performance_parallel.py > experiments.log 2>&1 &
```

The parallelized script uses `joblib` to distribute workload across multiple CPUs: Parallelizes incident simulations within each experiment; Runs method benchmarks concurrently; Significantly faster on multi-core systems (tested on 126 CPUs); Same results as sequential script (reproducible with same seed)

Monitoring parallel execution:

While the script runs, you can monitor progress using these commands:

```sh
# Check running processes
ps aux | grep evaluate_performance_parallel | grep -v grep

# Monitor CPU and memory usage
top -b -n 1 | head -20

# View progress in real-time (if running in background)
# The background process ID is shown when you start the script

# Check generated results
ls -lth results/*.csv | head -10

# Monitor progress in real-time
tail -f experiments.log
```

**Expected execution time** (with 126 CPUs, 200 incidents per configuration):
- Experiment 1 (Scale × Visibility): ~20-30 minutes (6 dataset sizes × 7 visibilities)
- Experiment 2 (Full Comparison): ~15-20 minutes (7 visibilities on full dataset)
- Experiment 3 (Computational Benchmarks): ~5-10 minutes (5 dataset sizes × 13 methods)
- Experiment 4 (Convergence Analysis): ~10-15 minutes (4 dataset sizes × 7 alphas × 20 trials)
- Experiment 5 (Per-Incident Ranks): ~15-20 minutes (7 visibilities × 200 incidents)
- **Total**: ~65-95 minutes for all experiments

**Available experiments:**
1. **Scale × Visibility**: Evaluate methods across dataset sizes (100K-10M rows) and visibility levels (0.0-0.5)
2. **Full Comparison**: Compare all methods on the largest dataset with all baselines
3. **Computational Benchmarks**: Measure wall-clock time and peak memory for each method
4. **Convergence Analysis**: Measure IRWR iteration counts vs alpha and graph size
5. **Per-Incident Analysis**: Save detailed per-incident ranks for statistical tests
6. **Statistical Tests**: Run Friedman and Wilcoxon signed-rank tests on results from Experiment 5
7. **Real-World Evaluation**: Evaluate methods on real anomaly windows (requires running `analyze_real_metrics.py` first)

**Results** are saved as CSV files in `results/`:
- `scale_visibility_metrics.csv` - Experiment 1
- `full_comparison_metrics.csv` - Experiment 2
- `computational_metrics.csv` - Experiment 3
- `convergence_metrics.csv` - Experiment 4
- `per_incident_ranks.csv` - Experiment 5
- `statistical_*.csv` - Experiment 6
- `real_world_evaluation.csv` - Experiment 7

### Option 2: Analyze real metrics (optional)

Extract real anomaly patterns from the dataset:

```sh
# Analyze real metrics and detect anomaly windows
uv run python scripts/analyze_real_metrics.py

# Custom configuration
uv run python scripts/analyze_real_metrics.py --call-graph-index 297 --nrows 1000000 --z-threshold 3.0
```

This generates:
- `results/rt_distributions.csv` - Per-service response time statistics
- `results/cpu_distributions.csv` - Per-service CPU/memory statistics
- `results/real_anomaly_windows.csv` - Detected real anomaly timestamps
- `results/anomaly_cascades.csv` - Temporal cascade patterns

### Option 3: Analysis

```sh
# Regenerate the four figures in paper/figures/ from results/*.csv
uv run python scripts/make_paper_figures.py

# Ablation study (Table 3 in the paper): PPR vs transpose vs tuned restart vs eIRWR
uv run python scripts/run_ablation.py --dataset-index 84

# Wilcoxon signed-rank tests + effect sizes (eIRWR vs each baseline)
uv run python scripts/run_wilcoxon_eirwr.py
```

## Experiments

The experiment pipeline evaluates the RWR algorithm using a chaos engineering approach:

1. **Graph construction** -- Build the dependency graph from the sampled call records of each topology
2. **Anomaly injection** -- For each simulated incident, randomly select a root cause node and propagate cascading failures backward through the graph with geometric decay and background noise
3. **RCA execution** -- Run the iterative RWR to produce a ranked list of root cause candidates
4. **Evaluation** -- Measure how accurately the algorithm identifies the true root cause

### Evaluation metrics

- **PR@K** (Precision at K): fraction of incidents where the true root cause appears in the top K candidates
- **MRR** (Mean Reciprocal Rank): average of 1/rank across all incidents (higher is better, max 1.0)
