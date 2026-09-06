"""
Data loading functions for Alibaba Microservice Trace Dataset v2022.

Each loader reads one CSV file from DATA_ROOT and returns a pandas DataFrame.
All loaders accept an optional ``nrows`` parameter so we can sample large files
quickly during exploration.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd

DATA_ROOT: Path = Path(__file__).resolve().parent.parent / "data" / "data"
CALLGRAPH_DIR: Path = DATA_ROOT / "CallGraph"

# ── Column dtypes (keep memory usage low on 10M+ row files) ─────────

_CALL_GRAPH_DTYPES: dict[str, str] = {
    "timestamp": "int64",
    "traceid": "str",
    "service": "str",
    "rpc_id": "str",
    "rpctype": "str",
    "um": "str",
    "uminstanceid": "str",
    "interface": "str",
    "dm": "str",
    "dminstanceid": "str",
    "rt": "float64",
}

_MS_METRICS_DTYPES: dict[str, str] = {
    "timestamp": "int64",
    "msname": "str",
    "msinstanceid": "str",
    "nodeid": "str",
    "cpu_utilization": "float64",
    "memory_utilization": "float64",
}

_NODE_METRICS_DTYPES: dict[str, str] = {
    "timestamp": "int64",
    "nodeid": "str",
    "cpu_utilization": "float64",
    "memory_utilization": "float64",
}


# ── Loaders ──────────────────────────────────────────────────────────


def load_call_graph(
    path: str | Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load the MSCallGraph distributed-trace dataset.

    Columns: timestamp, traceid, service, rpc_id, rpctype, um, uminstanceid,
             interface, dm, dminstanceid, rt
    """
    path: str | Path = path or DATA_ROOT / "CallGraph" / "CallGraph_0.csv"
    return pd.read_csv(
        path,
        dtype=_CALL_GRAPH_DTYPES,
        engine="python",
        on_bad_lines="skip",
        nrows=nrows,
    )


def list_call_graph_csvs() -> list[Path]:
    """Return sorted list of already-extracted CallGraph CSV files."""
    return sorted(
        CALLGRAPH_DIR.glob("CallGraph_*.csv"),
        key=lambda p: int(p.stem.split("_")[1]),
    )


def load_ms_metrics(
    path: str | Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load the MSMetrics (container-level CPU / memory) dataset.

    Columns: timestamp, msname, msinstanceid, nodeid, cpu_utilization,
             memory_utilization
    """
    path: str | Path = path or DATA_ROOT / "MSMetrics" / "MSMetricsUpdate_0.csv"
    return pd.read_csv(path, dtype=_MS_METRICS_DTYPES, nrows=nrows)


def load_msrt_mcr(
    path: str | Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load the MSRTMCR (response-time & call-rate) dataset.

    Contains 10 metric pairs (*_rt, *_mcr) for rpc/db/mc/mq/http.
    """
    path: str | Path = path or DATA_ROOT / "MSRTMCR" / "MCRRTUpdate_0.csv"
    return pd.read_csv(path, nrows=nrows)


def load_node_metrics(
    path: str | Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load the NodeMetrics (bare-metal node CPU / memory) dataset.

    Columns: timestamp, nodeid, cpu_utilization, memory_utilization
    """
    path: str | Path = (
        path or DATA_ROOT / "NodeMetrics" / "NodeMetricsUpdate_0.csv"
    )
    return pd.read_csv(path, dtype=_NODE_METRICS_DTYPES, nrows=nrows)


# ── Indexed loaders for MSMetrics / MSRTMCR ───────────────────────────

MSMETRICS_DIR: Path = DATA_ROOT / "MSMetrics"
MSRTMCR_DIR: Path = DATA_ROOT / "MSRTMCR"


def load_ms_metrics_by_index(
    index: int,
    nrows: int | None = None,
    *,
    extract: bool = True,
) -> pd.DataFrame:
    """Load MSMetrics CSV by archive index, extracting from tar.gz if needed."""
    csv_path = MSMETRICS_DIR / f"MSMetricsUpdate_{index}.csv"
    if not csv_path.exists():
        if extract:
            tar_path = MSMETRICS_DIR / f"MSMetricsUpdate_{index}.tar.gz"
            if tar_path.exists():
                import tarfile
                with tarfile.open(tar_path, "r:gz") as tf:
                    tf.extractall(MSMETRICS_DIR, filter="data")
            else:
                raise FileNotFoundError(f"Archive not found: {tar_path}")
        else:
            raise FileNotFoundError(f"CSV not found: {csv_path}")
    return load_ms_metrics(path=csv_path, nrows=nrows)


def load_msrt_mcr_by_index(
    index: int,
    nrows: int | None = None,
    *,
    extract: bool = True,
) -> pd.DataFrame:
    """Load MSRTMCR CSV by archive index, extracting from tar.gz if needed."""
    csv_path = MSRTMCR_DIR / f"MCRRTUpdate_{index}.csv"
    if not csv_path.exists():
        if extract:
            tar_path = MSRTMCR_DIR / f"MCRRTUpdate_{index}.tar.gz"
            if tar_path.exists():
                import tarfile
                with tarfile.open(tar_path, "r:gz") as tf:
                    tf.extractall(MSRTMCR_DIR, filter="data")
            else:
                raise FileNotFoundError(f"Archive not found: {tar_path}")
        else:
            raise FileNotFoundError(f"CSV not found: {csv_path}")
    return load_msrt_mcr(path=csv_path, nrows=nrows)


# CallGraph whole dataset related functions


def load_call_graph_by_index(
    index: int,
    nrows: int | None = None,
    *,
    extract: bool = True,
) -> pd.DataFrame:
    """Load a CallGraph CSV by its archive index.

    Parameters
    ----------
    index : int
        Archive index (0–499).
    nrows : int, optional
        Limit rows read.
    extract : bool
        If True and CSV doesn't exist, extract from tar.gz first.
    """
    csv_path = CALLGRAPH_DIR / f"CallGraph_{index}.csv"
    if not csv_path.exists():
        if extract:
            extract_call_graph(index)
        else:
            raise FileNotFoundError(f"CSV not found: {csv_path}")
    return load_call_graph(path=csv_path, nrows=nrows)


def list_call_graph_archives() -> list[int]:
    """Return sorted list of available CallGraph archive indices (0–499)."""
    return sorted(
        int(p.name.split("_")[1].split(".")[0])
        for p in CALLGRAPH_DIR.glob("CallGraph_*.tar.gz")
    )


def extract_call_graph(index: int, *, force: bool = False) -> Path:
    """Extract a single CallGraph tar.gz and return the CSV path.

    Parameters
    ----------
    index : int
        Archive index (e.g. 0 for CallGraph_0.tar.gz).
    force : bool
        Re-extract even if the CSV already exists.

    Returns
    -------
    Path to the extracted CSV file.
    """
    csv_path = CALLGRAPH_DIR / f"CallGraph_{index}.csv"
    if csv_path.exists() and not force:
        return csv_path
    tar_path = CALLGRAPH_DIR / f"CallGraph_{index}.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"Archive not found: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(CALLGRAPH_DIR, filter="data")
    return csv_path


def extract_all_call_graphs(*, force: bool = False) -> list[Path]:
    """Extract all CallGraph tar.gz archives. Returns list of CSV paths."""
    paths = []
    for idx in list_call_graph_archives():
        paths.append(extract_call_graph(idx, force=force))
    return paths


def scan_call_graph_stats(
    index: int,
    *,
    extract: bool = True,
) -> dict:
    """Scan a CallGraph CSV for graph statistics without full loading.

    Reads only the um and dm columns to compute node/edge counts efficiently.

    Returns
    -------
    dict with keys: index, n_rows, n_unique_um, n_unique_dm, n_nodes,
    n_edges, csv_path, file_size_mb.
    """
    csv_path = CALLGRAPH_DIR / f"CallGraph_{index}.csv"
    if not csv_path.exists():
        if extract:
            extract_call_graph(index)
        else:
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        usecols=["um", "dm"],
        dtype={"um": "str", "dm": "str"},
        engine="python",
        on_bad_lines="skip",
    )
    unique_um = set(df["um"].dropna()) - {"UNKNOWN"}
    unique_dm = set(df["dm"].dropna()) - {"UNKNOWN"}
    all_nodes = unique_um | unique_dm
    edges = df[
        ~df["um"].isin(["UNKNOWN"]) & ~df["dm"].isin(["UNKNOWN"])
    ].dropna(subset=["um", "dm"])
    n_unique_edges = len(edges.drop_duplicates(subset=["um", "dm"]))

    return {
        "index": index,
        "n_rows": len(df),
        "n_unique_um": len(unique_um),
        "n_unique_dm": len(unique_dm),
        "n_nodes": len(all_nodes),
        "n_unique_edges": n_unique_edges,
        "csv_path": str(csv_path),
        "file_size_mb": csv_path.stat().st_size / (1024 * 1024),
    }
