"""Procrustes alignment benchmark — `procrustes` (Rust) vs `scipy`.

Private dev tool; outputs a markdown table and
``benchmarks/results_procrustes.csv`` so the methods paper can cite a
reproducible head-to-head.

Op coverage:
    orthogonal             : procrustes-rs vs scipy.linalg.orthogonal_procrustes
    signed_permutation     : procrustes-rs only (scipy has no equivalent — that's
                             part of the value prop)

Cells are matched to the criterion sweep plus two practitioner shapes:
    M=32   K ∈ {2, 3, 4, 6, 10, 16}    # criterion parity
    M=300  K=5                          # PLS loadings (n_features, n_LV)
    M=1024 K=10                         # fMRI-ish (parcels, n_LV)

BLAS is pinned to one thread for both sides (the env block at the top
must run before numpy is imported). The Rust runner is single-threaded
by construction (faer with ``Par::Seq`` inside small ops).

Run with any python that has numpy + scipy, e.g.:
    /home/plenartowicz/Projekty/PLSKit-Project/venv/bin/python \\
        benchmarks/bench_procrustes.py

The script will (re)build the Rust runner on first invocation; pass
``--no-build`` to skip when iterating on the Python side.
"""

from __future__ import annotations

import os

# BLAS pin — must precede numpy/scipy import. Fairness: scipy uses
# BLAS DGEMM; without this the host's default thread pool would fight
# the Rust side's single-threaded measurement.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as sla

PROCRUSTES_DIR = Path(__file__).resolve().parent.parent
RUNNER_BIN = PROCRUSTES_DIR / "target" / "release" / "examples" / "bench_runner"
CSV_OUT = Path(__file__).resolve().parent / "results_procrustes.csv"

REPS = 1000          # microsecond ops — many reps for a stable median
SEED = 0xC0_FF_EE    # matches benches/alignment.rs

CELLS: list[tuple[int, int]] = [
    (32, 2), (32, 3), (32, 4), (32, 6), (32, 10), (32, 16),
    (300, 5),
    (1024, 10),
]

OPS_TWO_BACKEND = ["orthogonal"]            # both procrustes and scipy
OPS_PROCRUSTES_ONLY = ["signed_permutation"]  # showcase: no scipy equivalent


# ── data generation & I/O ──────────────────────────────────────────────────

def synth_pair(m: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two M×K matrices in [-1, 1), seeded for cross-call reproducibility.

    NOT bit-identical to the Rust criterion bench (different RNG); residual
    agreement is checked numerically on each cell.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(m, k))
    b = rng.uniform(-1.0, 1.0, size=(m, k))
    return np.asfortranarray(a), np.asfortranarray(b)


def dump_mat(path: Path, x: np.ndarray) -> None:
    """Column-major f64 with [u64 nrows][u64 ncols] LE header."""
    assert x.dtype == np.float64
    x = np.asfortranarray(x)
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", x.shape[0], x.shape[1]))
        f.write(x.tobytes(order="F"))


# ── runners ────────────────────────────────────────────────────────────────

def run_rust(op: str, a_path: Path, b_path: Path, reps: int) -> dict:
    proc = subprocess.run(
        [str(RUNNER_BIN), op, str(a_path), str(b_path), str(reps)],
        check=True, capture_output=True, text=True, timeout=120,
        env={**os.environ, "RAYON_NUM_THREADS": "1"},
    )
    return json.loads(proc.stdout.strip())


def run_scipy_orthogonal(a: np.ndarray, b: np.ndarray, reps: int) -> dict:
    sla.orthogonal_procrustes(a, b)  # warm-up (excluded)
    times_us: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        sla.orthogonal_procrustes(a, b)
        times_us.append((time.perf_counter_ns() - t0) / 1000.0)
    r, _ = sla.orthogonal_procrustes(a, b)
    residual = float(np.linalg.norm(a @ r - b))
    return {
        "backend": "scipy",
        "op": "orthogonal",
        "m": a.shape[0], "k": a.shape[1], "reps": reps,
        "median_us": statistics.median(times_us),
        "min_us": min(times_us),
        "residual": residual,
    }


# ── orchestration ──────────────────────────────────────────────────────────

@dataclass
class Row:
    op: str
    m: int
    k: int
    proc_med_us: float
    proc_min_us: float
    scipy_med_us: float | None
    scipy_min_us: float | None
    residual_proc: float
    residual_scipy: float | None
    residual_match: bool | None  # None when no scipy comparator

    def speedup(self) -> float | None:
        if self.scipy_med_us is None:
            return None
        return self.scipy_med_us / self.proc_med_us


def build_runner() -> None:
    print(f"building bench runner in {PROCRUSTES_DIR} ...", file=sys.stderr)
    subprocess.run(
        ["cargo", "build", "--release", "--example", "bench_runner", "--locked"],
        cwd=PROCRUSTES_DIR, check=True,
    )
    if not RUNNER_BIN.exists():
        sys.exit(f"runner not at {RUNNER_BIN} after build")


def measure_cell(op: str, m: int, k: int, with_scipy: bool, tmpdir: Path) -> Row:
    a, b = synth_pair(m, k, SEED + 17 * m + k)  # cell-unique, deterministic
    a_path, b_path = tmpdir / "a.bin", tmpdir / "b.bin"
    dump_mat(a_path, a)
    dump_mat(b_path, b)

    rust = run_rust(op, a_path, b_path, REPS)

    sci = run_scipy_orthogonal(a, b, REPS) if (with_scipy and op == "orthogonal") else None
    residual_match = None
    if sci is not None:
        # 1e-9 absolute tolerance is generous — both sides solve the same
        # least-squares problem, residuals should agree to floating-point.
        residual_match = abs(rust["residual"] - sci["residual"]) < 1e-9

    return Row(
        op=op, m=m, k=k,
        proc_med_us=rust["median_us"], proc_min_us=rust["min_us"],
        scipy_med_us=sci["median_us"] if sci else None,
        scipy_min_us=sci["min_us"] if sci else None,
        residual_proc=rust["residual"],
        residual_scipy=sci["residual"] if sci else None,
        residual_match=residual_match,
    )


def fmt_us(x: float | None) -> str:
    return "—" if x is None else f"{x:,.2f}"


def render_table(rows: list[Row]) -> str:
    lines = [
        "| op | M | K | procrustes (µs) | scipy (µs) | speedup | residuals match |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        sp = r.speedup()
        match = "—" if r.residual_match is None else ("✓" if r.residual_match else "✗")
        lines.append(
            f"| `{r.op}` | {r.m} | {r.k} "
            f"| **{fmt_us(r.proc_med_us)}** "
            f"| {fmt_us(r.scipy_med_us)} "
            f"| {'—' if sp is None else f'{sp:.2f}×'} "
            f"| {match} |"
        )
    return "\n".join(lines)


def write_csv(rows: list[Row]) -> None:
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "op", "m", "k",
            "procrustes_median_us", "procrustes_min_us",
            "scipy_median_us", "scipy_min_us",
            "residual_procrustes", "residual_scipy", "residual_match",
        ])
        for r in rows:
            w.writerow([
                r.op, r.m, r.k,
                f"{r.proc_med_us:.6f}", f"{r.proc_min_us:.6f}",
                "" if r.scipy_med_us is None else f"{r.scipy_med_us:.6f}",
                "" if r.scipy_min_us is None else f"{r.scipy_min_us:.6f}",
                f"{r.residual_proc:.12e}",
                "" if r.residual_scipy is None else f"{r.residual_scipy:.12e}",
                "" if r.residual_match is None else int(r.residual_match),
            ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-build", action="store_true",
                    help="skip cargo build (assume runner is fresh)")
    args = ap.parse_args()

    if not args.no_build or not RUNNER_BIN.exists():
        build_runner()

    rows: list[Row] = []
    with tempfile.TemporaryDirectory(prefix="procrustes-bench-") as td:
        tmpdir = Path(td)
        for op in OPS_TWO_BACKEND:
            for m, k in CELLS:
                rows.append(measure_cell(op, m, k, with_scipy=True, tmpdir=tmpdir))
        for op in OPS_PROCRUSTES_ONLY:
            for m, k in CELLS:
                rows.append(measure_cell(op, m, k, with_scipy=False, tmpdir=tmpdir))

    print(render_table(rows))
    write_csv(rows)
    print(f"\nwrote {CSV_OUT.relative_to(PROCRUSTES_DIR)}", file=sys.stderr)

    mismatches = [r for r in rows if r.residual_match is False]
    if mismatches:
        for r in mismatches:
            print(f"!! residual mismatch: {r.op} M={r.m} K={r.k} "
                  f"proc={r.residual_proc:.6e} scipy={r.residual_scipy:.6e}",
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
