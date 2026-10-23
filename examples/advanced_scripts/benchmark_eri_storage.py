# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reproducible benchmark for GradDFT symmetry-compressed ERI storage (#15).

Each measurement runs in its own child process (``--child`` is internal) so
that peak RSS is reported per ``(format, mode)`` without pollution from the
other configurations. The parent spawns the children, collects one JSON line
per child and prints a markdown table.

Notes
-----
- Linux/macOS only (``resource.getrusage``): run it on the machine whose
  numbers you want to quote.
- ``jax_enable_x64`` is enabled, matching the test suite (float64 storage).
- "compile" is the first jitted call after tracing (compile + execution);
  "steady" is the median of ``--repeats`` calls with ``jax.block_until_ready``
  after warmup. CPU numbers are for guidance, not a hardware-neutral claim.
- Memory and runtime are reported separately; no speedup is claimed.

Examples
--------
    python examples/advanced_scripts/benchmark_eri_storage.py --molecule H2O --basis cc-pvdz
    python examples/advanced_scripts/benchmark_eri_storage.py --mode generate --formats dense,s8
"""

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time

from jax import config

import jax
import jax.numpy as jnp
from jax.nn import gelu, sigmoid
from jax.random import PRNGKey
import numpy as np
from flax import linen as nn
from pyscf import dft, gto

from grad_dft import (
    Molecule,
    NeuralFunctional,
    molecule_from_pyscf,
    non_scf_predictor,
    simple_energy_loss,
)
from grad_dft.utils.eri import PackedERI, packed_to_dense

config.update("jax_enable_x64", True)

RESULT_PREFIX = "ERI_BENCH_RESULT "
MIB = float(2**20)
SCF_ITERS = 200
STEADY_REPEATS = 20
WARMUP_REPEATS = 3
FORMATS = ("dense", "s4", "s8")

# Geometries in Angstrom (one atom per line: PySCF's atom parser).
GEOMETRIES = {
    "H2": "H 0.0 0.0 -0.3705\nH 0.0 0.0 0.3705",
    "H2O": "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692",
    "CH4": (
        "C 0.0 0.0 0.0\nH 0.6291 0.6291 0.6291\nH -0.6291 -0.6291 0.6291\n"
        "H -0.6291 0.6291 -0.6291\nH 0.6291 -0.6291 -0.6291"
    ),
    "C6H6": (
        "C 1.2090 0.0000 0.0000\nC 0.6045 1.0470 0.0000\nC -0.6045 1.0470 0.0000\n"
        "C -1.2090 0.0000 0.0000\nC -0.6045 -1.0470 0.0000\nC 0.6045 -1.0470 0.0000\n"
        "H 2.1490 0.0000 0.0000\nH 1.0745 1.8610 0.0000\nH -1.0745 1.8610 0.0000\n"
        "H -2.1490 0.0000 0.0000\nH -1.0745 -1.8610 0.0000\nH 1.0745 -1.8610 0.0000"
    ),
}


def converged_mf(molecule: str, basis: str):
    r"""Converge a spin-restricted Kohn-Sham calculation with no XC energy.

    ``molecule`` is a key of :data:`GEOMETRIES`.
    """
    mol = gto.M(atom=GEOMETRIES[molecule], basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "0.00*LDA"  # no XC energy: isolates the ERI-dependent terms
    mf.max_cycle = SCF_ITERS
    mf.kernel()
    return mf


def build_molecule(molecule: str, basis: str, fmt: str):
    r"""Converge SCF and return ``(mf, molecule, scf_seconds)``."""
    start = time.perf_counter()
    mf = converged_mf(molecule, basis)
    scf_seconds = time.perf_counter() - start
    mol = molecule_from_pyscf(mf, eri_storage=fmt)
    jax.block_until_ready(mol.rep_tensor)
    return mf, mol, scf_seconds


def timed_steady(fn, repeats: int):
    r"""Return ``(compile_seconds, steady_median_seconds)`` for a jitted call."""
    start = time.perf_counter()
    jax.block_until_ready(fn())
    compile_seconds = time.perf_counter() - start
    for _ in range(WARMUP_REPEATS):
        jax.block_until_ready(fn())
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        durations.append(time.perf_counter() - start)
    return compile_seconds, statistics.median(durations)


def storage_stats(rep_tensor):
    r"""Physical storage shape and bytes of a dense array or PackedERI."""
    array = rep_tensor.packed if isinstance(rep_tensor, PackedERI) else rep_tensor
    return {
        "stored_shape": list(np.shape(array)),
        "stored_bytes": int(array.size * array.dtype.itemsize),
    }


def peak_rss_mib():
    r"""Peak resident set size of this process in MiB (Linux: KiB units)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def tracked_peak_mib():
    r"""Peak device memory in MiB if the backend reports it, else None."""
    try:
        stats = jax.devices()[0].memory_stats()
    except (AttributeError, NotImplementedError, RuntimeError):
        return None
    if not stats or "peak_bytes_in_use" not in stats:
        return None
    return stats["peak_bytes_in_use"] / MIB


def coefficient_inputs(molecule: Molecule, *_, **__):
    r"""Single-feature density input (clipped), following the test suite."""
    rho = jnp.clip(molecule.density(), min=1e-30)
    return jnp.concatenate((rho,), axis=1)


def energy_densities(molecule: Molecule, *_, **__):
    r"""LSDA exchange energy density feature, following the test suite."""
    rho = jnp.clip(molecule.density(), min=1e-30)
    return (
        -3
        / 2
        * (3 / (4 * jnp.pi)) ** (1 / 3)
        * (rho ** (4 / 3)).sum(axis=1, keepdims=True)
    )


def coefficients(_instance, rhoinputs):
    r"""A minimal parametric energy-density head (one trainable Dense layer)."""
    x = nn.Dense(features=1)(rhoinputs)
    x = nn.LayerNorm()(x)
    return sigmoid(gelu(x))


def measure_generate(args) -> dict:
    r"""ERI generation: SCF time, packing time, storage shape and bytes."""
    start = time.perf_counter()
    mf = converged_mf(args.molecule, args.basis)
    scf_seconds = time.perf_counter() - start

    start = time.perf_counter()
    mol = molecule_from_pyscf(mf, eri_storage=args.format)
    jax.block_until_ready(mol.rep_tensor)
    generate_seconds = time.perf_counter() - start

    rep_tensor = mol.rep_tensor
    n_orbitals = (
        rep_tensor.n_orbitals
        if isinstance(rep_tensor, PackedERI)
        else rep_tensor.shape[0]
    )
    return {
        "n_orbitals": int(n_orbitals),
        "scf_seconds": scf_seconds,
        "generate_seconds": generate_seconds,
        **storage_stats(rep_tensor),
    }


def measure_contraction(args) -> dict:
    r"""Coulomb contraction on stored ERIs: to_dense, compile and steady time."""
    _, mol, scf_seconds = build_molecule(args.molecule, args.basis, args.format)
    to_dense_seconds = None
    if isinstance(mol.rep_tensor, PackedERI):
        start = time.perf_counter()
        jax.block_until_ready(packed_to_dense(mol.rep_tensor))
        to_dense_seconds = time.perf_counter() - start

    def contraction(molecule):
        return molecule.get_coulomb_potential()

    jit_contraction = jax.jit(contraction)
    compile_seconds, steady_seconds = timed_steady(
        lambda: jit_contraction(mol), args.repeats
    )
    return {
        "scf_seconds": scf_seconds,
        "to_dense_seconds": to_dense_seconds,
        "contraction_compile_seconds": compile_seconds,
        "contraction_steady_seconds": steady_seconds,
    }


def measure_train(args) -> dict:
    r"""One training step (loss + parameter gradients) on stored ERIs."""
    mf, mol, scf_seconds = build_molecule(args.molecule, args.basis, args.format)
    functional = NeuralFunctional(coefficients, energy_densities, coefficient_inputs)
    params = functional.init(PRNGKey(args.seed), coefficient_inputs(mol))
    predictor = non_scf_predictor(functional)
    truth_energy = mf.e_tot

    def step(params, molecule):
        (loss, _), grads = simple_energy_loss(params, predictor, molecule, truth_energy)
        return loss, grads

    jit_step = jax.jit(step)
    compile_seconds, steady_seconds = timed_steady(
        lambda: jit_step(params, mol), args.repeats
    )
    loss_value = float(jit_step(params, mol)[0])
    return {
        "scf_seconds": scf_seconds,
        "train_compile_seconds": compile_seconds,
        "train_steady_seconds": steady_seconds,
        "train_loss": loss_value,
    }


MEASUREMENTS = {
    "generate": measure_generate,
    "contraction": measure_contraction,
    "train": measure_train,
}


def run_child(args) -> int:
    r"""Run one (format, mode) measurement and emit a single JSON result line."""
    payload = MEASUREMENTS[args.mode](args)
    payload.update(
        {
            "mode": args.mode,
            "format": args.format,
            "molecule": args.molecule,
            "basis": args.basis,
            "peak_rss_mib": peak_rss_mib(),
            "tracked_peak_mib": tracked_peak_mib(),
        }
    )
    print(RESULT_PREFIX + json.dumps(payload))
    return 0


def run_parent(args) -> int:
    r"""Spawn one child per (format, mode) and print the aggregated table."""
    modes = ["generate", "contraction", "train"] if args.mode == "all" else [args.mode]
    results = {}
    for fmt in args.formats:
        for mode in modes:
            command = [
                sys.executable,
                __file__,
                "--child",
                "--molecule",
                args.molecule,
                "--basis",
                args.basis,
                "--format",
                fmt,
                "--mode",
                mode,
                "--repeats",
                str(args.repeats),
                "--seed",
                str(args.seed),
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            lines = [
                line
                for line in completed.stdout.splitlines()
                if line.startswith(RESULT_PREFIX)
            ]
            if completed.returncode != 0 or not lines:
                sys.stderr.write(completed.stdout)
                sys.stderr.write(completed.stderr)
                raise SystemExit(
                    f"child failed for format={fmt!r} mode={mode!r} "
                    f"(exit {completed.returncode})"
                )
            results.setdefault(fmt, {}).update(
                json.loads(lines[-1][len(RESULT_PREFIX) :])
            )
    print(render_table(results))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        print(f"\nRaw results written to {args.json_out}")
    return 0


def _cell(value, unit="", precision=4):
    r"""Format a table cell, using ``n/a`` for unavailable measurements."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}{unit}"
    return f"{value}{unit}"


def render_table(results: dict) -> str:
    r"""Render the aggregated per-format results as a markdown table."""
    columns = [
        ("stored shape", lambda row: _shape(row.get("stored_shape"))),
        ("stored size", lambda row: _cell(_to_mib(row.get("stored_bytes")), " MiB", 4)),
        ("generate (s)", lambda row: _cell(row.get("generate_seconds"))),
        ("to_dense (s)", lambda row: _cell(row.get("to_dense_seconds"))),
        (
            "coul. compile (s)",
            lambda row: _cell(row.get("contraction_compile_seconds")),
        ),
        (
            "coul. steady (s)",
            lambda row: _cell(row.get("contraction_steady_seconds"), "", 6),
        ),
        ("train compile (s)", lambda row: _cell(row.get("train_compile_seconds"))),
        ("train steady (s)", lambda row: _cell(row.get("train_steady_seconds"), "", 6)),
        ("peak RSS (MiB)", lambda row: _cell(row.get("peak_rss_mib"), "", 1)),
        ("tracked peak (MiB)", lambda row: _cell(row.get("tracked_peak_mib"), "", 1)),
    ]
    header = "| format | " + " | ".join(name for name, _ in columns) + " |"
    divider = "| --- | " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for fmt in FORMATS:
        row = results.get(fmt)
        if row is None:
            continue
        cells = [fmt] + [render(row) for _, render in columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _shape(shape):
    r"""Render a stored shape (or ``n/a``), keeping 1-D shapes unambiguous."""
    if shape is None:
        return "n/a"
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def _to_mib(byte_count):
    r"""Convert bytes to MiB, or ``None``."""
    if byte_count is None:
        return None
    return byte_count / MIB


def parse_args(argv=None):
    r"""Parse command-line arguments (``--child`` is internal)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=sorted(GEOMETRIES), default="H2O")
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument(
        "--mode", choices=["generate", "contraction", "train", "all"], default="all"
    )
    parser.add_argument("--repeats", type=int, default=STEADY_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None, help="optional results file")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--format", choices=FORMATS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    selected = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    invalid = [fmt for fmt in selected if fmt not in FORMATS]
    if invalid:
        parser.error(
            f"unknown format(s) {invalid}; expected a comma-separated subset of {FORMATS}"
        )
    args.formats = selected
    if args.child and args.format is None:
        parser.error("--format is required for child measurements")
    return args


def main(argv=None) -> int:
    r"""Dispatch to the child measurement or the parent aggregator."""
    args = parse_args(argv)
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
