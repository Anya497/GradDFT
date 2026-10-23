These last two examples are pretty slow and require to have precomputed the molecule in an hdf5 file in advance. Please modify accordingly lines

```python
training_data_dirpath = os.path.normpath(dirpath + "/data/training/dissociation/")
training_files = ["H2_dissociation_omegas.h5"]
```

in those two files. In the case of `train_complex_model.py`, the hdf5 file should have processed the molecules with

```python
from grad_dft.interface.pyscf import molecule_from_pyscf
omegas = jnp.array([0., 0.4])
molecule = molecule_from_pyscf(mf, omegas = omegas)
```

The `test_constraints.py` file executes an example showing that constraints run. However, note that they have not been tested thoroughly.

## Benchmarking compressed ERI storage

`benchmark_eri_storage.py` measures the memory and runtime footprint of the
three ERI storage formats (`dense`, `s4`, `s8`) end to end. Each measurement
runs in its own child process, so peak RSS is reported per `(format, mode)`;
the parent aggregates the JSON lines into one markdown table.

```bash
python examples/advanced_scripts/benchmark_eri_storage.py --molecule H2O --basis cc-pvdz
```

Modes (`--mode`):

- `generate` — SCF time, ERI generation/packing time, physical storage shape and bytes.
- `contraction` — `to_dense()` time (packed formats) and `Molecule.get_coulomb_potential()` compile vs steady-state time.
- `train` — one training step (a tiny `NeuralFunctional` with `simple_energy_loss`, including parameter gradients) compile vs steady-state time.
- `all` (default): the three above.

Measurement notes:

* float64 (`jax_enable_x64`) is enabled, matching the test suite; the tiny
  functional uses a fixed seed and SCF uses `xc="0.00*LDA"`, so only
  ERI-dependent terms are timed.
* "compile" is the first jitted call (tracing + compilation + execution);
  "steady" is the median of `--repeats` calls after warmup, each forced to
  completion with `jax.block_until_ready`. `to_dense()` is a plain gather (not
  jitted) and is timed once.
* Peak RSS is Linux `ru_maxrss` per child (`device.memory_stats()` is used when
  the backend reports it). It includes the Python/JAX runtime (roughly 0.8-1.7
  GiB here), so read it as a build-time footprint; the `stored size` column is
  the exact ERI payload.
* Numbers are CPU-only and machine-specific: re-run to get your own.