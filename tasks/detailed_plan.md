# Detailed Plan — #6 Restore the DM21 test: port the vendored numint to TF-Hub 0.16 and PySCF 2.13

Integration branch: `agent_settings` (stacked on `main`).
Feature branch: `restore_dm21_test` (created from the rebased integration branch).

Problem: `tests/integration/molecules/test_functional_implementations.py::test_dm21`
fails on the current dependency set (`tensorflow 2.21.0`, `tensorflow-hub 0.16.1`,
`pyscf 2.13.1`) with two independent upstream breaks stacked in the vendored
DeepMind code `grad_dft/external/density_functional_approximation_dm21/.../neural_numint.py`:

1. **TF-Hub 0.16 removed the TF1 `Module` API** — `hub.Module` raised
   `AttributeError` at `neural_numint.py:219` (reproduced on the feature branch
   before any change).
2. **PySCF 2.13's `eval_xc_eff` no longer routes through `eval_xc`** —
   `nr_rks`/`nr_uks` now call `ni.eval_xc_eff(...)`, which is implemented as
   `eval_xc1` + `xc_deriv.transform_xc`. `NeuralNumInt` overrides `eval_xc` but
   not `eval_xc1`, so the call falls through to libxc, which resolves the
   SCF's `xc_code` (the default `LDA,VWN`) against a meta-GGA-shaped density and
   dies with `ValueError: cannot reshape array of size 240720 into shape
   (2,1,24072)`. `NeuralNumInt.eval_xc` is dead code today.

The issue's acceptance criteria: `test_dm21` passes with no tolerance change;
the `spin=0` path is either covered by a new test or explicitly rejected with a
clear `NotImplementedError`; the `hub.load` port and the PySCF shim are separate
commits; `export_functional_and_derivatives()` gets an explicit outcome.

## Verification performed during planning (prototypes in /tmp/opencode, no repo changes)

- **`hub.load` port is viable.** `hub.load(checkpoint).signatures["default"]`
  loads the vendored TF1 SavedModel under `tf.disable_v2_behavior()` graph mode.
  The signature's keyword keys keep the `$` separator (`tensor_dict$rho_a`,
  only the TensorSpec `name=` fields are sanitised to `_`), so
  `self._functional(**{f"tensor_dict${k}": v ...})` works as-is — the issue's
  `$`→`_` mangling claim is not needed. The output key is `grid_contribution`
  (unchanged). `tf.gradients` through the resulting `PartitionedCall` works and
  returns finite gradients with the expected shapes (`vrho` 2×(N,1), `vsigma`
  3×(N,1), `vtau` 2×(N,1), `vhf` 2×(N,2)) — i.e. the whole existing
  derivative machinery survives the port untouched.
- **The `eval_xc_eff` chain rule is settled, not guessed.** The layout PySCF
  expects was validated against PySCF's own `NumInt.eval_xc_eff` on the real
  libxc functional `mgga_x_tpss` (spin=1 and spin=0): building the effective
  density-parameter derivatives from `eval_xc`'s `(vrho, vsigma, vtau)` with

  | spin | row layout of `vxc` |
  |---|---|
  | 0 | `(5, N)`: `[vrho, 2·vsigma·∇ρ, vtau]` |
  | 1 | `(2, 5, N)`: `[:,0]=vrho`, `[0,1:4]=2·vsigma_aa·∇ρ_a + vsigma_ab·∇ρ_b`, `[1,1:4]=2·vsigma_bb·∇ρ_b + vsigma_ab·∇ρ_a`, `[:,4]=vtau` |

  reproduces `eval_xc_eff`'s output **exactly (max |diff| = 0.0)** on both spins.
  (The alternative of feeding `eval_xc`'s outputs into
  `pyscf.dft.xc_deriv.transform_xc` directly was prototyped too and produced
  garbage — the `out` row order follows the xcfun convention and `transform_xc`
  reads past the initialised rows for our input — so the explicit chain rule is
  the design of record.)
- **Row counts.** `eval_rho` for MGGA runs with `with_lapl=False`
  (`MGGA_DENSITY_LAPL = False`), so rho arrives with 5 rows (or `(2,5,N)`),
  while the graph's placeholders expect 6 rows; a zero laplacian row is
  inserted at index 4 (the graph discards index 4 anyway).
- **`tensorflow-hub<0.16` is unusable on the declared TF.** Verified from the
  wheels: 0.14.0 and 0.15.0 both do
  `from tensorflow_hub.estimator import ...` at `__init__.py:90`, which imports
  `tensorflow.compat.v1.estimator`, absent from TF 2.21. The honest floor for
  this repo is `tensorflow-hub>=0.16.0` (root `setup.py` reads
  `requirements.txt`, so that one line is the whole dependency change).
- **The restricted path's local-HF features are spin-consistent:**
  `compute_hfx_density.get_hf_density` splits a restricted `dm` into `dm/2` per
  spin internally, so the RKS and UKS paths evaluate the same functional at
  `ρ_a = ρ_b`; an RKS-vs-UKS energy comparison is therefore a sound end-to-end
  test of the `spin=0` layout.

## Reuse analysis

- **`NeuralNumInt.eval_xc` (existing)** — the shim does not re-implement any
  DM21 evaluation: `eval_xc_eff` routes to it, reusing its derivative
  computation, its float64 casts, and its `_vmat_hf` (local-HF potential)
  side effects exactly as before the PySCF upgrade.
- **PySCF's own contract** — the layout table above is PySCF's
  `NumInt.eval_xc_eff` contract, not a new invention; it was confirmed against
  `pyscf.dft.numint.NumInt.eval_xc_eff` empirically (see Verification).
- **Test file conventions** — the new test reuses the module-level `mols` list,
  `dft.RKS`/`dft.UKS` + `NeuralNumInt` setup of the existing `test_dm21`, the
  `Hartree2kcalmol` conversion already imported at
  `tests/.../test_functional_implementations.py:24`, and the file's `atol=1`
  kcal/mol tolerance style. No new helpers.
- No `grad_dft/utils/` additions are needed; nothing else in the repo calls
  `eval_xc`/`eval_xc_eff` (verified by grep: only `neural_numint.py` itself and
  `grad_dft/interface/pyscf.py:1050`, which only monkey-patches `mf._numint`).

### S1: Port `neural_numint.py` to the TF-Hub 0.16 API (`hub.load`) and raise the declared hub floor [done]

**Code:** `grad_dft/external/density_functional_approximation_dm21/density_functional_approximation_dm21/neural_numint.py`
(`_build_graph`, `export_functional_and_derivatives`), `requirements.txt`
(`tensorflow-hub>=0.14.0` → `tensorflow-hub>=0.16.0`)
**Tests:** run the new test only via construction smoke test
(`python -c "from grad_dft.external import NeuralNumInt, Functional;
NeuralNumInt(Functional.DM21)"` must build the graph), then
`pytest -v tests/integration/molecules/test_functional_implementations.py::test_dm21`
— expected to fail at the **documented second break**
(`ValueError: cannot reshape array of size 240720 into shape (2,1,24072)`), not
at `AttributeError: hub.Module`; that error is the S2 input. The suite must
otherwise be unchanged (all other tests in the file still pass).
**Docs:** No README/example change needed (TF-Hub internals are invisible to
users). The `export_functional_and_derivatives` docstring is updated in-code to
state the new `NotImplementedError` behaviour and why; the class docstring's
usage example is untouched.

**Spec:**
- In `_build_graph`:
  - Replace `self._functional = hub.Module(spec=self._model_path)` with
    `self._functional = hub.load(self._model_path).signatures["default"]`.
  - Keep `tensor_dict = {f"tensor_dict${k}": v for k, v in features.items()}`
    and change `predictions = self._functional(tensor_dict, as_dict=True)` to
    `predictions = self._functional(**tensor_dict)`; `local_xc =
    predictions["grid_contribution"]` stays (verified output key).
  - Delete the `outputs` dict and the trailing
    `hub.add_signature(inputs=..., outputs=...)` call (API removed in 0.16; the
    only consumer was the spec-based export, which now raises — see below), and
    drop the "no-op if called outside hub.create_module_spec" comment.
  - Everything else (placeholders, feature construction, `tf.gradients`,
    session) is untouched — verified to work through the port.
- Replace the body of `export_functional_and_derivatives` with a
  `raise NotImplementedError` whose message says: TF-Hub >= 0.16 removed the
  TF1 `Module`/`create_module_spec` API this export was built on; exporting
  would require a TF2 `tf.saved_model.save` reimplementation; the method is
  only used by the vendored `export_saved_model.py` / `neural_numint_test.py`,
  which are not part of the test suite. Keep the original docstring and append
  the Raises section.
- `requirements.txt`: raise the floor to `tensorflow-hub>=0.16.0` (0.14/0.15
  import `tensorflow_hub.estimator` → `tensorflow.compat.v1.estimator`, which no
  longer exists in TF 2.21; verified from the published wheels).
- Style: `black` is **not** applied to this vendored file — the baseline already
  fails `black --check`, and a file-wide reformat is out of scope (task #5
  precedent: edits stay diff-only to limit churn/upstream divergence). `pylint
  -rn -sn --rcfile=.pylintrc` must show no new messages vs the subtask parent.
- Commit: `(#6-S1): port neural_numint to the TF-Hub 0.16 hub.load API`.

- [x] S1: Implement (`hub.load` port, `outputs`/`hub.add_signature` removal,
      `NotImplementedError` export, hub floor `>=0.16.0`)
- [x] S1: Write tests (no new test in this subtask; smoke test builds the graph
      and `WrappedFunction`, and with `PYSCF_CONFIG_FILE=.github/workflows/pyscf_conf.py`
      the file gives 12 passed / 2 failed — `test_dm21` alone, now at the
      documented second break, the reshape `ValueError`, instead of
      `AttributeError: hub.Module`)
- [x] S1: Update documentation (no user-facing change needed — TF-Hub internals
      are invisible to users; the `Raises` section added to
      `export_functional_and_derivatives`' docstring is the explicit outcome)
- [x] S1: Pre-Commit Check (`black` deliberately not applied: the baseline
      already fails `black --check`, so the file stays as-is per the plan's
      style note; `pylint -rn -sn --rcfile=.pylintrc` vs subtask parent: no new
      messages, and the 4 baseline `E1101 tensorflow_hub no-member` errors are
      gone)
- [x] S1: Quality checks (no duplication: single evaluation path retained;
      nothing added to library modules outside the vendored shim; no tolerance
      or assertion touched)
- [x] S1: Commit

### S2: Add the `eval_xc_eff` override that routes PySCF's derivative contract to `eval_xc` [done]

**Code:** new method `NeuralNumInt.eval_xc_eff` in
`grad_dft/external/density_functional_approximation_dm21/density_functional_approximation_dm21/neural_numint.py`
(placed directly after `eval_xc`); one class-docstring sentence updated
("The actual evaluation ... is performed in NeuralNumInt.eval_xc" → mention
that PySCF reaches it through the `eval_xc_eff` override)
**Tests:** `pytest -v tests/integration/molecules/test_functional_implementations.py`
— with the shim, `test_dm21`'s PySCF leg (`mf.kernel()` on both parametrized
molecules) completes and the other 9 tests stay green; the test as a whole
**cannot pass until S3** because its second leg,
`generate_DM21_weights`, hits an independent break (TF1 `tf.saved_model.load`
on TF 2.21 — see S3). The `atol=1` tolerance is untouched. Also run
`pytest -v tests/integration/molecules/test_predict_B88.py
tests/integration/molecules/test_non_xc_energy.py tests/integration/molecules/test_Harris.py`
to confirm no collateral damage.
**Docs:** numpydoc docstring on `eval_xc_eff` documenting the argument
semantics (delegates to `eval_xc`), the returned layout table for `spin=0`/
`spin=1`, and the Raises conditions; no README/example change (public usage of
`NeuralNumInt` is unchanged).

**Spec:**
- Signature mirrors PySCF's base method exactly:
  `def eval_xc_eff(self, xc_code, rho, deriv=1, omega=None, xctype=None, verbose=None, spin=None)`.
- Behaviour:
  1. Mirror the base class preamble: `omega = self.omega if omega is None else
     omega` (`NumInt.omega` is `None`, so a plain call passes `omega=None`
     through to `eval_xc`, which accepts `None` and raises a clear
     `NotImplementedError` for an explicit RSH value — unchanged semantics);
     normalise `rho` with `np.asarray(rho, order="C")`; infer `spin` from
     `rho.shape[0] == 2` when it is not given (same rule as the base class).
  2. `deriv == 0` → still evaluate (the `eval_xc` call carries the `_vmat_hf`
     side effect) but return `[exc, None, None, None]`; `deriv > 1` → raise
     `NotImplementedError` ("DM21 provides no second derivatives"; the base
     class would die inside `transform_xc`).
  3. Re-insert the laplacian row: if `rho.shape[-2] == 5`, insert a zero row at
     index 4 (graph placeholders are `(6, N)`; index 4 is discarded by the
     graph). If 6 rows arrive, pass through.
  4. `exc, (vrho, vsigma, vlapl, vtau), _, _ = self.eval_xc(xc_code, rho6,
     spin=spin, deriv=deriv, omega=omega, verbose=verbose)` — full reuse of the
     existing evaluation, including `_vmat_hf` accumulation.
  5. Build the effective derivatives with the validated chain rule
     (outputs of `eval_xc` are `(N, ncomp)` — transpose to `(ncomp, N)` for the
     spin-polarised case):
     - `spin == 0`: `vxc = np.concatenate([vrho[None, :],
       2·vsigma[None, :]·rho[1:4], vtau[None, :]])` → shape `(5, N)`, rows
       `[dE/dρ, dE/d∇ρ (3), dE/dτ]` — **no laplacian row** (this line
       previously claimed a `zeros` row at index 2; re-validated against
       `NumInt.eval_xc_eff` on `mgga_x_tpss`, max |diff| = 0.0, so the layout
       table in Verification is the correct one).
     - `spin == 1`: shape `(2, 5, N)` with
       `vxc[:,0] = vrho.T`, `vxc[0,1:4] = 2·vsigma[:,0]·∇ρ_a + vsigma[:,1]·∇ρ_b`,
       `vxc[1,1:4] = 2·vsigma[:,2]·∇ρ_b + vsigma[:,1]·∇ρ_a`,
       `vxc[:,4] = vtau.T`, where `∇ρ_a = rho[0,1:4]`, `∇ρ_b = rho[1,1:4]`.
  6. Return `(exc, vxc, None, None)` (float64 — `eval_xc` already casts).
- The docstring states the layout contract and notes the chain rule was
  verified against `pyscf.dft.numint.NumInt.eval_xc_eff`.
- Comment where the chain rule lives, pointing at the PySCF call sites
  (`nr_rks` MGGA branch: `wv[0]/wv[4]` halving, `_scale_ao_sparse(ao[:4],
  wv[:4])`, `_tau_dot_sparse(..., wv[4])`; `nr_uks` the same per spin row) so a
  future reader can re-derive it.
- Commit: `(#6-S2): add an eval_xc_eff shim routing PySCF 2.13 back to eval_xc`.

- [x] S2: Implement (`eval_xc_eff` override + class-docstring sentence)
- [x] S2: Write tests (chain rule vs `NumInt.eval_xc_eff` on `mgga_x_tpss`:
      max |diff| = 0.0 for both spins; `test_dm21`'s PySCF leg — `mf.kernel()`
      on both molecules — completes; collateral files green: 45 passed across
      `test_predict_B88.py`, `test_non_xc_energy.py`, `test_Harris.py`; the
      full `test_dm21` is deferred to S3, see Tests)
- [x] S2: Update documentation (numpydoc docstring on `eval_xc_eff` with the
      layout contract and raises; class docstring updated; no user-facing change)
- [x] S2: Pre-Commit Check (`black` not applied to the vendored file, same
      precedent as S1; `pylint -rn -sn --rcfile=.pylintrc` vs S1's parent: no
      new messages)
- [x] S2: Quality checks (no duplication: single evaluation path reused via
      `eval_xc`; no tolerance or assertion touched)
- [x] S2: Commit

### S3: Read the vendored TF1 DM21 checkpoint directly in `generate_DM21_weights` [done]

**Discovered while running S2's tests** — not part of the original issue
analysis: with the `eval_xc_eff` shim in place, `mf.kernel()` completes, and
`test_dm21` moves to its second leg, `DM21().generate_DM21_weights()`
(`grad_dft/functional.py`), which loads the same vendored checkpoint with
`tf.saved_model.load(folder)`. On TF 2.21 the V1 SavedModel restore machinery
removed in TF 2.x fails inside `load_v1_in_v2.restore_variables` with
`TypeError: Binding inputs to tf.function failed due to 'too many positional
arguments'` (the pruned restore function has an empty signature but is called
with the variables path). The dependency floor already forces TF >= 2.13, so
`tf.saved_model.load` on a V1 SavedModel is not a supported path.

**Code:** `grad_dft/functional.py`, `generate_DM21_weights`
**Tests:** `test_dm21` (both parametrized molecules) must now pass end-to-end
unchanged (`atol=1`), which also validates the entire PySCF+shim leg against
the independently implemented JAX port of DM21; the whole
`test_functional_implementations.py` file must stay green.
**Docs:** No user-facing change needed — `generate_DM21_weights`'s interface
(`folder` default, return value) is unchanged.

**Spec:**
- Replace `variables = tf.saved_model.load(folder).variables` with a direct
  checkpoint read: `tf.train.load_checkpoint(os.path.join(folder,
  "variables", "variables"))` plus `tf.train.list_variables(...)`, appending
  the `":0"` op suffix to each name so the existing name patterns in
  `vars_to_params` (`"/w:"`, `"/b:"`, `"gamma:"`, `"beta:"`) still match.
- `vars_to_params` iterates `(name, value)` pairs; `tf_tensor_to_jax` becomes
  a plain `jnp.asarray(value)` (the checkpoint reader returns numpy arrays).
- Comment in code: the vendored checkpoint is a TF1 SavedModel that
  `tf.saved_model.load` can no longer import on TF >= 2.13.
- Commit: `(#6-S3): read the vendored TF1 DM21 checkpoint directly`.

- [x] S3: Implement (direct checkpoint read; `vars_to_params` on `(name,
      value)` pairs; interface unchanged)
- [x] S3: Write tests (`test_dm21` end-to-end: 2 passed; whole file 14 passed,
      `atol=1` untouched)
- [x] S3: Update documentation (no user-facing change needed, justified above)
- [x] S3: Pre-Commit Check (pylint vs S2's parent: symbol multiset identical;
      black: no new debt — baseline `black --check` already fails, task #7)
- [x] S3: Quality checks (no duplication: reader stays local to the method;
      no tolerance or assertion touched)
- [x] S3: Commit

### S4: Cover the restricted (`spin=0`) path with an RKS-vs-UKS consistency test [done]

**Code:** new test `test_dm21_rks` in
`tests/integration/molecules/test_functional_implementations.py` (after
`test_dm21`), reusing the module-level `mols[0]` closed-shell molecule and the
existing imports
**Tests:** the new test itself: build `dft.RKS(mol)` and `dft.UKS(mol)` on the
closed-shell HF molecule, give both `mf._numint = NeuralNumInt(Functional.DM21)`
(identical default grids, mirroring `test_dm21`'s setup exactly), run both
`mf.kernel()` calls, and assert `(e_rks - e_uks) * Hartree2kcalmol` is
`allclose(..., 0, atol=1)` and that neither energy is NaN. The whole file must
stay green.
**Docs:** No user-facing change needed (test-only; nothing in README/examples
documents individual test names).

**Spec:**
- Rationale (recorded in the test as a comment): DM21 maps restricted inputs to
  `ρ_a = ρ_b = ρ/2` and `get_hf_density` splits a restricted `dm` into `dm/2`
  per spin, so on a closed-shell molecule the restricted and unrestricted
  self-consistent solutions must agree; UKS is already validated end-to-end by
  `test_dm21`. A wrong `spin=0` derivative layout therefore shows up as an
  energy mismatch (or a diverged/NaN SCF), not as a silent pass.
- Only the closed-shell molecule (`mols[0]`, HF) is used — RKS on the open-shell
  Li atom (`mols[1]`) is not a valid restricted calculation.
- Fallback (decided in advance, per the issue): if this test demonstrates that
  the `spin=0` path cannot be made to agree (i.e. the failure is not the layout
  shim's to fix), replace the RKS path with an explicit
  `raise NotImplementedError("The restricted (spin=0) path ...")` in
  `eval_xc_eff` and change the test to assert that raise. The test-coverage
  route is tried first because the restricted route is the documented usage in
  the class docstring (`mf = dft.RKS(...)`).
- Commit (last subtask): `(#6-S4): cover the DM21 restricted path with an
  RKS-vs-UKS consistency test` plus a standalone `Closes #6` line.

- [x] S4: Implement (new `test_dm21_rks` after `test_dm21`, reusing `mols[0]`
      and the existing imports; local `mol_rks` avoids shadowing the module-level
      `mol`)
- [x] S4: Write tests (`test_dm21_rks` passes standalone — 1 passed in 144 s;
      whole file 15 passed in 303 s — the RKS and UKS energies agree within
      `atol=1`, neither NaN)
- [x] S4: Update documentation (no user-facing change needed — test-only; the
      rationale is recorded in the test's docstring)
- [x] S4: Pre-Commit Check (pylint vs S3's parent: symbol multiset identical;
      black: no new debt — baseline `black --check` already fails, task #7)
- [x] S4: Quality checks (no duplication: reuses `mols[0]`, `NeuralNumInt`,
      `Functional.DM21`; no tolerance or assertion touched)
- [x] S4: Commit (Closes #6)

## Acceptance criteria mapping

| Criterion (issue #6) | Where |
|---|---|
| `test_dm21` passes, no tolerance change | S2 (shim) + S3 (checkpoint reader) — `atol=1` untouched |
| `spin=0` covered by a new test or explicitly rejected | S4 (test first, `NotImplementedError` fallback pre-decided) |
| `hub.load` port and PySCF shim in separate commits | S1 / S2 commits |
| `export_functional_and_derivatives()` explicit outcome | S1 (`NotImplementedError` + docstring) |

After S4: whole-repo code review (`code-review` skill), then the quality gate
(`quality-gates` skill: CI-file pytest invocations + `black` + `pylint`), then
integration into `agent_settings` after user confirmation.
