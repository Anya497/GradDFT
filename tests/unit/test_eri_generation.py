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

"""Tests for the PySCF-side generation of compressed ERIs (S3 of #15).

Covers:

(A) `calc_eri_with_pyscf` isolated molecules: packed formats match the PySCF
    reference packings bitwise (`intor(aosym=...)`) / to machine precision
    for the density-fitted branch (fresh DF references);
(A/B) the dense path is behaviorally unchanged (isolated and periodic);
(F) error handling: invalid formats fail before any integral work, periodic
    systems reject compression in both `calc_eri_with_pyscf` and
    `molecule_from_pyscf` (fail-fast, no SCF required);
(D) `to_device_arrays` preserves PackedERI metadata and converts its leaf;
(D) end-to-end `molecule_from_pyscf(eri_storage=...)` pass-through.
"""

from jax import config

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import df, gto, scf
from pyscf.ao2mo import restore
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf

from grad_dft.interface.pyscf import (
    calc_eri_with_pyscf,
    molecule_from_pyscf,
    process_mol,
    to_device_arrays,
)
from grad_dft.utils.eri import PackedERI

config.update("jax_enable_x64", True)

PACKED_FORMATS = ("s4", "s8")

MOL = gto.M(
    atom=[["H", (0, 0, 0)], ["H", (0.74, 0, 0)]],
    basis="cc-pvdz",
    verbose=0,
)
N = MOL.nao_nr()

CELL = pbc_gto.Cell()
CELL.atom = "H 0 0 0; H 0.74 0 0"
CELL.a = np.eye(3) * 4.0
CELL.basis = "gth-szv"
CELL.verbose = 0
CELL.build()


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_calc_eri_isolated_matches_intor(fmt):
    """Isolated no-DF storage equals PySCF's own intor(aosym=fmt) bitwise."""
    mf = scf.RHF(MOL)
    eri = calc_eri_with_pyscf(mf, eri_storage=fmt)
    assert isinstance(eri, PackedERI)
    assert eri.storage_format == fmt
    assert eri.n_orbitals == N
    assert eri.packed.dtype == np.float64
    np.testing.assert_array_equal(np.asarray(eri.packed), MOL.intor("int2e", aosym=fmt))


def test_calc_eri_isolated_dense_unchanged():
    """The default dense branch still returns the uncompressed intor output."""
    mf = scf.RHF(MOL)
    eri = calc_eri_with_pyscf(mf)
    assert isinstance(eri, np.ndarray)
    np.testing.assert_array_equal(eri, MOL.intor("int2e"))


@pytest.mark.parametrize("fmt", ("dense",) + PACKED_FORMATS)
def test_calc_eri_density_fitted(fmt):
    """DF branch: values come from DF.get_eri() (always s8 storage).

    Packed outputs are never compared against exact integrals — density
    fitting approximates those by construction.
    """
    mf = scf.RHF(MOL).density_fit()
    n = MOL.nao_nr()
    eri_ref = df.DF(MOL).get_eri()  # fresh reference, same construction as the function
    eri = calc_eri_with_pyscf(mf, eri_storage=fmt)
    if fmt == "dense":
        assert isinstance(eri, np.ndarray)
        np.testing.assert_allclose(eri, restore(1, eri_ref, n), atol=1e-14, rtol=0)
    else:
        assert isinstance(eri, PackedERI)
        assert eri.storage_format == fmt
        expected = restore(4, eri_ref, n) if fmt == "s4" else eri_ref
        np.testing.assert_allclose(np.asarray(eri.packed), expected, atol=1e-14, rtol=0)


def test_calc_eri_density_fitted_formats_consistent():
    """The two packed DF formats reconstruct the same dense tensor."""
    s4 = calc_eri_with_pyscf(scf.RHF(MOL).density_fit(), eri_storage="s4")
    s8 = calc_eri_with_pyscf(scf.RHF(MOL).density_fit(), eri_storage="s8")
    np.testing.assert_allclose(s4.to_dense(), s8.to_dense(), atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        s4.to_dense(), restore(1, df.DF(MOL).get_eri(), N), atol=1e-14, rtol=0
    )


@pytest.mark.parametrize("bad", ["s2", "s2ij", "4", "8", "S8", ""])
def test_invalid_format_fails_before_integral_work(bad):
    """Invalid formats raise on mf=None: validation precedes any mf access."""
    with pytest.raises(ValueError, match="Supported formats are"):
        calc_eri_with_pyscf(None, eri_storage=bad)
    with pytest.raises(ValueError, match="Supported formats are"):
        molecule_from_pyscf(None, eri_storage=bad)


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_periodic_rejects_compression(fmt):
    """Periodic systems reject compressed storage in both entry points."""
    kmf = pbc_scf.RHF(CELL)
    with pytest.raises(ValueError, match="Periodic"):
        calc_eri_with_pyscf(kmf, eri_storage=fmt)
    with pytest.raises(ValueError, match="eri_storage"):
        molecule_from_pyscf(kmf, eri_storage=fmt)


def test_periodic_dense_unchanged():
    """The dense periodic (gamma-point) path still returns a dense array."""
    kmf = pbc_scf.RHF(CELL)
    eri = calc_eri_with_pyscf(kmf)
    assert isinstance(eri, np.ndarray)
    assert eri.shape == (CELL.nao_nr(),) * 4


def test_to_device_arrays_preserves_packed():
    """Packed metadata survives device conversion; the leaf becomes jnp."""
    eri_np = MOL.intor("int2e", aosym="s8")
    out = to_device_arrays(PackedERI(eri_np, "s8", N))[0]
    assert isinstance(out, PackedERI)
    assert out.storage_format == "s8"
    assert out.n_orbitals == N
    assert isinstance(out.packed, jnp.ndarray)
    assert out.packed.dtype == jnp.float64
    np.testing.assert_array_equal(np.asarray(out.packed), eri_np)


def test_molecule_from_pyscf_pass_through():
    """eri_storage flows through _package_outputs into Molecule.rep_tensor."""
    _, mf = process_mol(MOL, compute_energy=False, grid_level=1)
    mol_dense = molecule_from_pyscf(mf)
    mol_packed = molecule_from_pyscf(mf, eri_storage="s8")

    rep_dense = np.asarray(mol_dense.rep_tensor)
    rep_packed = mol_packed.rep_tensor
    assert isinstance(rep_packed, PackedERI)
    assert rep_packed.storage_format == "s8"
    assert rep_packed.packed.dtype == jnp.float64
    np.testing.assert_array_equal(rep_dense, MOL.intor("int2e"))
    np.testing.assert_allclose(
        np.asarray(rep_packed.to_dense()), MOL.intor("int2e"), atol=1e-14, rtol=0
    )
    # The end-to-end non-XC energy agrees between the two molecules.
    np.testing.assert_allclose(
        mol_packed.nonXC(), mol_dense.nonXC(), atol=1e-11, rtol=1e-10
    )
