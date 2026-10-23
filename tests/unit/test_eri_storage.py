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

"""Tests for the symmetry-compressed ERI storage container (grad_dft.utils.eri).

Covers, at the container level:

(A) the `aosym` options actually accepted by the installed PySCF for
    `int2e` (shapes, dtypes, aliases, rejected options);
(B) ERI symmetry relations of real molecular integrals;
(C) dense expansion against `pyscf.ao2mo.restore` as the trusted reference;
(F) error handling for invalid formats, dtypes and shapes;
(D, partial) PyTree registration and jit round-trip of the container.
"""

from functools import lru_cache

import numpy as np
import pytest

from jax import config
import jax
import jax.numpy as jnp
from pyscf import gto
from pyscf.ao2mo import restore

from grad_dft.utils.eri import (
    ERI_STORAGE_FORMATS,
    PackedERI,
    pair_indices,
    packed8_to_packed4,
    validate_eri_storage,
)

# Enable x64 after imports (grad_dft creates no arrays at import time) but
# before any array is created, so PySCF's float64 ERIs are not downcast.
config.update("jax_enable_x64", True)

# A symmetric diatomic (H2) and an asymmetric polyatomic (H2O). The latter
# avoids accidental all-zero ERI blocks that make pair-matching ambiguous.
MOLECULES = {
    "H2": [["H", (0, 0, 0)], ["H", (0.74, 0, 0)]],
    "H2O": [["O", (0, 0, 0)], ["H", (0, 0, 0.97)], ["H", (0.76, 0, -0.5)]],
}


@lru_cache(maxsize=None)
def build_mol(name: str) -> gto.Mole:
    """Small cc-pvdz test molecule (cached: integrals are reused across tests)."""
    return gto.M(atom=MOLECULES[name], basis="cc-pvdz", verbose=0)


def n_pairs(n_orbitals: int) -> int:
    """Number of unique index pairs M = n (n + 1) / 2."""
    return n_orbitals * (n_orbitals + 1) // 2


@pytest.mark.parametrize("name", list(MOLECULES))
def test_pyscf_aosym_shapes_and_dtypes(name):
    """Document the layout PySCF returns for each aosym option (task 8A)."""
    mol = build_mol(name)
    n = mol.nao_nr()
    m = n_pairs(n)
    t = n_pairs(m)
    dense = mol.intor("int2e", aosym="s1")

    assert dense.shape == (n, n, n, n)
    assert mol.intor("int2e", aosym="s2ij").shape == (m, n, n)
    assert mol.intor("int2e", aosym="s2kl").shape == (n, n, m)
    assert mol.intor("int2e", aosym="s4").shape == (m, m)
    assert mol.intor("int2e", aosym="s8").shape == (t,)

    for fmt, shape in [("s1", (n, n, n, n)), ("s4", (m, m)), ("s8", (t,))]:
        eri = mol.intor("int2e", aosym=fmt)
        assert eri.shape == shape
        assert eri.dtype == np.float64
        assert eri.nbytes == eri.size * 8


@pytest.mark.parametrize("alias,canonical", [("4", "s4"), ("8", "s8")])
def test_pyscf_numeric_aosym_aliases(alias, canonical):
    """PySCF accepts '4'/'8' as aliases of 's4'/'s8'; GradDFT's config takes
    only the canonical names (see test_validate_eri_storage)."""
    mol = build_mol("H2")
    np.testing.assert_array_equal(
        mol.intor("int2e", aosym=alias), mol.intor("int2e", aosym=canonical)
    )


def test_pyscf_s2_unsupported_for_int2e():
    """aosym='s2' is not supported for int2e in the installed PySCF 2.13.1
    (AttributeError from libcgto.so: undefined symbol GTOnr2e_fill_s2).
    If a future PySCF gains support, this test should fail so the storage
    audit is re-run."""
    mol = build_mol("H2")
    with pytest.raises(Exception):  # pylint: disable=broad-except
        mol.intor("int2e", aosym="s2")


@pytest.mark.parametrize("fmt", ["s4", "s8"])
@pytest.mark.parametrize("name", list(MOLECULES))
def test_to_dense_matches_pyscf_restore(fmt, name):
    """Our expansion must reproduce pyscf.ao2mo.restore exactly.

    Bitwise equality is expected: both perform pure index copies of the same
    stored values, so any difference would indicate a packing-convention bug.
    """
    mol = build_mol(name)
    n = mol.nao_nr()
    packed = mol.intor("int2e", aosym=fmt)
    eri = PackedERI(packed, fmt, n)

    trusted = restore(1, packed, n)
    np.testing.assert_array_equal(np.asarray(eri.to_dense()), trusted)
    # And the reconstructed tensor equals the direct dense integral up to the
    # machine-precision difference between independently computed packings.
    np.testing.assert_allclose(
        np.asarray(eri.to_dense()), mol.intor("int2e", aosym="s1"), atol=1e-13, rtol=0
    )


@pytest.mark.parametrize("name", list(MOLECULES))
def test_packed8_to_packed4_matches_intor_s4(name):
    """s8 storage expands to the same (M, M) matrix as the direct s4 packing.

    Tolerance 1e-14 rather than bitwise: PySCF evaluates the s4 and s8
    packings independently, and they differ at machine precision (observed
    max 2.2e-16 on PySCF 2.13.1). The bitwise reference test is
    `test_to_dense_matches_pyscf_restore`, which compares against a pure
    index-copy of the same stored values.
    """
    mol = build_mol(name)
    n = mol.nao_nr()
    m = n_pairs(n)
    e8 = mol.intor("int2e", aosym="s8")
    e4 = mol.intor("int2e", aosym="s4")
    np.testing.assert_allclose(
        np.asarray(packed8_to_packed4(jnp.asarray(e8), m)), e4, atol=1e-14, rtol=0
    )


def test_pair_indices_convention():
    """pair(a, b) = a (a + 1) / 2 + b for a >= b, symmetric, bijective onto [0, M)."""
    n = 6
    m = n_pairs(n)
    index = np.asarray(pair_indices(n))
    assert index.shape == (n, n)
    for a in range(n):
        for b in range(n):
            expected = max(a, b) * (max(a, b) + 1) // 2 + min(a, b)
            assert index[a, b] == expected
    np.testing.assert_array_equal(index, index.T)
    assert set(index.ravel().tolist()) == set(range(m))


def test_symmetry_relations_real_molecular_eri():
    """Real molecular ERIs satisfy (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij).

    Validity scope: real, isolated molecules only — complex/periodic ERIs
    are not covered by design (they keep dense storage).
    """
    dense = build_mol("H2O").intor("int2e", aosym="s1")
    np.testing.assert_allclose(dense, dense.transpose(1, 0, 2, 3), atol=1e-14, rtol=0)
    np.testing.assert_allclose(dense, dense.transpose(0, 1, 3, 2), atol=1e-14, rtol=0)
    np.testing.assert_allclose(dense, dense.transpose(2, 3, 0, 1), atol=1e-14, rtol=0)


@pytest.mark.parametrize("fmt", ERI_STORAGE_FORMATS)
def test_validate_eri_storage_accepts_supported(fmt):
    """The three supported configuration values are accepted unchanged."""
    assert validate_eri_storage(fmt) == fmt


@pytest.mark.parametrize("value", ["s2", "s2ij", "s2kl", "4", "8", "S8", ""])
def test_validate_eri_storage_rejects_unsupported(value):
    """Unsupported values fail with an error listing the valid formats."""
    with pytest.raises(ValueError, match="Supported formats are"):
        validate_eri_storage(value)


def test_packed_eri_rejects_unknown_format():
    """PackedERI only holds the packed formats; 'dense' stays a plain array."""
    with pytest.raises(ValueError, match="storage_format"):
        PackedERI(np.zeros((6, 6)), "s2", 3)


def test_packed_eri_rejects_complex_dtype():
    """Complex ERIs (e.g. multi-k-point) must keep eri_storage='dense'."""
    with pytest.raises(ValueError, match="complex"):
        PackedERI(np.zeros((6, 6), dtype=np.complex128), "s4", 3)


def test_packed_eri_rejects_non_floating_dtype():
    """Only real floating-point dtypes are valid ERI storage."""
    with pytest.raises(ValueError, match="floating"):
        PackedERI(np.zeros((6, 6), dtype=np.int32), "s4", 3)


def test_packed_eri_rejects_inconsistent_shape():
    """Physical storage must match format and n_orbitals.

    n = 3 -> M = 6: s4 must be (6, 6), s8 must be (21,).
    """
    with pytest.raises(ValueError, match="requires packed shape"):
        PackedERI(np.zeros((21, 21)), "s4", 3)
    with pytest.raises(ValueError, match="requires packed shape"):
        PackedERI(np.zeros((21,)), "s8", 4)


@pytest.mark.parametrize("n", [0, -1, 2.5])
def test_packed_eri_rejects_invalid_n_orbitals(n):
    """n_orbitals must be a positive integer."""
    with pytest.raises(ValueError, match="n_orbitals"):
        PackedERI(np.zeros((6, 6)), "s4", n)


def test_packed_eri_logical_vs_physical_shapes():
    """`.shape` is the logical (n, n, n, n); storage shape lives on `.packed`."""
    n = 4
    m = n_pairs(n)
    t = n_pairs(m)
    e4 = PackedERI(np.zeros((m, m)), "s4", n)
    e8 = PackedERI(np.zeros((t,)), "s8", n)
    assert e4.shape == (n, n, n, n)  # logical
    assert e4.packed.shape == (m, m)  # physical
    assert e8.shape == (n, n, n, n)
    assert e8.packed.shape == (t,)
    assert e4.dtype == np.float64


def test_packed_eri_is_pytree_with_static_metadata():
    """The packed array is the single leaf; format and n_orbitals are aux data."""
    n = 3
    m = n_pairs(n)
    eri = PackedERI(jnp.zeros((m, m)), "s4", n)
    leaves, treedef = jax.tree_util.tree_flatten(eri)
    assert len(leaves) == 1
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, PackedERI)
    assert rebuilt.storage_format == "s4"
    assert rebuilt.n_orbitals == n


def test_packed_eri_jit_roundtrip():
    """jit round-trips values exactly and accepts mixed formats (one compile
    per distinct static metadata)."""
    mol = build_mol("H2")
    n = mol.nao_nr()
    identity = jax.jit(lambda x: x)
    for fmt in ("s4", "s8"):
        eri = PackedERI(jnp.asarray(mol.intor("int2e", aosym=fmt)), fmt, n)
        out = identity(eri)
        assert isinstance(out, PackedERI)
        assert out.storage_format == fmt
        assert out.n_orbitals == n
        np.testing.assert_array_equal(np.asarray(out.packed), np.asarray(eri.packed))
