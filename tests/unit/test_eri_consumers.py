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

"""Tests for remaining consumers and metadata (S5 of #15).

Covers:

- `eri_allclose`: None/None, None/mixed, same-format packed equal/unequal,
  mixed representations (packed-vs-packed of different formats, packed-vs-
  dense) compared via dense materialization, dense pair.
- `constraints.xc1`: rep_tensor comparison via `eri_allclose`; the full
  constraint loss is identical for packed (mixed s8/s4) and dense storage.
- `Molecule`: `rep_tensor=None` default construction and non-ERI methods
  (`density`, `grad_density`) work.
"""

from jax import config

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto

from grad_dft import LSDA, Grid, Molecule
from grad_dft.constraints import xc1
from grad_dft.utils.eri import PackedERI, eri_allclose, packed_to_dense as to_dense

config.update("jax_enable_x64", True)

MOL = gto.M(
    atom=[["H", (0, 0, 0)], ["H", (0.74, 0, 0)]],
    basis="cc-pvdz",
    verbose=0,
)
N = MOL.nao_nr()
REP_DENSE = jnp.asarray(MOL.intor("int2e", aosym="s1"))


def packed_eri(fmt: str) -> PackedERI:
    """Build PackedERI storage of the test molecule's integrals."""
    return PackedERI(np.asarray(MOL.intor("int2e", aosym=fmt)), fmt, N)


def test_eri_allclose_none_pairs():
    """None equals only None; mixed None/non-None is False."""
    assert eri_allclose(None, None) is True
    assert eri_allclose(REP_DENSE, None) is False
    assert eri_allclose(None, REP_DENSE) is False


def test_eri_allclose_same_format_equal_unequal():
    """Same-format packed comparisons run on the packed storage directly."""
    p4 = packed_eri("s4")
    p8 = packed_eri("s8")
    # equal
    assert eri_allclose(p4, packed_eri("s4")) is True
    assert eri_allclose(p8, packed_eri("s8")) is True
    # unequal
    p4_bad = PackedERI(
        jnp.array(p4.packed).at[0, 0].set(jnp.array(p4.packed)[0, 0] + 1.0), "s4", N
    )
    assert eri_allclose(p4, p4_bad) is False


def test_eri_allclose_mixed_formats_via_todense():
    """Mixed representations (packed-vs-packed, packed-vs-dense) compare equal."""
    np.testing.assert_allclose(
        to_dense(packed_eri("s4")), REP_DENSE, atol=1e-14, rtol=0
    )
    assert eri_allclose(packed_eri("s4"), packed_eri("s8")) is True
    assert eri_allclose(packed_eri("s4"), REP_DENSE) is True
    assert eri_allclose(REP_DENSE, packed_eri("s8")) is True


def test_eri_allclose_dense_pair():
    """Dense arrays compare elementwise within tolerance."""
    rep2 = jnp.array(REP_DENSE)
    assert eri_allclose(rep2, REP_DENSE) is True
    rep_bad = rep2.at[0, 0, 0, 0].set(rep2[0, 0, 0, 0] + 2.0)
    assert eri_allclose(rep_bad, REP_DENSE) is False


def test_eri_allclose_mismatched_packed_formats_compare_via_dense():
    """Different packed formats holding equal values compare equal (via dense)."""
    s4 = packed_eri("s4")
    s8 = packed_eri("s8")
    assert eri_allclose(s4, s8) is True


def make_minimal_molecule(rep_tensor, rdm_seed=17):
    """Synthetic Molecule with fields as produced by molecule_from_pyscf.

    The grid/ao rng is fixed so that molecules built with different
    ``rdm_seed`` differ only in their density matrix (xc1 compares ao).
    """
    grid_size = 6
    rng = np.random.default_rng(17)  # fixed: xc1 compares ao/grad_ao across molecules
    grid = Grid(
        coords=jnp.zeros((grid_size, 3)), weights=jnp.ones((grid_size,)) / grid_size
    )
    rdm_rng = np.random.default_rng(rdm_seed)
    rdm1 = np.eye(N) + 0.2 * rdm_rng.standard_normal((N, N))
    rdm1 = jnp.asarray(0.5 * (rdm1 + rdm1.T))
    spin_rdm1 = jnp.stack([0.5 * rdm1, 0.5 * rdm1])
    return Molecule(
        grid=grid,
        atom_index=jnp.array([0, 0]),
        nuclear_pos=jnp.zeros((2, 3)),
        ao=jnp.asarray(rng.standard_normal((grid_size, N))),
        grad_ao=jnp.asarray(rng.standard_normal((grid_size, N, 3))),
        grad_n_ao={
            0: jnp.zeros((grid_size, N)),
            1: jnp.zeros((grid_size, N)),
            2: jnp.zeros((grid_size, N)),
        },
        rdm1=spin_rdm1,
        nuclear_repulsion=jnp.asarray(1.0),
        h1e=jnp.eye(N),
        vj=jnp.zeros((2, N, N)),
        mo_coeff=jnp.zeros((2, N, N)),
        mo_occ=jnp.zeros((2, N)),
        mo_energy=jnp.zeros((2, N)),
        # fields xc1 compares unguarded (as produced by molecule_from_pyscf):
        s1e=jnp.eye(N),
        omegas=jnp.zeros((1,)),
        basis=jnp.array([ord("H"), ord("2")]),
        rep_tensor=rep_tensor,
        name=jnp.array([ord("H"), ord("2")]),
    )


def test_rep_tensor_none_methods_work():
    """Non-ERI methods (density, grad_density) work with rep_tensor=None."""
    mol = make_minimal_molecule(rep_tensor=None)
    # non-ERI methods should work
    _ = mol.density()
    _ = mol.grad_density()
    # jit also works
    _ = jax.jit(lambda m: (m.density(), m.grad_density()))(mol)


def test_xc1_with_packed_molecules():
    """xc1's constraint loss is identical for packed and dense ERI storage.

    xc1 evaluates the full energy predictor (functional + coulomb path, i.e.
    the S2 hot dispatch) on both molecules, so this exercises the consumer
    end-to-end: mixed s8/s4 storage must give the same loss as all-dense.
    """
    gamma = 0.3
    packed_loss = xc1(
        LSDA,
        {},
        make_minimal_molecule(packed_eri("s8")),
        make_minimal_molecule(packed_eri("s4"), rdm_seed=19),
        gamma,
        mol=MOL,
    )
    dense_loss = xc1(
        LSDA,
        {},
        make_minimal_molecule(REP_DENSE),
        make_minimal_molecule(REP_DENSE, rdm_seed=19),
        gamma,
        mol=MOL,
    )
    assert jnp.isfinite(packed_loss)
    assert float(packed_loss) > 0  # non-trivial: the two molecules differ
    np.testing.assert_allclose(packed_loss, dense_loss, atol=1e-10, rtol=1e-10)
