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

"""Tests for the compressed Coulomb contractions (grad_dft.utils.eri).

Covers, at the contraction level:

(C) numerical equivalence of dense vs ``s4`` vs ``s8`` contractions for
    symmetric *and asymmetric* density matrices (an asymmetric D proves no
    density symmetry is assumed);
(D) jit/eager agreement, gradients wrt ``rdm1`` matching the dense path,
    and jaxpr inspection showing the packed path never materializes a
    ``(n, n, n, n)`` tensor;
(F) the ``s8`` row-chunked scan (forced small blocks) equals the default
    single-block contraction.

H2O/cc-pvdz is used rather than H2: its ERI has few exact-zero blocks, so
pair-indexing mistakes cannot be masked by zeros.
"""

from jax import config

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto

from grad_dft.molecule import Grid, Molecule, coulomb_energy, coulomb_potential
from grad_dft.molecule import nonXC
from grad_dft.utils.eri import PackedERI, packed_coulomb_potential

config.update("jax_enable_x64", True)

MOL = gto.M(
    atom=[["O", (0, 0, 0)], ["H", (0, 0, 0.97)], ["H", (0.76, 0, -0.5)]],
    basis="cc-pvdz",
    verbose=0,
)
N = MOL.nao_nr()
REP_DENSE = jnp.asarray(MOL.intor("int2e", aosym="s1"))
PACKED_FORMATS = ("s4", "s8")
RDM_KINDS = ("symmetric", "asymmetric")


def packed_eri(fmt: str) -> PackedERI:
    """Build PackedERI storage of the test molecule's integrals."""
    return PackedERI(jnp.asarray(MOL.intor("int2e", aosym=fmt)), fmt, N)


def make_rdm1(kind: str) -> jnp.ndarray:
    """Random (n, n) density matrix; 'symmetric' enforces D = D^T."""
    rng = np.random.default_rng(7 if kind == "symmetric" else 8)
    rdm1 = rng.standard_normal((N, N))
    if kind == "symmetric":
        rdm1 = 0.5 * (rdm1 + rdm1.T)
    return jnp.asarray(rdm1)


def make_molecule(rep_tensor, rdm1: jnp.ndarray) -> Molecule:
    """Minimal synthetic Molecule whose nonXC/Coulomb paths are exercised."""
    grid_size = 8
    rng = np.random.default_rng(11)
    grid = Grid(
        coords=jnp.zeros((grid_size, 3)),
        weights=jnp.full((grid_size,), 1.0 / grid_size),
    )
    spin_rdm1 = jnp.stack([0.6 * rdm1, 0.4 * rdm1])
    return Molecule(
        grid=grid,
        atom_index=jnp.arange(3),
        nuclear_pos=jnp.zeros((3, 3)),
        ao=jnp.asarray(rng.standard_normal((grid_size, N))),
        grad_ao=jnp.asarray(rng.standard_normal((grid_size, N, 3))),
        grad_n_ao=jnp.zeros((3, grid_size, N)),
        rdm1=spin_rdm1,
        nuclear_repulsion=jnp.asarray(0.89),
        h1e=jnp.asarray(rng.standard_normal((N, N))),
        vj=jnp.zeros((2, N, N)),
        mo_coeff=jnp.zeros((2, N, N)),
        mo_occ=jnp.zeros((2, N)),
        mo_energy=jnp.zeros((2, N)),
        rep_tensor=rep_tensor,
    )


def collect_jaxpr_shapes(jaxpr, shapes) -> None:
    """Recursively collect variable shapes, including inside nested jaxprs."""
    if hasattr(jaxpr, "eqns") and hasattr(jaxpr, "invars"):
        for var in list(jaxpr.invars) + list(jaxpr.outvars) + list(jaxpr.constvars):
            shapes.append(tuple(var.aval.shape))
        for eqn in jaxpr.eqns:
            for var in list(eqn.invars) + list(eqn.outvars):
                shapes.append(tuple(var.aval.shape))
            for param in eqn.params.values():
                if isinstance(param, (tuple, list)):
                    for item in param:
                        collect_jaxpr_shapes(item, shapes)
                else:
                    collect_jaxpr_shapes(param, shapes)
    elif isinstance(jaxpr, (tuple, list)):
        for item in jaxpr:
            collect_jaxpr_shapes(item, shapes)


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
@pytest.mark.parametrize("kind", RDM_KINDS)
def test_coulomb_potential_matches_dense(fmt, kind):
    """Packed J equals the dense einsum up to summation order (atol 1e-11)."""
    rdm1 = make_rdm1(kind)
    expected = coulomb_potential(rdm1, REP_DENSE)
    actual = coulomb_potential(rdm1, packed_eri(fmt))
    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
@pytest.mark.parametrize("kind", RDM_KINDS)
def test_coulomb_energy_matches_dense(fmt, kind):
    """Packed Coulomb energy equals the dense one for both density kinds."""
    rdm1 = make_rdm1(kind)
    expected = coulomb_energy(rdm1, REP_DENSE)
    actual = coulomb_energy(rdm1, packed_eri(fmt))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_non_xc_matches_dense(fmt):
    """nonXC is identical between dense and packed rep_tensor."""
    rdm1 = make_rdm1("asymmetric")
    h1e = jnp.asarray(np.random.default_rng(5).standard_normal((N, N)))
    expected = nonXC(rdm1, h1e, REP_DENSE, jnp.asarray(0.89))
    actual = nonXC(rdm1, h1e, packed_eri(fmt), jnp.asarray(0.89))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_molecule_methods_match_dense(fmt):
    """Molecule.get_coulomb_potential and Molecule.nonXC dispatch correctly."""
    rdm1 = make_rdm1("symmetric")
    dense_mol = make_molecule(REP_DENSE, rdm1)
    packed_mol = make_molecule(packed_eri(fmt), rdm1)
    np.testing.assert_allclose(
        packed_mol.get_coulomb_potential(),
        dense_mol.get_coulomb_potential(),
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        packed_mol.nonXC(), dense_mol.nonXC(), rtol=1e-10, atol=1e-11
    )


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_packed_contraction_jit_matches_eager(fmt):
    """The packed contraction is pure: jit output equals the eager output."""
    rdm1 = make_rdm1("asymmetric")
    eri = packed_eri(fmt)
    eager = packed_coulomb_potential(rdm1, eri)
    jitted = jax.jit(
        packed_coulomb_potential, static_argnames=("precision", "max_block_elements")
    )(rdm1, eri)
    np.testing.assert_allclose(jitted, eager, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_coulomb_energy_grad_matches_dense(fmt):
    """d Coulomb energy / d rdm1 matches the dense path through the packed ops."""
    rdm1 = make_rdm1("asymmetric")
    grad_fn = jax.grad(coulomb_energy)
    expected = grad_fn(rdm1, REP_DENSE)
    actual = grad_fn(rdm1, packed_eri(fmt))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_jaxpr_never_materializes_dense(fmt):
    """Packed trace contains no (n, n, n, n) aval and no N^4-sized buffer.

    The dense trace is checked in the opposite direction as a sanity check.
    (Benign rank-3 (n, n, 1) reshapes from JAX's gather lowering are fine —
    what must never appear is an N^4 allocation.)
    """
    rdm1 = make_rdm1("asymmetric")

    def trace(rep):
        return jax.make_jaxpr(coulomb_potential)(rdm1, rep)

    packed_shapes = []
    collect_jaxpr_shapes(trace(packed_eri(fmt)).jaxpr, packed_shapes)
    assert (N, N, N, N) not in packed_shapes
    assert max(np.prod(shape) for shape in packed_shapes) < N**4

    dense_shapes = []
    collect_jaxpr_shapes(trace(REP_DENSE).jaxpr, dense_shapes)
    assert (N, N, N, N) in dense_shapes


@pytest.mark.parametrize("block_rows", [1, 16])
def test_s8_chunked_scan_matches_single_block(block_rows):
    """Forcing tiny scan blocks (multi-chunk path) keeps the same result.

    For M = 300 pair rows the default cap yields a single full block; small
    `max_block_elements` exercise the multi-block `lax.scan` branch that
    large systems use, proving chunk boundaries are value-preserving.
    """
    rdm1 = make_rdm1("asymmetric")
    eri = packed_eri("s8")
    n_pairs = N * (N + 1) // 2
    single_block = packed_coulomb_potential(rdm1, eri)
    chunked = packed_coulomb_potential(
        rdm1, eri, max_block_elements=block_rows * n_pairs
    )
    assert block_rows < n_pairs  # the multi-block branch is actually forced
    np.testing.assert_allclose(chunked, single_block, rtol=1e-12, atol=1e-12)
