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

"""End-to-end equivalence of compressed ERI storage (S6 of #15).

On real converged SCF molecules (H2/sto-3g and, larger, H2O/sto-3g) this
verifies that symmetry-compressed ERIs (s4/s8) are numerically equivalent to
the dense representation through the full GradDFT pipeline: molecule
construction, dense expansion, Coulomb contractions, nonXC energies, and a
training loss with gradients wrt the functional parameters — while the
default (no ``eri_storage``) behavior stays dense and unchanged.
"""

# The test functions below take pytest fixtures as arguments.
# pylint: disable=redefined-outer-name

from jax import config

import jax
import jax.numpy as jnp
from jax.nn import gelu, sigmoid
from jax.random import PRNGKey
import numpy as np
import pytest
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

SCF_ITERS = 200
PACKED_FORMATS = ("s4", "s8")
ENERGY_TOL = 1e-9  # contraction summation order differs from the dense einsum
GRAD_TOL = 1e-8  # gradients accumulate the same 1e-14-level differences

H2_MOL = gto.M(
    atom="H 0.0 0.0 -0.3705\nH 0.0 0.0 0.3705",
    basis="sto-3g",
    verbose=0,
)
H2O_MOL = gto.M(
    atom="O 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692",
    basis="sto-3g",
    verbose=0,
)


def converged_mf(mol, density_fit: bool = False):
    r"""Converge a spin-restricted Kohn-Sham calculation with no XC energy."""
    mf = dft.RKS(mol)
    mf.xc = "0.00*LDA"  # no XC energy, as in test_non_xc_energy.py
    mf.max_cycle = SCF_ITERS
    if density_fit:
        mf = mf.density_fit()
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def h2_data():
    r"""H2/sto-3g SCF plus its dense and packed Molecules."""
    mf = converged_mf(H2_MOL)
    molecules = {"dense": molecule_from_pyscf(mf)}
    for fmt in PACKED_FORMATS:
        molecules[fmt] = molecule_from_pyscf(mf, eri_storage=fmt)
    return mf, molecules


@pytest.fixture(scope="module")
def h2o_data():
    r"""H2O/sto-3g SCF plus its dense and packed Molecules."""
    mf = converged_mf(H2O_MOL)
    molecules = {"dense": molecule_from_pyscf(mf)}
    for fmt in PACKED_FORMATS:
        molecules[fmt] = molecule_from_pyscf(mf, eri_storage=fmt)
    return mf, molecules


@pytest.fixture(scope="module")
def h2_df_data():
    r"""Density-fitted H2/sto-3g SCF plus dense and packed Molecules."""
    mf = converged_mf(H2_MOL, density_fit=True)
    molecules = {"dense": molecule_from_pyscf(mf)}
    for fmt in PACKED_FORMATS:
        molecules[fmt] = molecule_from_pyscf(mf, eri_storage=fmt)
    return mf, molecules


def coefficient_inputs(molecule: Molecule, *_, **__):
    r"""Single-feature density input (clipped), following test_training.py."""
    rho = jnp.clip(molecule.density(), min=1e-30)
    return jnp.concatenate((rho,), axis=1)


def energy_densities(molecule: Molecule, *_, **__):
    r"""LSDA exchange energy density feature, following test_training.py."""
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


def make_functional() -> NeuralFunctional:
    r"""Tiny neural functional with trainable parameters."""
    return NeuralFunctional(coefficients, energy_densities, coefficient_inputs)


def test_default_eri_storage_stays_dense(h2_data, h2o_data):
    r"""Without eri_storage, rep_tensor is the unchanged dense (n, n, n, n) array."""
    for mf, molecules in (h2_data, h2o_data):
        dense = molecules["dense"].rep_tensor
        n_orb = mf.mol.nao_nr()
        assert isinstance(dense, jnp.ndarray)
        assert not isinstance(dense, PackedERI)
        assert dense.shape == (n_orb, n_orb, n_orb, n_orb)
        np.testing.assert_allclose(
            dense, mf.mol.intor("int2e", aosym="s1"), atol=1e-14, rtol=0
        )


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_packed_storage_metadata_and_values(h2_data, h2o_data, fmt):
    r"""Packed molecules carry correct metadata and expand to the PySCF ERI."""
    for mf, molecules in (h2_data, h2o_data):
        packed = molecules[fmt].rep_tensor
        assert isinstance(packed, PackedERI)
        assert packed.storage_format == fmt
        assert packed.n_orbitals == mf.mol.nao_nr()
        reference = jnp.asarray(mf.mol.intor("int2e", aosym="s1"))
        np.testing.assert_allclose(
            packed_to_dense(packed), reference, atol=1e-14, rtol=0
        )


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_nonxc_and_coulomb_match_dense(h2_data, h2o_data, fmt):
    r"""nonXC energies and Coulomb potentials agree with dense storage."""
    for _, molecules in (h2_data, h2o_data):
        dense, packed = molecules["dense"], molecules[fmt]
        np.testing.assert_allclose(
            dense.nonXC(), packed.nonXC(), atol=ENERGY_TOL, rtol=1e-10
        )
        np.testing.assert_allclose(
            dense.get_coulomb_potential(),
            packed.get_coulomb_potential(),
            atol=ENERGY_TOL,
            rtol=1e-10,
        )


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_energy_loss_and_grad_match_dense(h2_data, fmt):
    r"""A training loss and its parameter gradients are storage-independent.

    `simple_energy_loss` is itself value_and_grad-wrapped (has_aux=True) and
    returns `((loss, E_predict), grads_wrt_params)`. The loss includes the
    nonXC (ERI) energy of each storage, while the parameter gradients cover
    the neural-functional path of the same computation.
    """
    mf, molecules = h2_data
    functional = make_functional()
    params = functional.init(PRNGKey(42), coefficient_inputs(molecules["dense"]))
    predictor = non_scf_predictor(functional)
    truth_energy = mf.e_tot

    def loss_and_grads(params, molecule):
        (loss, _), grads = simple_energy_loss(params, predictor, molecule, truth_energy)
        return loss, grads

    dense_loss, dense_grads = loss_and_grads(params, molecules["dense"])
    packed_loss, packed_grads = loss_and_grads(params, molecules[fmt])

    assert not jnp.isnan(dense_loss) and not jnp.isnan(packed_loss)
    np.testing.assert_allclose(packed_loss, dense_loss, atol=GRAD_TOL, rtol=1e-10)

    flat_dense, _ = jax.tree_util.tree_flatten(dense_grads)
    flat_packed, _ = jax.tree_util.tree_flatten(packed_grads)
    for grad_dense, grad_packed in zip(flat_dense, flat_packed):
        np.testing.assert_allclose(grad_packed, grad_dense, atol=GRAD_TOL, rtol=1e-8)
    # guard against a vacuous all-zero comparison
    assert any(float(jnp.abs(g).max()) > 1e-10 for g in flat_dense)


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_nonxc_rdm1_gradient_matches_dense(h2_data, h2o_data, fmt):
    r"""d(nonXC)/d(rdm1) is storage-independent: autodiff through the packed ops.

    Unlike the parameter gradients above, this gradient flows *through* the
    compressed Coulomb contractions (the nonXC energy depends on rdm1), so it
    exercises backpropagation of the packed8/packed4 matmuls directly.
    """
    for _, molecules in (h2_data, h2o_data):
        dense, packed = molecules["dense"], molecules[fmt]

        def grad_wrt_rdm1(molecule):
            return jax.grad(lambda r: molecule.replace(rdm1=r).nonXC())(molecule.rdm1)

        grad_dense = grad_wrt_rdm1(dense)
        grad_packed = grad_wrt_rdm1(packed)
        assert not jnp.isnan(grad_packed).any()
        assert float(jnp.abs(grad_dense).max()) > 1e-10  # non-trivial
        np.testing.assert_allclose(grad_packed, grad_dense, atol=GRAD_TOL, rtol=1e-9)


def test_density_fitted_consistent_across_formats(h2_df_data):
    r"""Density-fitted ERIs are consistent across dense/s4/s8 storage."""
    _, molecules = h2_df_data
    reference = np.asarray(molecules["dense"].rep_tensor)
    for fmt in PACKED_FORMATS:
        packed = molecules[fmt].rep_tensor
        assert isinstance(packed, PackedERI)
        np.testing.assert_allclose(
            packed_to_dense(packed), reference, atol=1e-14, rtol=0
        )
        np.testing.assert_allclose(
            molecules["dense"].nonXC(),
            molecules[fmt].nonXC(),
            atol=ENERGY_TOL,
            rtol=1e-10,
        )


def test_rep_tensor_none_molecule_path(h2_data):
    r"""A molecule without ERIs still supports its non-ERI methods."""
    _, molecules = h2_data
    no_eri = molecules["dense"].replace(rep_tensor=None)
    density = no_eri.density()
    grad_density = no_eri.grad_density()
    assert not jnp.isnan(density).any()
    assert not jnp.isnan(grad_density).any()
