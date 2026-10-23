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

"""Tests for serialization of compressed ERIs (S4 of #15).

Covers:

(B/D/E) `saver`/`loader` round trips with a synthetic Molecule (no SCF):
    dense layout unchanged (no attributes, old files load), packed s4/s8
    preserves storage_format/n_orbitals/values/dtype, `nonXC` agrees with
    the original, and `loader(training=True)` still reconstructs packed
    molecules (metadata rides on dataset attrs, which survive the
    training-mode group-attr skip);
(B) the reaction dataset loop of `loader` as well as the molecule loop;
(F) a corrupt `storage_format` attribute raises an informative ValueError;
(E) an old-style dense file (plain dataset, no attributes) loads as dense.
"""

from jax import config

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto

from grad_dft.interface.pyscf import loader, saver
from grad_dft.molecule import Grid, Molecule, make_reaction
from grad_dft.utils.eri import PackedERI

config.update("jax_enable_x64", True)

PACKED_FORMATS = ("s4", "s8")

MOL = gto.M(
    atom=[["H", (0, 0, 0)], ["H", (0.74, 0, 0)]],
    basis="cc-pvdz",
    verbose=0,
)
N = MOL.nao_nr()
REP_DENSE = jnp.asarray(MOL.intor("int2e", aosym="s1"))


def packed_eri(fmt: str) -> PackedERI:
    """Build PackedERI storage of the test molecule's integrals (numpy leaf)."""
    return PackedERI(np.asarray(MOL.intor("int2e", aosym=fmt)), fmt, N)


def make_molecule(rep_tensor) -> Molecule:
    """Minimal synthetic Molecule with real ERIs, ready for save/load."""
    grid_size = 8
    rng = np.random.default_rng(13)
    grid = Grid(
        coords=jnp.zeros((grid_size, 3)),
        weights=jnp.full((grid_size,), 1.0 / grid_size),
    )
    rdm1 = jnp.asarray(rng.standard_normal((N, N)))
    rdm1 = 0.5 * (rdm1 + rdm1.T)
    spin_rdm1 = jnp.stack([0.6 * rdm1, 0.4 * rdm1])
    return Molecule(
        grid=grid,
        atom_index=jnp.array([0, 0]),
        nuclear_pos=jnp.zeros((2, 3)),
        ao=jnp.asarray(rng.standard_normal((grid_size, N))),
        grad_ao=jnp.asarray(rng.standard_normal((grid_size, N, 3))),
        grad_n_ao={k: jnp.zeros((grid_size, N)) for k in range(3)},
        rdm1=spin_rdm1,
        nuclear_repulsion=jnp.asarray(0.35),
        h1e=jnp.asarray(rng.standard_normal((N, N))),
        vj=jnp.zeros((2, N, N)),
        mo_coeff=jnp.zeros((2, N, N)),
        mo_occ=jnp.zeros((2, N)),
        mo_energy=jnp.zeros((2, N)),
        rep_tensor=rep_tensor,
        name=jnp.array([ord(char) for char in "H2"]),
        basis=jnp.array([ord(char) for char in "ccpvdz"]),
        energy=jnp.asarray(-1.16),
    )


def write_old_style_dense_file(path: str, molecule: Molecule) -> None:
    """Write an HDF5 file with the pre-compression layout (no dataset attrs)."""
    with h5py.File(f"{path}.hdf5", "w") as file:
        group = file.create_group("molecule_0")
        group.create_dataset("coords", data=np.asarray(molecule.grid.coords))
        group.create_dataset("weights", data=np.asarray(molecule.grid.weights))
        for key in (
            "atom_index",
            "nuclear_pos",
            "ao",
            "grad_ao",
            "rdm1",
            "nuclear_repulsion",
            "h1e",
            "vj",
            "mo_coeff",
            "mo_occ",
            "mo_energy",
            "energy",
        ):
            group.create_dataset(key, data=np.asarray(getattr(molecule, key)))
        grad_group = group.create_group("grad_n_ao")
        for k, v in molecule.grad_n_ao.items():
            grad_group.create_dataset(f"{k}", data=np.asarray(v))
        group.create_dataset("rep_tensor", data=np.asarray(molecule.rep_tensor))
        group.create_dataset("name", data="H2")
        group.create_dataset("basis", data="ccpvdz")


def load_one(path: str, training: bool):
    """Load the single molecule of a file as (kind, molecule)."""
    return next(iter(loader(path, randomize=False, training=training)))


def test_dense_round_trip_unchanged(tmp_path):
    """Dense files keep the old layout (no attrs) and load as dense arrays."""
    path = str(tmp_path / "dense")
    original = make_molecule(REP_DENSE)
    saver(path, molecules=[original])

    with h5py.File(f"{path}.hdf5", "r") as file:
        (group_name,) = [key for key in file.keys() if "molecule" in key]
        dataset = file[group_name]["rep_tensor"]
        assert dataset.ndim == 4
        assert not dict(dataset.attrs)

    kind, molecule = load_one(path, training=False)
    assert kind == "molecule"
    assert isinstance(molecule.rep_tensor, jnp.ndarray)
    np.testing.assert_array_equal(
        np.asarray(molecule.rep_tensor), np.asarray(original.rep_tensor)
    )


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_packed_round_trip(tmp_path, fmt):
    """Packed s4/s8 storage survives save/load with metadata, values, dtype."""
    path = str(tmp_path / f"packed_{fmt}")
    original = make_molecule(packed_eri(fmt))
    saver(path, molecules=[original])

    with h5py.File(f"{path}.hdf5", "r") as file:
        (group_name,) = [key for key in file.keys() if "molecule" in key]
        dataset = file[group_name]["rep_tensor"]
        # s4 is stored as the M x M packed matrix, s8 as the flat T-vector.
        assert dataset.ndim == (2 if fmt == "s4" else 1)
        assert str(dataset.attrs["storage_format"]) == fmt
        assert int(dataset.attrs["n_orbitals"]) == N

    kind, molecule = load_one(path, training=False)
    assert kind == "molecule"
    loaded = molecule.rep_tensor
    assert isinstance(loaded, PackedERI)
    assert loaded.storage_format == fmt
    assert loaded.n_orbitals == N
    assert loaded.packed.dtype == jnp.float64
    np.testing.assert_array_equal(
        np.asarray(loaded.packed), np.asarray(original.rep_tensor.packed)
    )
    np.testing.assert_allclose(
        molecule.nonXC(), original.nonXC(), atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_training_mode_reconstructs_packed(tmp_path, fmt):
    """Dataset attrs survive the training-mode group-attr skip."""
    path = str(tmp_path / f"train_{fmt}")
    original = make_molecule(packed_eri(fmt))
    saver(path, molecules=[original])

    _, molecule = load_one(path, training=True)
    loaded = molecule.rep_tensor
    assert isinstance(loaded, PackedERI)
    assert loaded.storage_format == fmt
    assert loaded.n_orbitals == N
    np.testing.assert_array_equal(
        np.asarray(loaded.packed), np.asarray(original.rep_tensor.packed)
    )


def test_reaction_round_trip_packed(tmp_path):
    """The reaction dataset loop reconstructs packed reactants and products."""
    path = str(tmp_path / "reaction")
    reactant = make_molecule(packed_eri("s8"))
    product = make_molecule(packed_eri("s4"))
    saver(path, reactions=[make_reaction([reactant], [product], energy=-1.25)])

    _, reaction = next(iter(loader(path, randomize=False, training=True)))
    reactant_loaded = reaction.reactants[0].rep_tensor
    product_loaded = reaction.products[0].rep_tensor
    assert isinstance(reactant_loaded, PackedERI)
    assert reactant_loaded.storage_format == "s8"
    assert isinstance(product_loaded, PackedERI)
    assert product_loaded.storage_format == "s4"
    np.testing.assert_array_equal(
        np.asarray(reactant_loaded.packed), np.asarray(reactant.rep_tensor.packed)
    )
    np.testing.assert_array_equal(
        np.asarray(product_loaded.packed), np.asarray(product.rep_tensor.packed)
    )


def test_old_style_dense_file_loads_as_dense(tmp_path):
    """A pre-compression file (plain dense dataset, no attrs) loads as dense."""
    path = str(tmp_path / "old_style")
    original = make_molecule(REP_DENSE)
    write_old_style_dense_file(path, original)

    kind, molecule = load_one(path, training=False)
    assert kind == "molecule"
    assert isinstance(molecule.rep_tensor, jnp.ndarray)
    assert molecule.rep_tensor.shape == (N, N, N, N)
    np.testing.assert_array_equal(
        np.asarray(molecule.rep_tensor), np.asarray(original.rep_tensor)
    )


def test_unknown_storage_format_raises(tmp_path):
    """A corrupt storage_format attribute raises an informative ValueError."""
    path = str(tmp_path / "corrupt")
    saver(path, molecules=[make_molecule(packed_eri("s8"))])
    with h5py.File(f"{path}.hdf5", "a") as file:
        (group_name,) = [key for key in file.keys() if "molecule" in key]
        file[group_name]["rep_tensor"].attrs["storage_format"] = "bogus"

    with pytest.raises(ValueError, match="Supported formats are"):
        load_one(path, training=False)
