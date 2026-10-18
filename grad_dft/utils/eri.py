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
"""Symmetry-compressed electron-repulsion integral (ERI) storage.

This module defines the explicit storage configuration for ERIs and the
dedicated container used when GradDFT stores PySCF symmetry-compressed
integrals instead of the dense four-index tensor.

Storage formats
---------------
``"dense"``
    Plain real array of shape ``(n, n, n, n)`` (PySCF ``aosym="s1"``).
    The default. A dense ERI is always a plain array; it is never wrapped
    in :class:`PackedERI`, so existing behavior is unchanged.
``"s4"``
    Both within-pair symmetries applied, physical shape ``(M, M)`` with
    ``M = n (n + 1) / 2`` unique index pairs. Both index pairs are packed
    in PySCF's lower-triangular pair order, ``pair(a, b) = a (a + 1) / 2 + b``
    for ``a >= b`` (verified against ``pyscf.ao2mo.restore``). Roughly 4x
    smaller than dense.
``"s8"``
    Eightfold symmetry ``(ij|kl) = (kl|ij)`` applied on top of ``s4``,
    physical shape ``(T,)`` with ``T = M (M + 1) / 2``: the ``(M, M)``
    ``s4`` matrix packed into its lower triangle, row ``i`` occupying
    ``[i (i + 1) / 2, i (i + 1) / 2 + i + 1)``. Roughly 8x smaller than
    dense.

The eightfold relations ``(ij|kl) = (ji|kl) = (ij|lk) = (kl|ij)`` hold for
real ERIs of isolated molecules. They are *not* assumed for complex or
periodic integrals: those calculation modes are rejected by the generation
code (``eri_storage != "dense"`` raises) and keep their dense behavior.

Only PySCF's canonical names ``"s4"``/``"s8"`` are accepted as storage
configuration; PySCF also accepts the aliases ``"4"``/``"8"`` and the
partial compressions ``"s2ij"``/``"s2kl"``, which GradDFT deliberately
rejects with an informative error (see :func:`validate_eri_storage`).
"""

import numpy as np

from jax import numpy as jnp
from flax import struct
from jaxtyping import Array

#: Storage formats accepted by the ``eri_storage`` configuration.
ERI_STORAGE_FORMATS = ("dense", "s4", "s8")

#: Formats physically held by :class:`PackedERI`. ``"dense"`` is a plain array.
PACKED_STORAGE_FORMATS = ("s4", "s8")


def validate_eri_storage(value: str) -> str:
    r"""Validate an ``eri_storage`` configuration value.

    Parameters
    ----------
    value : str
        Requested ERI storage format.

    Returns
    -------
    str
        The validated format, unchanged.

    Raises
    ------
    ValueError
        If ``value`` is not one of :data:`ERI_STORAGE_FORMATS`. The message
        lists the supported formats and names the PySCF options that are
        intentionally not accepted (``"s2ij"``, ``"s2kl"``, ``"4"``, ``"8"``).
    """
    if value not in ERI_STORAGE_FORMATS:
        raise ValueError(
            f"Unsupported eri_storage format {value!r}. Supported formats are "
            f"{list(ERI_STORAGE_FORMATS)}: 'dense' is the full (n, n, n, n) "
            "tensor, 's4' and 's8' are PySCF symmetry-compressed packings. "
            "PySCF's partial compressions 's2ij'/'s2kl' and aliases '4'/'8' "
            "are not accepted; periodic and complex ERIs only support 'dense'."
        )
    return value


def pair_indices(n_orbitals: int) -> Array:
    r"""Map both members of an index pair to their unique-pair index.

    Uses PySCF's lower-triangular pair order: for :math:`a \ge b` the pair
    :math:`(a, b)` gets index :math:`a(a+1)/2 + b`, and the map is
    symmetric, ``pair(a, b) == pair(b, a)``. This is the packing convention
    behind ``aosym="s4"``/``"s8"`` (verified against ``pyscf.ao2mo.restore``).

    Parameters
    ----------
    n_orbitals : int
        Number of orbitals :math:`n`.

    Returns
    -------
    Int[Array, "orbitals orbitals"]
        Array of shape ``(n, n)`` with values in ``[0, M)``,
        ``M = n (n + 1) / 2``. The dtype follows the default JAX integer
        dtype (``int32`` without x64, ``int64`` with x64).
    """
    rows, cols = jnp.tril_indices(n_orbitals)
    pair = rows * (rows + 1) // 2 + cols
    index = jnp.zeros((n_orbitals, n_orbitals), dtype=pair.dtype)
    index = index.at[rows, cols].set(pair).at[cols, rows].set(pair)
    return index


def packed8_to_packed4(packed8: Array, n_pairs: int) -> Array:
    r"""Expand an ``s8``-packed ERI to the ``s4`` ``(M, M)`` representation.

    Row ``i`` of the ``(M, M)`` matrix occupies
    ``[i (i + 1) / 2, i (i + 1) / 2 + i + 1)`` in the ``s8`` vector; entries
    above the diagonal are recovered with the pair-exchange symmetry
    ``(ij|kl) = (kl|ij)``.

    Parameters
    ----------
    packed8 : Array
        Rank-1 ``s8`` ERI of shape ``(T,)``, ``T = M (M + 1) / 2``.
    n_pairs : int
        Number of unique index pairs ``M`` (i.e. ``n (n + 1) / 2`` for
        ``n`` orbitals).

    Returns
    -------
    Float[Array, "pair pair"]
        The symmetric ``(M, M)`` matrix in ``s4`` pair order.
    """
    pair = jnp.arange(n_pairs)
    start = pair * (pair + 1) // 2  # row start offsets in the s8 vector
    rows, cols = pair[:, None], pair[None, :]
    source = jnp.where(cols <= rows, start[:, None] + cols, start[None, :] + rows)
    return packed8[source]


def packed_to_dense(eri: "PackedERI") -> Array:
    r"""Materialize the dense ``(n, n, n, n)`` ERI from a :class:`PackedERI`.

    This is the explicit compatibility boundary of the compressed
    representation: it allocates a full :math:`N^4` array (and, for
    ``s8``, an additional ``(M, M)`` intermediate). It is never called on
    the training path — contractions consume packed storage directly.

    Parameters
    ----------
    eri : PackedERI
        Compressed ERI (``storage_format`` ``"s4"`` or ``"s8"``).

    Returns
    -------
    Float[Array, "orbitals orbitals orbitals orbitals"]
        Dense ERI with the same dtype as ``eri.packed``.

    Raises
    ------
    ValueError
        If ``eri.storage_format`` is not a known packed format.
    """
    if eri.storage_format == "s4":
        packed4 = eri.packed
    elif eri.storage_format == "s8":
        n_pairs = eri.n_orbitals * (eri.n_orbitals + 1) // 2
        packed4 = packed8_to_packed4(eri.packed, n_pairs)
    else:  # unreachable through the constructor; defensive against crafted treedefs
        raise ValueError(
            f"Unknown PackedERI storage_format {eri.storage_format!r}; "
            f"expected one of {list(PACKED_STORAGE_FORMATS)}."
        )
    index = pair_indices(eri.n_orbitals)
    return packed4[index[:, :, None, None], index[None, None, :, :]]


@struct.dataclass
class PackedERI:
    r"""Symmetry-compressed electron-repulsion integrals.

    A JAX PyTree holding PySCF symmetry-compressed ERIs in :class:`Molecule`
    in place of the dense ``(n, n, n, n)`` tensor. The packed numerical
    array is the single dynamic leaf; ``storage_format`` and ``n_orbitals``
    are static (auxiliary) metadata, so the format is explicit metadata and
    never inferred from array shape.

    The *logical* shape is always ``(n, n, n, n)``; the *physical* storage
    shape is ``packed.shape`` (see module docstring for both).

    Parameters
    ----------
    packed : Array or ndarray
        Physical storage: ``(M, M)`` for ``"s4"`` (``M = n (n + 1) / 2``)
        or ``(T,)`` for ``"s8"`` (``T = M (M + 1) / 2``).
    storage_format : str
        One of :data:`PACKED_STORAGE_FORMATS`.
    n_orbitals : int
        Number of orbitals :math:`n` of the logical tensor.

    Raises
    ------
    ValueError
        If the format is unknown, ``n_orbitals`` is not a positive integer,
        the dtype is not a real floating dtype (e.g. complex ERIs are
        unsupported), or ``packed``'s shape does not match the format and
        ``n_orbitals``.
    """

    packed: Array
    storage_format: str = struct.field(pytree_node=False)
    n_orbitals: int = struct.field(pytree_node=False)

    def __post_init__(self):
        if self.storage_format not in PACKED_STORAGE_FORMATS:
            raise ValueError(
                f"Unknown PackedERI storage_format {self.storage_format!r}; "
                f"expected one of {list(PACKED_STORAGE_FORMATS)}. "
                "Dense ERIs are stored as plain arrays, not as PackedERI."
            )
        if not isinstance(self.n_orbitals, (int, np.integer)) or self.n_orbitals < 1:
            raise ValueError(
                f"PackedERI n_orbitals must be a positive integer, got {self.n_orbitals!r}."
            )
        dtype = np.dtype(self.packed.dtype)
        if np.issubdtype(dtype, np.complexfloating):
            raise ValueError(
                f"PackedERI requires a real dtype, got {dtype}; complex ERIs "
                "(e.g. multi-k-point periodic calculations) are unsupported and "
                "must keep eri_storage='dense'."
            )
        if not np.issubdtype(dtype, np.floating):
            raise ValueError(f"PackedERI requires a floating dtype, got {dtype}.")
        n = int(self.n_orbitals)
        n_pairs = n * (n + 1) // 2
        expected = (
            (n_pairs, n_pairs)
            if self.storage_format == "s4"
            else (n_pairs * (n_pairs + 1) // 2,)
        )
        if tuple(self.packed.shape) != expected:
            raise ValueError(
                f"storage_format={self.storage_format!r} with n_orbitals={n} "
                f"requires packed shape {expected}, got {tuple(self.packed.shape)}."
            )

    @property
    def shape(self) -> tuple:
        r"""Logical ERI shape ``(n_orbitals, n_orbitals, n_orbitals, n_orbitals)``.

        The physical storage shape is ``self.packed.shape``.
        """
        return (self.n_orbitals,) * 4

    @property
    def dtype(self):
        r"""Dtype of the physical packed storage (preserved by conversion)."""
        return self.packed.dtype

    def to_dense(self) -> Array:
        r"""Materialize the dense ``(n, n, n, n)`` ERI.

        Explicit :math:`N^4` allocation — never used on the training path.
        See :func:`packed_to_dense` for details.
        """
        return packed_to_dense(self)
