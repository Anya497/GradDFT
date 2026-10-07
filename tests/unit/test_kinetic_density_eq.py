"""Equivalence tests for :func:`grad_dft.molecule.kinetic_density`.

Checks that the rewritten einsum formulation produces the same result as the
original one, which materializes an intermediate of shape
``[grid, orbitals, orbitals]``, across spin counts, system sizes, grid sizes
and XLA precisions.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.lax import Precision

from grad_dft.molecule import kinetic_density


@partial(jax.jit, static_argnames="precision")
def kinetic_density_old(
    rdm1: jax.Array, grad_ao: jax.Array, precision: Precision = Precision.HIGHEST
) -> jax.Array:
    """Reference implementation: the pre-optimization kinetic density einsum."""
    return 0.5 * jnp.einsum(
        "...ab,raj,rbj->r...", rdm1, grad_ao, grad_ao, precision=precision
    )


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("orbitals", [10, 25])
@pytest.mark.parametrize("grid", [30, 50])
@pytest.mark.parametrize(
    "precision",
    [
        Precision.HIGHEST,
        Precision.HIGH,
        Precision.DEFAULT,
    ],
)
def test_kinetic_density_equivalence(spin, orbitals, grid, precision):
    """The optimized and reference kinetic_density agree within 1e-6."""
    rng = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(rng)

    rdm1_real = jax.random.randint(
        key1, (spin, orbitals, orbitals), minval=-80, maxval=80
    ).astype(jnp.float64)
    rdm1 = 0.5 * (rdm1_real + rdm1_real.transpose(0, 2, 1))
    grad_ao = jax.random.randint(
        key2, (grid, orbitals, 3), minval=-80, maxval=80
    ).astype(jnp.float64)
    result_old = kinetic_density_old(rdm1, grad_ao, precision)
    result_new = kinetic_density(rdm1, grad_ao, precision)

    np.testing.assert_allclose(
        result_old,
        result_new,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"Results differ for spin={spin}, orbitals={orbitals}, "
        f"grid={grid}, precision={precision}",
    )
