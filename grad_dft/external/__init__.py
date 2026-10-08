"""Adapters around external code that GradDFT builds on.

``Functional``, ``NeuralNumInt`` and ``_SystemState`` come from the installed
``density_functional_approximation_dm21`` package through the compatibility
layer in :mod:`grad_dft.external._dm21_compat`; ``_nu_chunk`` is GradDFT's
own JAX implementation.
"""

from grad_dft.external._dm21_compat import Functional, NeuralNumInt, _SystemState
from grad_dft.external._hf_density import _nu_chunk
