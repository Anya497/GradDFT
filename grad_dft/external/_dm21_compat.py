# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility layer between the installed DM21 package and GradDFT.

``density_functional_approximation_dm21`` is installed from its upstream
repository (pinned in ``requirements.txt``). Upstream targets TF-Hub's TF1
Module API and a pre-setuptools-81 environment, while GradDFT pins
tensorflow-hub >= 0.16, runs on setuptools >= 81 (no ``pkg_resources``) and
PySCF >= 2.3. This module carries exactly the three deltas that make the
installed package run in this environment; everything else is used
unmodified:

1. A ``pkg_resources`` shim, required because ``tensorflow_hub.__init__``
   imports ``pkg_resources.parse_version`` at import time while setuptools
   >= 81 no longer ships ``pkg_resources``.
2. Shims for the TF1 Module API symbols TF-Hub 0.16 removed
   (``hub.Module``, ``hub.add_signature``, ``hub.create_module_spec``), so
   upstream's ``_build_graph`` runs unmodified against the SavedModel
   checkpoints via ``hub.load``.
3. The ``eval_xc_eff`` override PySCF >= 2.3 requires: its nr_rks/nr_uks
   grid loops call ``eval_xc_eff``, whose base-class implementation
   dispatches to the libxc backend (``self.libxc.eval_xc1``) instead of this
   subclass's ``eval_xc``, and strips the laplacian row that DM21's graph
   placeholders expect.

The three deltas were developed in the vendored copy under
``grad_dft/external/density_functional_approximation_dm21/`` (tasks #6 and
#10) and are carried over verbatim where task #13 replaced the vendored copy
with the installed package.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

# Delta 1: tensorflow-hub imports pkg_resources.parse_version at import time,
# so the shim must run before tensorflow_hub is imported. try/except keeps the
# real pkg_resources when the environment has one (setuptools < 81).
try:
    import pkg_resources  # pylint: disable=unused-import
except ModuleNotFoundError:
    # setuptools >= 81 removed pkg_resources. Provide the single symbol
    # tensorflow-hub needs.
    import sys as _sys
    from types import SimpleNamespace as _SimpleNamespace

    def _parse_version(version):
        """Returns a comparable key for a dotted numeric version string."""
        return tuple(int(part) for part in version.split(".") if part.isdigit())

    _sys.modules.setdefault(
        "pkg_resources", _SimpleNamespace(parse_version=_parse_version)
    )

# Delta 2 (and the delta-1 shim above) intentionally come before these
# imports: the module's whole purpose is to control import order.
# pylint: disable=wrong-import-position
import tensorflow_hub as hub
import density_functional_approximation_dm21.neural_numint as _neural_numint

if not hasattr(hub, "Module"):

    class _SavedModelModule:  # pylint: disable=too-few-public-methods
        """Stand-in for the TF1 ``hub.Module`` that TF-Hub 0.16 removed.

        The DM21 checkpoints are SavedModels, so ``spec`` (a checkpoint
        directory) is loaded once with ``hub.load`` and called through its
        ``default`` signature, matching upstream's call convention
        ``module(tensor_dict, as_dict=True)`` whose keyword names keep the
        ``tensor_dict$`` prefix.
        """

        def __init__(self, spec):
            """Loads the SavedModel's ``default`` signature once."""
            self._signature = hub.load(spec).signatures["default"]

        def __call__(self, tensor_dict, as_dict=True):
            """Applies the functional to the feature tensors in ``tensor_dict``."""
            del as_dict  # the signature result already is a dict of tensors
            return self._signature(**tensor_dict)

    def _add_signature(*args, **kwargs):
        """No-op replacing the removed ``hub.add_signature``.

        Upstream registers the functional-derivative outputs as a module
        signature here. The derivatives stay graph attributes consumed by
        ``eval_xc`` through the session, so only the export path (unsupported,
        see below) misses the registered signature.
        """
        del args, kwargs  # unused

    def _create_module_spec(*args, **kwargs):
        """Raises for upstream's export path, which needs the removed TF1 API.

        ``export_functional_and_derivatives`` builds an exportable TF1 Hub
        module from these primitives; TF-Hub 0.16 removed them, and a TF2
        ``tf.saved_model.save`` reimplementation would be needed instead. Only
        upstream's ``export_saved_model.py`` script and ``neural_numint_test.py``
        use this path, and neither is part of GradDFT's test suite.
        """
        del args, kwargs  # unused
        raise NotImplementedError(
            "export_functional_and_derivatives requires the TF1 Hub Module API "
            "(hub.create_module_spec / hub.Module.export) that tensorflow-hub "
            "0.16 removed; a TF2 tf.saved_model.save reimplementation would be "
            "needed instead."
        )

    hub.Module = _SavedModelModule
    hub.add_signature = _add_signature
    hub.create_module_spec = _create_module_spec

Functional = _neural_numint.Functional
_SystemState = _neural_numint._SystemState  # pylint: disable=protected-access


class NeuralNumInt(  # pylint: disable=too-few-public-methods
    _neural_numint.NeuralNumInt
):
    """DM21 numerical integration from the installed package, plus PySCF >= 2.3.

    PySCF's nr_rks/nr_uks grid loops reach the functional through
    ``eval_xc_eff`` (delta 3 in the module docstring), which upstream does not
    implement; the override below routes the evaluation back into upstream's
    ``eval_xc`` and converts its sigma derivatives to the density-parameter
    derivatives PySCF contracts with the atomic orbitals.
    """

    def eval_xc_eff(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        xc_code: str,
        rho: np.ndarray,
        deriv: int = 1,
        omega: Optional[float] = None,
        xctype: Optional[str] = None,
        verbose=None,
        spin: Optional[int] = None,
    ) -> Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray, None, None]]:
        """Evaluates the XC energy and derivatives against density parameters.

        PySCF's nr_rks/nr_uks grid loops call this method rather than eval_xc.
        It differs from eval_xc only in the derivative convention: eval_xc
        returns derivatives with respect to sigma = |nabla rho|^2, while
        eval_xc_eff returns them with respect to the density parameters
        [rho, nabla rho, tau]. This override applies that chain rule to
        eval_xc's outputs, so the DM21 evaluation itself (including the local
        Hartree-Fock side effects on self._vmat_hf) stays in eval_xc.

        See pyscf.dft.numint.NumInt.eval_xc_eff for more details on the
        interface. The layouts below were verified against that method
        (max |diff| = 0.0 on mgga_x_tpss for both spin channels).

        Args:
          xc_code: unused (see eval_xc).
          rho: density and density derivatives at each grid point, shape
            (5, N) [rho, nabla_x, nabla_y, nabla_z, tau], or (2, 5, N) for
            spin-polarized calculations; a laplacian-carrying (6, N) /
            (2, 6, N) rho is accepted as well. eval_rho runs with
            MGGA_DENSITY_LAPL off, so rho arrives without the laplacian row
            the graph's (6, N) placeholders expect; a zero row is inserted
            before evaluation (the graph unstacks and discards it).
          deriv: derivative order. 1 (the default) is supported; 0 evaluates
            the energy and the local Hartree-Fock side effects but returns no
            potential; greater than 1 raises.
          omega: RSH parameter. None (the default) is passed through to
            eval_xc, which raises for an explicit value: DM21 has no
            range-separation support.
          xctype: unused. The DM21 functional is always a meta-GGA.
          verbose: unused.
          spin: 0 for a spin-unpolarized (restricted) calculation, 1 for
            spin-polarized. Inferred from rho's leading dimension when None,
            the same rule as pyscf.dft.numint.NumInt.eval_xc_eff.

        Returns:
          exc, vxc, fxc, kxc, where:
            exc is the XC energy density at each grid point, shape (N).
            vxc has shape (5, N) for spin=0, with rows
            [dE/d rho, dE/d nabla rho (3 components), dE/d tau], or shape
            (2, 5, N) for spin=1 with the same rows for the alpha and beta
            channels (rows [rho, nabla rho, tau] of each).
            fxc is None. (Second derivatives are not implemented.)
            kxc is None. (Third derivatives are not implemented.)
            For deriv=0 the return is [exc, None, None, None].

        Raises:
          NotImplementedError: if deriv > 1 (DM21 provides no second or higher
            derivatives; the base class would fail inside transform_xc).
        """
        if omega is None:
            omega = self.omega
        if deriv > 1:
            raise NotImplementedError(
                "DM21 functionals provide no second or higher derivatives, so "
                "eval_xc_eff cannot be called with deriv > 1."
            )
        del xctype  # unused: the DM21 functional is always a meta-GGA

        rho = np.asarray(rho, order="C", dtype=np.float64)
        if spin is None:
            spin = 1 if (rho.ndim >= 2 and rho.shape[0] == 2) else 0
        if rho.shape[-2] == 5:
            rho = np.insert(rho, 4, 0.0, axis=-2)

        exc, (vrho, vsigma, _vlapl, vtau), _, _ = self.eval_xc(
            xc_code, rho, spin=spin, deriv=deriv, omega=omega, verbose=verbose
        )
        if deriv < 1:
            # Evaluated for exc and the _vmat_hf side effect; no potential is
            # requested at derivative order 0.
            return [exc, None, None, None]

        # Chain rule from eval_xc's sigma derivatives (dE/d sigma, with sigma
        # the PySCF/libxc (sigma_aa, sigma_ab, sigma_bb) components) to the
        # density-parameter derivatives PySCF contracts with the AOs. To
        # re-derive it, follow PySCF's MGGA branches: nr_rks does
        # wv = weight * vxc, wv[0] *= .5, wv[4] *= .5,
        # _scale_ao_sparse(ao[:4], wv[:4], ...) and _tau_dot_sparse(ao, ao,
        # wv[4], ...); nr_uks does the same per spin row (wv[:, 0] and
        # wv[:, 4]) — i.e. row 0 is dE/d rho, rows 1:4 are dE/d nabla rho and
        # row 4 is dE/d tau. dE/d nabla rho = 2 (dE/d sigma) nabla rho for a
        # single sigma, and the pairwise expansion below for sigma_ab.
        if spin == 0:
            # sigma = nabla rho . nabla rho, so the gradient rows pick up a
            # factor of 2.
            vxc = np.concatenate(
                [
                    vrho[None, :],
                    2.0 * vsigma[None, :] * rho[1:4],
                    vtau[None, :],
                ],
                axis=0,
            )
        else:
            nabla_a, nabla_b = rho[0, 1:4], rho[1, 1:4]
            vsigma_aa, vsigma_ab, vsigma_bb = vsigma[:, 0], vsigma[:, 1], vsigma[:, 2]
            vxc_a = np.concatenate(
                [
                    vrho[:, 0][None, :],
                    2.0 * vsigma_aa[None, :] * nabla_a + vsigma_ab[None, :] * nabla_b,
                    vtau[:, 0][None, :],
                ],
                axis=0,
            )
            vxc_b = np.concatenate(
                [
                    vrho[:, 1][None, :],
                    2.0 * vsigma_bb[None, :] * nabla_b + vsigma_ab[None, :] * nabla_a,
                    vtau[:, 1][None, :],
                ],
                axis=0,
            )
            vxc = np.stack([vxc_a, vxc_b], axis=0)
        return exc, vxc, None, None
