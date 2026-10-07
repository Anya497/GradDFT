"""PySCF configuration consumed by CI via PYSCF_CONFIG_FILE.

Kept in sync with local CI-reproduction runs: set
PYSCF_CONFIG_FILE=.github/workflows/pyscf_conf.py to get identical PySCF
behavior locally.
"""

B3LYP_WITH_VWN5 = True

# PySCF's default memory budget is 4000 MB. The CI process carries JAX,
# TensorFlow and PySCF in one interpreter, so its RSS is already ~4300 MB
# when the solids ERI tests run; fft_ao2mo then sees a negative remaining
# budget and crashes with "ValueError: negative dimensions are not allowed".
# 8000 MB restores headroom over the observed RSS while keeping total usage
# well inside the runner's RAM. This file overrides PYSCF_MAX_MEMORY
# (pyscf/__config__.py loads it after the environment).
MAX_MEMORY = 8000
