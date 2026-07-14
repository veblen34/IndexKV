"""Method registry population.

Importing this package imports every method module, each of which registers
itself via ``@register``. This build ships the evaluation framework and the
source-backed baseline adapters. The set is a dense reference (``full``), an exact-recall oracle-style baseline
(``range_search``), and the source-backed baseline adapters.
"""

from __future__ import annotations

# Dense reference ------------------------------------------------------------
from . import full  # noqa: F401

# Baselines (source repos under baselines/) ----------------------------------
from . import chunkkv  # noqa: F401         (kvpress ChunkKV — static eviction)
from . import hashattention  # noqa: F401   (HashAttention-1.0)
from . import hata  # noqa: F401            (HATA)
from . import range_search  # noqa: F401    (Louver — sampled-threshold halfspace)
from . import selfindexing  # noqa: F401    (selfindexingkv VQ/LUT)
from . import wave_index  # noqa: F401      (RetrievalAttention / RetroInfer)
