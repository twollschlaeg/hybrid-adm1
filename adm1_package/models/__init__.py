"""The ADM1 variants: the reduced R2, R3 and R4 with their hybrids, and the full ADM1
(``base``). Shared by all: the solve and output (``simulation``), the physical chemistry
(``chemistry``) and the dataset (``dataset``)."""

from . import r2, r3, r4

__all__ = ['r2', 'r3', 'r4']
