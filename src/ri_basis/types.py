from enum import Enum
from typing import Union

class CutoffType(str, Enum):
    ESTIMATE = "estimate"
    NON_PERIODIC = "non-periodic"

Cutoff = Union[float, CutoffType]

