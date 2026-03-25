from typing import Protocol, Union, Iterator
import numpy as np
import numpy.typing as npt

ArrayF = npt.NDArray[np.float64]
ScalarOrArray = Union[float, ArrayF]
