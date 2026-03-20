from dataclasses import dataclass
from typing import Protocol
import numpy as np
import numpy.typing as npt
from collections.abc import Mapping

ArrayF = npt.NDArray[np.float64]
ArrayC = npt.NDArray[np.complex128]


@dataclass(frozen=True)
class RIKey:
    species: str
    n: int
    l: int
    m: int
    center: tuple[float, float, float]


@dataclass(frozen=True)
class BasisSetDimension:
    elements: list[str]
    l_max: dict[str, int]
    n_max: dict[tuple[str, int], int]

    def size(self) -> int:
        total = 0
        for element in self.elements:
            for l in range(self.l_max[element] + 1):
                for n in range(self.n_max[(element, l)] + 1):
                    total += 2 * l + 1
        return total


class RadialFunction(Protocol):
    def __call__(self, r: float | ArrayF) -> float | ArrayF:
        ...


class AngularFunction(Protocol):
    def __call__(self, theta: float | ArrayF, phi: float | ArrayF) -> complex | ArrayC:
        ...


class RIBasisFunction(Protocol):
    def __call__(self, r: ArrayF) -> complex | ArrayC:
        ...

    def get_key(self) -> RIKey:
        ...

    def get_radial(self) -> RadialFunction:
        ...

    def get_angular(self) -> AngularFunction:
        ...


class RIBasis(Mapping[RIKey, RIBasisFunction], Protocol):
    pass