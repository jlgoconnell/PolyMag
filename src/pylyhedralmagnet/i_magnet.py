from abc import ABC, abstractmethod

import numpy as np


class IMagnet(ABC):
    # Magnet abstract base class for various magnet types

    @abstractmethod
    def __init__(
        self,
    ) -> None:
        pass

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def mu_r(self) -> float:
        return self._mu_r

    @abstractmethod
    def calc_field(self) -> np.ndarray:
        pass

    @property
    def vertices(self) -> np.array:
        return self._vertices
