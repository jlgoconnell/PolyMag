from abc import ABC, abstractmethod

import numpy as np


class IMagnet(ABC):
    # Magnet abstract base class for various magnet types

    @abstractmethod
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def calc_field(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def translate(self, displacement: np.ndarray) -> None:
        pass

    @abstractmethod
    def rotate(self, point: np.ndarray, rotation_matrix: np.ndarray) -> None:
        pass
