import numpy as np

from polymag.magnet import Magnet


class CuboidMagnet(Magnet):
    def __init__(
        self,
        length: float,
        width: float,
        height: float,
        centre: np.ndarray,
        magnetisation: np.ndarray,
    ) -> None:
        l2: float = length / 2
        w2: float = width / 2
        h2: float = height / 2
        self._vertices: np.ndarray = np.array(
            [
                [centre[0] - l2, centre[1] - w2, centre[2] - h2],
                [centre[0] - l2, centre[1] - w2, centre[2] + h2],
                [centre[0] - l2, centre[1] + w2, centre[2] - h2],
                [centre[0] - l2, centre[1] + w2, centre[2] + h2],
                [centre[0] + l2, centre[1] - w2, centre[2] - h2],
                [centre[0] + l2, centre[1] - w2, centre[2] + h2],
                [centre[0] + l2, centre[1] + w2, centre[2] - h2],
                [centre[0] + l2, centre[1] + w2, centre[2] + h2],
            ]
        )
        super().__init__(vertices=self._vertices, magnetisation=magnetisation)
