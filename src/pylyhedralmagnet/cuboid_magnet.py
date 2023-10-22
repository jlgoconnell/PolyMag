import numpy as np

from .i_magnet import IMagnet


class CuboidMagnet(IMagnet):
    def __init__(
        self,
        length: float,
        width: float,
        height: float,
        centre: np.array = np.array([0, 0, 0]),
        mu_r: float = 1,
    ) -> None:
        l2 = length / 2
        w2 = width / 2
        h2 = height / 2
        self._vertices = np.array(
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
        self._mu_r = mu_r

    def calc_field(self) -> np.array:
        return np.array([0])
