# A class describing a system of magnetic objects

import numpy as np

from polymag.i_magnet import IMagnet


class MagnetSystem:
    def __init__(self) -> None:
        self.magnets: list[IMagnet] = []

    def add_magnet(self, magnet: IMagnet) -> None:
        self.magnets.append(magnet)

    def calc_field_at(self, points: np.array) -> np.array:
        field = np.zeros(np.shape(points))
        for magnet in self.magnets:
            field = field + magnet.calc_field(points)

        return field

    def calc_forces(self) -> np.array:
        # I reckon each magnet node needs
        # to store the field at itself
        pass
