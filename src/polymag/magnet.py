# General magnet class

import numpy as np
import scipy.spatial

from polymag.charged_triangle import ChargedTriangle
from polymag.i_magnet import IMagnet
from polymag.magnet_node import MagnetNode


class Magnet(IMagnet):
    def __init__(
        self,
        vertices: np.ndarray,
        magnetisation: np.ndarray = np.array([0, 0, 0]),
        mu: float = 4 * np.pi * 10 ** (-7),
        mu_r: float = 1,
    ) -> None:
        self._vertices: np.ndarray = vertices
        self._mag: np.ndarray = magnetisation
        self._mu: float = mu
        self._mu_r: float = mu_r

        # Create convex hull
        self._hull: scipy.spatial.ConvexHull = scipy.spatial.ConvexHull(self._vertices)

        # Set up initial magnet nodes
        self._nodes: list[MagnetNode] = [
            MagnetNode(position=pos) for pos in self._hull.points
        ]

        # Set up triangular surfaces
        self._triangles: list[ChargedTriangle] = [
            ChargedTriangle(
                [self._nodes[index] for index in self._hull.simplices[ii]],
                self._hull.equations[ii][:3],
                self._mag,
            )
            for ii in range(len(self._hull.simplices))
        ]
        pass

    def inside_magnet(self, point: np.ndarray) -> list[bool]:
        # ToDo: Check if any points are inside the magnet
        return [True]

    def calc_field(self, points: np.ndarray) -> np.ndarray:
        # Error handling:
        if len(np.shape(points)) == 1:
            points = np.array([points])
        elif len(np.shape(points)) != 2:
            raise ValueError("Input points have incorrect dimensions")

        B: np.ndarray = np.array(np.zeros(np.shape(points)))

        for triangle in self._triangles:
            if triangle.charge != 0.0:
                B_tri: np.ndarray = triangle.calc_field(points)
                B += B_tri

        return B

    def translate(self, translation: np.ndarray) -> None:
        pass

    def rotate(self, point: np.ndarray, rotation_matrix: np.ndarray):
        pass
