# General magnet class

import numpy as np
import scipy.spatial

from polymag.charged_triangle import ChargedTriangle
from polymag.i_magnet import IMagnet


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
        self.hull: scipy.spatial.ConvexHull = scipy.spatial.ConvexHull(self._vertices)
        self._mu: float = mu
        self._mu_r: float = mu_r

        # Set up triangular surface objects
        self._triangles: list[ChargedTriangle] = [
            ChargedTriangle(
                self.hull.points[self.hull.simplices[ii]],
                self.hull.equations[ii][:3],
                self._mag,
            )
            for ii in range(len(self.hull.simplices))
        ]

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
