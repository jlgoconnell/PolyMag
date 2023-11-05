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

    def _subdivide_triangle_recursion(
        self, nodes: list[MagnetNode], n: int, normal: np.array, mag: np.array
    ) -> None:
        if n >= 2:
            node_br = nodes[1]
            node_tr = MagnetNode(
                position=np.array(
                    nodes[1].position + 1 / n * (nodes[0].position - nodes[1].position)
                )
            )
            self._nodes.append(node_tr)
            temp_node_idx = len(self._nodes) - 1
            for ii in range(n - 1):
                node_tl = node_tr
                node_bl = node_br
                node_br = MagnetNode(
                    position=np.array(
                        node_bl.position
                        + 1 / n * (nodes[2].position - nodes[1].position)
                    )
                )
                node_tr = MagnetNode(
                    position=np.array(
                        node_tl.position
                        + 1 / n * (nodes[2].position - nodes[1].position)
                    )
                )
                self._nodes.append(node_br)
                self._nodes.append(node_tr)
                self._triangles.append(
                    ChargedTriangle(
                        nodes=[node_bl, node_tl, node_br],
                        normal=normal,
                        magnetisation=mag,
                    )
                )
                self._triangles.append(
                    ChargedTriangle(
                        nodes=[node_tl, node_br, node_tr],
                        normal=normal,
                        magnetisation=mag,
                    )
                )
            # Add the last triangle
            self._triangles.append(
                ChargedTriangle(
                    [self._nodes[-1], self._nodes[-2], nodes[2]], normal, mag
                )
            )
            new_nodes = [nodes[0], self._nodes[temp_node_idx], self._nodes[-1]]
            self._subdivide_triangle_recursion(new_nodes, n - 1, normal, mag)
        else:
            self._triangles.append(
                ChargedTriangle(nodes=nodes, normal=normal, magnetisation=mag)
            )

    def subdivide_triangle(self, triangle: ChargedTriangle, n: int) -> None:
        # Recursively subdivide a triangle
        nodes = triangle.nodes()
        normal = triangle.normal()
        mag = triangle.magnetisation()
        self._subdivide_triangle_recursion(nodes=nodes, n=n, normal=normal, mag=mag)
        self._triangles.remove(triangle)

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
