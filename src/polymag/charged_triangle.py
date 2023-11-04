import numpy as np

from polymag.magnet_node import MagnetNode


class ChargedTriangle:
    def __init__(
        self, nodes: list[MagnetNode], normal: np.array, magnetisation: np.array
    ) -> None:
        self._nodes = nodes
        self._normal = normal
        self._mag = magnetisation
        self._charge = np.dot(self._mag, self._normal)

    def calculate_rot_matrix(self) -> np.ndarray:
        # Set up the desired local coordinate system
        local_y = np.array(self._nodes[1].position - self._nodes[0].position)
        local_y = local_y / np.sqrt(np.sum(local_y**2))
        local_x = np.array(self._nodes[2].position - self._nodes[0].position)
        local_z = np.cross(local_x, local_y)
        local_z = local_z / np.sqrt(np.sum(local_z**2))
        local_x = np.cross(local_y, local_z)
        local_x = local_x / np.sqrt(np.sum(local_x**2))

        # The rotation matrix required to convert between the local and global frame
        R = np.array([local_x, local_y, local_z])

        return R

    @property
    def charge(self) -> float:
        return self._charge

    def calc_field(self, points: np.ndarray) -> np.ndarray:
        # Rounding number to account for numeric precision
        rnd_num = 9

        # If the surface charge is zero, return nothing
        if np.round(self._charge, rnd_num) == 0:
            return np.array(np.zeros(np.shape(points)))

        # Calculate rotation matrix
        R = self.calculate_rot_matrix()

        # Calculate local vertices
        pts = np.vstack([node.position @ R.T for node in self._nodes])
        self.x1 = pts[0, 0]
        self.x3 = pts[2, 0]
        self.y1 = pts[0, 1]
        self.y2 = pts[1, 1]
        self.y3 = pts[2, 1]
        self.z = pts[2, 2]

        # Rotate points to local coordinate system
        local_points = np.array(points @ R.T)

        # All the cheeky calculations
        Xp1 = local_points[:, 0] - self.x1
        Xp2 = local_points[:, 0] - self.x3
        Y11 = local_points[:, 1] - self.y1
        Y21 = local_points[:, 1] - self.y2
        Yp2 = local_points[:, 1] - self.y3
        Z = local_points[:, 2] - self.z

        m1 = (self.y3 - self.y1) / (self.x3 - self.x1)
        m2 = (self.y3 - self.y2) / (self.x3 - self.x1)

        Zsq = np.power(Z, 2)

        R11 = np.sqrt(np.power(Xp1, 2) + np.power(Y11, 2) + Zsq)
        R21 = np.sqrt(np.power(Xp1, 2) + np.power(Y21, 2) + Zsq)
        Rp2 = np.sqrt(np.power(Xp2, 2) + np.power(Yp2, 2) + Zsq)

        Xp1sqpZsq = np.round(np.power(Xp1, 2) + Zsq, rnd_num)
        Xp2sqpZsq = np.round(np.power(Xp2, 2) + Zsq, rnd_num)

        R11m1Y = np.round(np.sqrt(1 + m1**2) * R11 - (m1 * Y11 + Xp1), rnd_num)
        R12m1Y = np.round(np.sqrt(1 + m1**2) * Rp2 - (m1 * Yp2 + Xp2), rnd_num)
        R21m2Y = np.round(np.sqrt(1 + m2**2) * R21 - (m2 * Y21 + Xp1), rnd_num)
        R22m2Y = np.round(np.sqrt(1 + m2**2) * Rp2 - (m2 * Yp2 + Xp2), rnd_num)
        S11 = Rp2 + np.sign(R11m1Y) * (R11m1Y - Rp2)
        S12 = R11 + np.sign(R12m1Y) * (R12m1Y - R11)
        S21 = Rp2 + np.sign(R21m2Y) * (R21m2Y - Rp2)
        S22 = R21 + np.sign(R22m2Y) * (R22m2Y - R21)

        R11Y11 = R11 - Y11
        R21Y21 = R21 - Y21
        T11 = R21 + np.sign(R11Y11) * (R11Y11 - R21)
        T21 = R11 + np.sign(R21Y21) * (R21Y21 - R11)

        N11 = m1 * Xp1sqpZsq - Xp1 * Y11
        N21 = m2 * Xp1sqpZsq - Xp1 * Y21
        N12 = m1 * Xp2sqpZsq - Xp2 * Yp2
        N22 = m2 * Xp2sqpZsq - Xp2 * Yp2
        D11 = R11 * abs(Z)
        D21 = R21 * abs(Z)
        Dp2 = Rp2 * abs(Z)

        # Final field in local coordinate frame
        b_x = (
            np.log(T11 / T21)
            - m1 / np.sqrt(1 + m1**2) * np.log(S11 / S12)
            + m2 / np.sqrt(1 + m2**2) * np.log(S21 / S22)
        )
        b_y = 1 / np.sqrt(1 + m1**2) * np.log(S11 / S12) - 1 / np.sqrt(
            1 + m2**2
        ) * np.log(S21 / S22)
        b_z = -np.sign(Z) * (
            np.arctan2(N11, D11)
            - np.arctan2(N21, D21)
            - np.arctan2(N12, Dp2)
            + np.arctan2(N22, Dp2)
        )

        B = np.round(
            np.array([b_x, b_y, b_z]).T @ R * self._charge / (4 * np.pi), rnd_num
        )

        return B
