import numpy as np


class ChargedTriangle:
    def __init__(
        self, vertices: np.ndarray, normal: np.ndarray, magnetisation: np.ndarray
    ) -> None:
        self.normal: np.ndarray = normal
        self.vertices: np.ndarray = vertices
        self.charge: float = np.dot(magnetisation, self.normal)

        # Set up the desired local coordinate system
        local_y: np.ndarray = np.array(self.vertices[1] - self.vertices[0])
        local_y = local_y / np.sqrt(np.sum(local_y**2))
        local_x: np.ndarray = np.array(self.vertices[2] - self.vertices[0])
        local_z: np.ndarray = np.cross(local_x, local_y)
        local_z = local_z / np.sqrt(np.sum(local_z**2))
        local_x = np.cross(local_y, local_z)
        local_x = local_x / np.sqrt(np.sum(local_x**2))
        # The rotation matrix required to convert between the local and global frame
        self.R: np.ndarray = np.array([local_x, local_y, local_z])

        # Calculate necessary parameters in local frame
        pts: np.ndarray = self.vertices @ self.R.T
        self.x1: float = pts[0, 0]
        self.x3: float = pts[2, 0]
        self.y1: float = pts[0, 1]
        self.y2: float = pts[1, 1]
        self.y3: float = pts[2, 1]
        self.z: float = pts[2, 2]

    def calc_field(self, points: np.ndarray) -> np.ndarray:
        # Rounding number to account for numeric precision
        rnd_num: int = 12

        # If the surface charge is zero, return nothing
        if np.round(self.charge, rnd_num) == 0:
            return np.array(np.zeros(np.shape(points)))

        # Rotate points to local coordinate system
        local_points: np.ndarray = np.array(points @ self.R.T)

        # All the cheeky calculations
        Xp1: np.ndarray = local_points[:, 0] - self.x1
        Xp2: np.ndarray = local_points[:, 0] - self.x3
        Y11: np.ndarray = local_points[:, 1] - self.y1
        Y21: np.ndarray = local_points[:, 1] - self.y2
        Yp2: np.ndarray = local_points[:, 1] - self.y3
        Z: np.ndarray = local_points[:, 2] - self.z

        m1: float = (self.y3 - self.y1) / (self.x3 - self.x1)
        m2: float = (self.y3 - self.y2) / (self.x3 - self.x1)

        Zsq: np.ndarray = np.power(Z, 2)

        R11: np.ndarray = np.sqrt(np.power(Xp1, 2) + np.power(Y11, 2) + Zsq)
        R21: np.ndarray = np.sqrt(np.power(Xp1, 2) + np.power(Y21, 2) + Zsq)
        Rp2: np.ndarray = np.sqrt(np.power(Xp2, 2) + np.power(Yp2, 2) + Zsq)

        Xp1sqpZsq: np.ndarray = np.round(np.power(Xp1, 2) + Zsq, rnd_num)
        Xp2sqpZsq: np.ndarray = np.round(np.power(Xp2, 2) + Zsq, rnd_num)

        R11m1Y: np.ndarray = np.round(
            np.sqrt(1 + m1**2) * R11 - (m1 * Y11 + Xp1), rnd_num
        )
        R12m1Y: np.ndarray = np.round(
            np.sqrt(1 + m1**2) * Rp2 - (m1 * Yp2 + Xp2), rnd_num
        )
        R21m2Y: np.ndarray = np.round(
            np.sqrt(1 + m2**2) * R21 - (m2 * Y21 + Xp1), rnd_num
        )
        R22m2Y: np.ndarray = np.round(
            np.sqrt(1 + m2**2) * Rp2 - (m2 * Yp2 + Xp2), rnd_num
        )
        S11: np.ndarray = Rp2 + np.sign(R11m1Y) * (R11m1Y - Rp2)
        S12: np.ndarray = R11 + np.sign(R12m1Y) * (R12m1Y - R11)
        S21: np.ndarray = Rp2 + np.sign(R21m2Y) * (R21m2Y - Rp2)
        S22: np.ndarray = R21 + np.sign(R22m2Y) * (R22m2Y - R21)

        R11Y11: np.ndarray = R11 - Y11
        R21Y21: np.ndarray = R21 - Y21
        T11: np.ndarray = R21 + np.sign(R11Y11) * (R11Y11 - R21)
        T21: np.ndarray = R11 + np.sign(R21Y21) * (R21Y21 - R11)

        N11: np.ndarray = m1 * Xp1sqpZsq - Xp1 * Y11
        N21: np.ndarray = m2 * Xp1sqpZsq - Xp1 * Y21
        N12: np.ndarray = m1 * Xp2sqpZsq - Xp2 * Yp2
        N22: np.ndarray = m2 * Xp2sqpZsq - Xp2 * Yp2
        D11: np.ndarray = R11 * abs(Z)
        D21: np.ndarray = R21 * abs(Z)
        Dp2: np.ndarray = Rp2 * abs(Z)

        # Final field in local coordinate frame
        b_x: np.ndarray = (
            np.log(T11 / T21)
            - m1 / np.sqrt(1 + m1**2) * np.log(S11 / S12)
            + m2 / np.sqrt(1 + m2**2) * np.log(S21 / S22)
        )
        b_y: np.ndarray = 1 / np.sqrt(1 + m1**2) * np.log(S11 / S12) - 1 / np.sqrt(
            1 + m2**2
        ) * np.log(S21 / S22)
        b_z: np.ndarray = -np.sign(Z) * (
            np.arctan2(N11, D11)
            - np.arctan2(N21, D21)
            - np.arctan2(N12, Dp2)
            + np.arctan2(N22, Dp2)
        )

        B: np.ndarray = np.array([b_x, b_y, b_z]).T @ self.R * self.charge / (4 * np.pi)

        return B
