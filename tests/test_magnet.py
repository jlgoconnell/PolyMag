import numpy as np

from polymag.charged_triangle import ChargedTriangle
from polymag.cuboid_magnet import CuboidMagnet
from polymag.magnet import Magnet


def test_frustum_magnet_field():
    # Create a frustum magnet
    mag = Magnet(
        vertices=np.array(
            [
                [0.01, 0.01, 0],
                [0.01, -0.01, 0],
                [-0.01, 0.01, 0],
                [-0.01, -0.01, 0],
                [0.015, 0.015, -0.02],
                [0.015, -0.015, -0.02],
                [-0.015, 0.015, -0.02],
                [-0.015, -0.015, -0.02],
            ]
        ),
        magnetisation=np.array([0, 0, 1.3]),
    )

    x_pts = np.linspace(0, 0.02, 201)
    y_pts = np.zeros(np.shape(x_pts))
    z_pts = 0.001 * np.ones(np.shape(x_pts))
    pts = np.array([x_pts, y_pts, z_pts]).T
    fld = mag.calc_field(pts)
    field_strengths = np.linalg.norm(fld, axis=1)

    assert np.abs(np.max(field_strengths) - 0.633) < 0.001  # Result from my PhD
