import numpy as np

from pylyhedralmagnet.cuboid_magnet import CuboidMagnet


def test_cuboid_magnet():
    mag = CuboidMagnet(length=1, width=1, height=1)

    assert (
        mag.vertices
        == np.array(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ]
        )
    ).all()


if __name__ == "__main__":
    test_cuboid_magnet()
