import numpy as np

from polymag.charged_triangle import ChargedTriangle


def test_singularities():
    # Create a triangle
    tri = ChargedTriangle(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
        normal=np.array([0, 0, 1]),
        magnetisation=np.array([0, 0, 1]),
    )

    fld = tri.calc_field(
        np.array(
            [
                [1.1, 0, 0],  # Should have 0 z-field, positive x, negative y
                [1, 1.1, 0],  # Should have 0 z-field, positive x and y
                [1.1, 1.1, 0],  # Should have 0 z-field, positive x and y
            ]
        )
    )

    # First make sure no entries are infinite or NaN
    assert not np.isinf(fld).any()
    assert not np.isnan(fld).any()

    # Check all z-fields are zero
    assert (fld[:, 2] == np.zeros(np.shape(fld[:, 2]))).all()

    # Check signs of x- and y-components
    expected_x_signs = [1, 1, 1]
    expected_y_signs = [-1, 1, 1]
    assert (np.sign(fld[:, 0]) == np.array(expected_x_signs)).all()
    assert (np.sign(fld[:, 1]) == np.array(expected_y_signs)).all()
