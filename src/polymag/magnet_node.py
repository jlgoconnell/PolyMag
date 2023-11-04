import numpy as np


class MagnetNode:
    # Class for magnet nodes used to more accurately
    # estimate forces, permeability effects, etc
    def __init__(self, position: np.ndarray) -> None:
        self.position = position
        self.field = np.array([0, 0, 0])

    def reset_field(self) -> None:
        self.field = np.array([0, 0, 0])

    def add_to_field(self, field_addition: np.array) -> None:
        self.field = self.field + field_addition

    def get_field(self) -> np.array:
        return self.field

    def translate(self, translation: np.array) -> None:
        pass
