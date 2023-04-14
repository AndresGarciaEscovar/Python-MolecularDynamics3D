"""
    File that contains several utility functions to calculate properties of
    molecules.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import numpy as np

# User defined.
import code.main.atom

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Load Functions
# ------------------------------------------------------------------------------


def load(path: str) -> tuple:
    """
        Loads the molecule's atoms from a file.

        :param path: The path from where the molecule will be loaded.

        :return: Returns a list with the atoms loaded and the name of the
         molecule.
    """



# ------------------------------------------------------------------------------
# Get Functions
# ------------------------------------------------------------------------------


def get_center_of_geometry(atoms: list) -> np.ndarray:
    """
        From the given set of coordinates and the radius array, gets the center
        of geometry of the molecule.

        :param atoms: A list of "atom" objects.

        :return: The average of the maximum and minimum coordinates in the
         system.
    """
    return np.array([0.0], dtype=float)


def get_center_of_mass(atoms: list) -> np.ndarray:
    """
        From the given set of coordinates and the radius array, gets the center
        of mass of the molecule.

        :param atoms: A list of "atom" objects.

        :return: The average of the maximum and minimum coordinates in the
         system.
    """
    return np.array([0.0], dtype=float)


def get_bounding_radius(atoms: list, shift: np.ndarray = None) -> float:
    """
        From the given set of atoms, in the given coordinate system,
        gets the minimum radius of the sphere that encloses the atom.

        :param atoms: The list with the atoms in the molecule.

        :param shift: A numpy array that represents the shift in the molecule,
         if any.

        :return: The minimum radius of the sphere that encloses the atom.
    """
    return 0.0


# ------------------------------------------------------------------------------
# Validate Functions
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
