"""
    File that contains several utility functions to calculate properties of
    molecules.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import numpy

# User defined.
import code.molecular_dynamics.main.atom as atom

# ##############################################################################
# Functions
# ##############################################################################

# ------------------------------------------------------------------------------
# Get Functions
# ------------------------------------------------------------------------------


def get_center_of_geometry(atoms: list[atom.Atom]) -> numpy.ndarray:
    """
        From the given set of coordinates and the radius array, gets the center
        of geometry of the molecule.

        :param atoms: A list of "atom" objects.

        :return: The average of the maximum and minimum coordinates in the
         system.
    """
    return numpy.array([0.0], dtype=float)


def get_center_of_mass(atoms: list[atom.Atom]) -> numpy.ndarray:
    """
        From the given set of coordinates and the radius array, gets the center
        of mass of the molecule.

        :param atoms: A list of "atom" objects.

        :return: The average of the maximum and minimum coordinates in the
         system.
    """
    return numpy.array([0.0], dtype=float)


# ------------------------------------------------------------------------------
# Validate Functions
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
