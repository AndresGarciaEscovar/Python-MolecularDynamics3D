"""
    File that contains several utility functions to calculate properties of
    molecules.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
from numpy import append as nappend, array, dot, ndarray, sum as nsum

# User defined.
import code.utilities.utilities_diffusion_tensor as udtensor

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Get Functions
# ------------------------------------------------------------------------------


def get_bounding_radius(atoms: list, shift: ndarray = None) -> float:
    """
        From the given set of atoms, in the given coordinate system,
        gets the minimum radius of the sphere that encloses the atom.

        :param atoms: The list with the atoms in the molecule.

        :param shift: A numpy array that represents the shift in the molecule,
         if any.

        :return: The minimum radius of the sphere that encloses the atom.
    """
    return 0.0


def get_cog(coordinates: ndarray, radii: ndarray) -> ndarray:
    """
        From the given set of coordinates and the radius array, gets the center
        of geometry of the molecule.

        :param coordinates: The numpy array with all the coordinates of the
         atom.

        :param radii: The radius of each atom.

        :return: The average of the maximum and minimum coordinates of the
         molecule.
    """

    # Auxiliary variables.
    maxpos = array([], dtype=float)
    minpos = array([], dtype=float)

    # Dimensions.
    length = len(coordinates[0])

    # For every component.
    for i in range(length):
        maxpos = nappend(maxpos, max(coordinates[:, i] + radii))
        minpos = nappend(minpos, min(coordinates[:, i] - radii))

    return (maxpos + minpos) * 0.5


def get_com(coordinates: ndarray, masses: ndarray) -> ndarray:
    """
        From the given set of coordinates and the radius array, gets the center
        of mass of the molecule.

        :param coordinates: The coordinates of the atoms.

        :param masses: The masses of the atoms.

        :return: The mass weighted average of the coordinates.
    """
    return dot(masses, coordinates) / nsum(masses)


def get_dtensor(
    coordinates: ndarray, radii: ndarray, shift: ndarray = None
) -> ndarray:
    """
        From the given set of coordinates and the radius array, gets the center
        of mass of the molecule.

        :param coordinates: The coordinates of the atoms.

        :param radii: The radius of each atom.

        :param shift: The shift in coordinates if the diffusion tensor needs to
         be calculated with respect to a certain reference point.

        :return: The diffusion tensor with respect to the given shift.
    """

    # No need to shift the coordinates.
    if shift is None:
        return udtensor.get_diffusion_tensor(coordinates, radii)

    # Shift all the coordinates before making the calculation.
    acoordinates = array([x + shift for x in coordinates], dtype=float)
    return udtensor.get_diffusion_tensor(acoordinates, radii)


# ------------------------------------------------------------------------------
# Validate Functions
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
