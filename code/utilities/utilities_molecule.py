"""
    File that contains several utility functions to calculate properties of
    molecules.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
from numpy import append as nappend, array, dot, identity, ndarray, sum as nsum

# User defined.
import code.utilities.utilities_diffusion_tensor as udtensor
import code.validation.validation_parameters as vparameters

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Get Functions
# ------------------------------------------------------------------------------


def get_bounding_radius(coordinates: ndarray, radii: ndarray) -> float:
    """
        From the given set of atoms, in the given coordinate system,
        gets the minimum radius of the sphere that encloses the atom.

        Pre-condition, variables must come from a molecule, so validation of
        arrays doesn't need to be checked for consistency.

        :param coordinates: The numpy array with all the coordinates of the
         atom.

        :param radii: The numpy array with the radius of each atom.

        :return: The minimum radius of the sphere that encloses the atom.
    """

    print(coordinates)

    return 0.0


def get_cog(coordinates: ndarray, radii: ndarray) -> ndarray:
    """
        From the given set of coordinates and the radius array, gets the center
        of geometry of the molecule.

        Pre-condition, variables must come from a molecule, so validation of
        arrays doesn't need to be checked for consistency.

        :param coordinates: The numpy array with all the coordinates of the
         atom.

        :param radii: The numpy array with the radius of each atom.

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

        Pre-condition, variables must come from a molecule, so validation of
        arrays doesn't need to be checked for consistency.

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

    # Check that the dimensionality is valid.
    vparameters.is_shape_matrix(coordinates[0], (3,))

    # No need to shift the coordinates.
    if shift is None:
        return udtensor.get_diffusion_tensor(coordinates, radii)

    # Check that the dimensionality of the shift is valid.
    vparameters.is_shape_matrix(shift, (3,))

    # Shift all the coordinates before making the calculation.
    acoordinates = array([x + shift for x in coordinates], dtype=float)
    return udtensor.get_diffusion_tensor(acoordinates, radii)


def get_dtensor_and_orientation(information: dict, dimensions: int) -> tuple:
    """
        From the given dictionary, gets the diffusion tensor and orientation
        from a dictionary. If a quantity doesn't exist, a None value will be
        returned.

        :param information: The dictionary with the molecule information.

        :param dimensions: The dimensionality of the space, i.e., 3D, 2D, etc.

        :return: A tuple with the values of the diffusion tensor and the
         orientation of the molecule, both quantities with respect to its center
         of mass.

    """
    # Check the diffusion tensor exists.
    dtensor = None
    if "diffusion_tensor" in information and dimensions == 3:
        dtensor = array(information["diffusion_tensor"], dtype=float)

    # Check the orientation exists.
    orientation = identity(dimensions)
    if "orientation" in information:
        orientation = array(information["orientation"], dtype=float)

    return dtensor, orientation
