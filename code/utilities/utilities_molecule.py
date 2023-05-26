"""
    File that contains several utility functions to calculate properties of
    molecules.
"""
import itertools

import numpy as np
# ##############################################################################
# Imports
# ##############################################################################

# General.
from numpy import append as nappend, arange, array, cos, dot, identity, ndarray
from numpy import pi, sin, sum as nsum, zeros
from numpy.linalg import inv, norm

from itertools import product

# User defined.
import code.utilities.utilities_diffusion_tensor as udtensor
import code.validation.validation_molecule as vmolecules
import code.validation.validation_parameters as vparameters

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Get Functions
# ------------------------------------------------------------------------------


def get_bounding_radius(
    coordinates: ndarray, radii: ndarray, shift: ndarray = None
) -> float:
    """
        From the given set of atoms, in the given coordinate system,
        gets the minimum radius of the sphere that encloses the atom.

        Pre-condition, variables must come from a molecule, so validation of
        arrays doesn't need to be checked for consistency.

        :param coordinates: The numpy array with all the coordinates of the
         atom.

        :param radii: The numpy array with the radius of each atom.

        :param shift: The shift in coordinates if the bounding radius needs to
         be calculated with respect to a certain reference point.

        :return: The minimum radius of the sphere that encloses the atom.
    """
    # Set the shift.
    sft = zeros((len(coordinates[0]),), dtype=float) if shift is None else shift

    return max(norm(x + sft) + y for x, y in zip(coordinates, radii))


def get_cod(dtensor: ndarray) -> ndarray:
    """
        From the given diffusion tensor and set of coordinates, gets the
        center of diffusion; with respect to the given point.

        Pre-condition, variables must come from a molecule, so validation of
        arrays doesn't need to be checked for consistency.

        :param dtensor: The 6x6 numpy array that represents the diffusion
         tensor.

        :return: The average of the maximum and minimum coordinates of the
         molecule.
    """
    # Check it's a 6x6 tensor.
    vparameters.is_shape_matrix(dtensor, (6, 6))

    drr = dtensor[3:, 3:]
    dtr = dtensor[3:, :3]

    matrix = inv(array(
        [
            [drr[1, 1] + drr[2, 2], -drr[0, 1], -drr[0, 2]],
            [-drr[1, 0], drr[0, 0] + drr[2, 2], -drr[1, 2]],
            [-drr[2, 0], -drr[2, 1], drr[0, 0] + drr[1, 1]]
        ],
        dtype=float
    ))
    vector = array(
        [dtr[1, 2] - dtr[2, 1], dtr[2, 0] - dtr[0, 2], dtr[0, 1] - dtr[1, 0]],
        dtype=float
    )

    return array([dot(vector, y) for y in matrix], dtype=float)


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
    vparameters.is_shape_matrix(radii, (len(coordinates),))

    # No need to shift the coordinates.
    if shift is None:
        return udtensor.get_diffusion_tensor(coordinates, radii)

    # Check that the dimensionality of the shift is valid.
    vparameters.is_shape_matrix(shift, (3,))

    # Shift all the coordinates before making the calculation.
    tcoordinates = array([x + shift for x in coordinates], dtype=float)
    return udtensor.get_diffusion_tensor(tcoordinates, radii)


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
    if "diffusion_tensor" in information:
        dtensor = array(information["diffusion_tensor"], dtype=float)
        vmolecules.is_diffusion_tensor(dimensions, dtensor)

    # Set the orientation.
    orientation = identity(dimensions)
    if "orientation" in information:
        if len(information["orientation"]) == dimensions:
            orientation = array(information["orientation"], dtype=float)

    return dtensor, orientation


def get_long_short_axes(
    coordinates: ndarray, radii: ndarray, step: float = 1e-3
) -> tuple:
    """
        Gets the long and short axis of the molecule in the given coordinate
        system.

        :param coordinates: The coordinates of the atoms of the molecule.

        :param radii: The radii of the atoms, in the same order as the
         coordinates.

        :param step: The incremental step that is used to generate the angles
         up to the complete solid angle.

        :return: The longest and shortes axes of the molecule, with respect to
         the given coordinate system.
    """
    # Check they are numpy arrays.
    vparameters.is_shape_matrix(radii, (len(coordinates),))
    vparameters.is_shape_matrix(
        coordinates, (len(coordinates), len(coordinates[0]))
    )

    # Particular case.
    if (length := len(coordinates[0])) == 1:
        return array([1], dtype=float), array([1], dtype=float)

    # Collection of angles.
    rangles = [arange(0, 2 * pi, step)] if length > 2 else [arange(0, pi, step)]
    for cntr in range(0, length):
        # Only half sphere is needed.
        if cntr == length - 2:
            rangles.append(arange(0, pi / 2, step))
            break

        # Full sphere is needed.
        rangles.append(arange(0, pi, step))

    # Make the projections.
    for value in product(*rangles):
        # Get the values.
        vector = list()

        # Get the normal vector.
        for cntr, angle in enumerate(value):
            if cntr == 0:
                vector = array([cos(angle), sin(angle)], dtype=float)
                continue

            vector = sin(angle) * vector
            vector = nappend(vector, [cos(angle)])


        # TODO: CONTINUE HERE!!!


    print(vector)

    return 1, 2

# ##############################################################################
# TO REMOVE
# ##############################################################################


if __name__ == "__main__":

    import random

    coordi = array([
        [random.uniform(0, 1) for _ in range(2)] for i in range(3)
    ], dtype=float)
    radiei = array([1.0 for i in range(len(coordi))], dtype=float)

    # Get the axes.
    long, short = get_long_short_axes(coordi, radiei)

    print(long, short)
