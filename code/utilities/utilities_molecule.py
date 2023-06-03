"""
    File that contains several utility functions to calculate properties of
    molecules.
"""
import itertools
import sys

import numpy as np
# ##############################################################################
# Imports
# ##############################################################################

# General.
from copy import deepcopy

from numpy import append as nappend, arange, array, cos, dot, identity, inf
from numpy import max as nmax, min as nmin, ndarray, pi, sin, sum as nsum
from numpy import vstack, zeros
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
    # //////////////////////////////////////////////////////////////////////////
    # Auxiliary Functions
    # //////////////////////////////////////////////////////////////////////////

    def get_angle_cosines(dimensions: int) -> tuple:
        """
            Gets the directional cosines of the angles to inspect.

            :param dimensions: The number of dimensions.

            :return: The angles to be checked.
        """
        # Auxiliary variables.
        cosines = [cosines_array(pi)] * (dimensions - 1)

        # No need to do more.
        if dimensions == 2:
            return tuple(cosines)

        # Set the first and last index.
        cosines[0] = cosines_array(2 * pi)
        cosines[-1] = cosines_array(pi / 2)

        return tuple(cosines)

    def cosines_array(angle: float) -> ndarray:
        """
            Gets the directive cosines from zero to the given angle, in radians.

            :param angle: Upper up to where to get the cosines.

            :return: The array with the directive cosines.
        """
        base = nappend(arange(0.0, angle, step), angle)
        return vstack((cos(base), sin(base))).T

    # //////////////////////////////////////////////////////////////////////////
    # Implementation
    # //////////////////////////////////////////////////////////////////////////

    import math

    # Check they are numpy arrays.
    vparameters.is_shape_matrix(radii, (len(coordinates),))
    vparameters.is_shape_matrix(
        coordinates, (len(coordinates), len(coordinates[0]))
    )

    # Particular case.
    if (dims := len(coordinates[0])) == 1:
        return array([1], dtype=float), array([1], dtype=float)

    # Collection of angles.
    rcosines = get_angle_cosines(dims)
    vector = zeros((dims,), dtype=float)

    # Reset the variables.
    long, shor = -inf, inf
    along, ashor = None, None

    length = math.prod(map(len, rcosines))

    # For all possible combinations.
    print(f"\rPercentage Done: {0:.7f} %", end="")
    for cntr, rcosine in enumerate(product(*rcosines)):
        # Get the unit vector.
        for i, dcosine in enumerate(rcosine, start=1):
            # First one is always the 2D directive cosines.
            if i == 1:
                vector[:i + 1] = dcosine
                continue

            # Multiply by the previous sine.
            vector[:i] = dcosine[1] * vector[:i]
            vector[i] = dcosine[0]

        # Get the projection and length of axis.
        proj = dot(coordinates, vector)
        distance = nmax(proj + radii) - nmin(proj - radii)

        # Evaluate long distance.
        if long < distance:
            long = deepcopy(distance)
            along = deepcopy(vector)

        # Evaluate short distance.
        if shor > distance:
            shor = deepcopy(distance)
            ashor = deepcopy(vector)

        print(f"\rPercentage Done: {cntr * 100 / length:.7f} %", end="")

    print(f"\rPercentage Done: {100:.7f} %", end="")

    return (along, long), (ashor, shor)


# ##############################################################################
# TO REMOVE
# ##############################################################################


if __name__ == "__main__":
    from datetime import datetime

    coordeei = array(
        [
            [1, 1, i] for i in range(0, 26)
        ],
        dtype=float
    )
    coords = len(coordeei)
    dimes = len(coordeei[0])
    radiei = array([1 for _ in range(coords)], dtype=float)

    print(f"Dimensionality: {dimes}")
    print(f"Number of coordinates: {coords}", end="\n\n")

    # Get the axes.
    start = datetime.now()
    longsss, shorsss = get_long_short_axes(coordeei, radiei, step=1e-4)
    end = datetime.now()

    print(f"elapsed time: {(end - start).total_seconds()} seconds")
    print(longsss, norm(longsss[0]))
    print(shorsss, norm(shorsss[0]))
