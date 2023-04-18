"""
    File that contains special utilities for vector operations.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import copy

from numpy import array, dot, ndarray

# User defined.
import code.validation.validation_parameters as vparameters

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Get Functions
# ------------------------------------------------------------------------------


def get_skew_symmetric_matrix(vector: ndarray) -> ndarray:
    """
        From the given 3D vector, returns the anti-symmetric matrix with the
        entries shuffled.

        :param vector: A vector with 3 real entries.

        :return: The anti-symmetric matrix with the entries shuffled.
    """

    # Check it's a valid vector.
    vparameters.is_shape_matrix(vector, (0, 3))

    return array(
        (
            (0.0, -vector[2], vector[1]),
            (vector[2], 0.0, -vector[0]),
            (-vector[1], vector[0], 0.0)
        ), dtype=float
    )


# ------------------------------------------------------------------------------
# Intersect Functions
# ------------------------------------------------------------------------------


def intersect_hspheres(
    coordinate_0: ndarray, radius_0: float, coordinate_1: ndarray,
    radius_1: float
) -> bool:
    """
        Returns a boolean flag indicating if two hyper-spheres intersect, given
        their position in space and their hyper-radius, using the Euclidean norm
        as the distance function.

        :param coordinate_0: The coordinates of the zeroth hyper-sphere.

        :param radius_0: The radius of the zeroth hyper-sphere.

        :param coordinate_1: The coordinates of the first hyper-sphere.

        :param radius_1: The radius of the first hyper-sphere.

        :return: A boolean flag indicating if two hyper-spheres intersect, given
         their position in space and their hyper-radius; i.e., their separation
         is greater than the sum of their radii.
    """
    # Difference between the two spheres.
    difference = coordinate_0 - coordinate_1
    sradius = (radius_0 + radius_1) ** 2

    return dot(difference, difference) < sradius


# ------------------------------------------------------------------------------
# Symmetrize Functions
# ------------------------------------------------------------------------------


def symmetrize(matrix: ndarray, passes: int = 0):
    """
        Symmetrizes the given square matrix.

        :param matrix: The matrix to be symmetrized.

        :param passes: The number of passes that are used to symmetrize before
         getting the output.

        :return: The numpy array with the matrix symmetrized.
    """

    # Only integer numbers.
    cpasses = int(passes)

    # Get a copy of the original matrix.
    cmatrix = copy.deepcopy(matrix)
    length = len(cmatrix)

    # Don't symmetrize.
    if cpasses <= 0:
        return cmatrix

    # Symmetrize the elements, the given number of times.
    for k in range(cpasses - 1):
        for i in range(length):
            for j in range(i + 1, length):
                cmatrix[i, j] = (cmatrix[i, j] + cmatrix[j, i]) * 0.5
                cmatrix[j, i] = (cmatrix[i, j] + cmatrix[j, i]) * 0.5

    # Assign the relevant values.
    for i in range(length):
        for j in range(i + 1, length):
            cmatrix[i, j] = (cmatrix[i, j] + cmatrix[j, i]) * 0.5
            cmatrix[j, i] = cmatrix[i, j]

    return cmatrix
