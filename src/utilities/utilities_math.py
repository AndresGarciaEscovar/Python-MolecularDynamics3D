"""
    File that contains special utilities for vector operations.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import copy
import warnings

from numpy import array, cos, cross, dot, float64, ndarray, sin
from numpy.linalg import norm

# User defined.
import code.validation.validation_parameters as vparameters

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Get Functions
# ------------------------------------------------------------------------------


def get_projection(vector_0: ndarray, vector_1: ndarray) -> ndarray:
    """
        Gets the projection of vector_0 along vector 1.

        :param vector_0: The vector to be projected along vector_1.

        :param vector_1: The vector along which vector_0 will be projected.

        :return: The projection of vector_0 along the **unit** vector defined by
         vector_1.
    """
    # Validate both vectors have the same dimensions.
    vparameters.is_shape_matrix(vector_0, (len(vector_0),))
    vparameters.is_shape_matrix(vector_1, (len(vector_0),))

    warnings.filterwarnings("error")
    projection = dot(vector_0, vector_1) * vector_1 / dot(vector_1, vector_1)
    warnings.filterwarnings("default")

    return projection


def get_skew_symmetric_matrix(vector: ndarray) -> ndarray:
    """
        From the given 3D vector, returns the anti-symmetric matrix with the
        entries shuffled.

        :param vector: A vector with 3 real entries.

        :return: The anti-symmetric matrix with the entries shuffled.
    """

    # Check it's a valid vector.
    vparameters.is_shape_matrix(vector, (3,))

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
    radius_1: float, check: bool = True
) -> bool:
    """
        Returns a boolean flag indicating if two hyper-spheres intersect, given
        their position in space and their hyper-radius, using the Euclidean norm
        as the distance function.

        :param coordinate_0: The coordinates of the zeroth hyper-sphere.

        :param radius_0: The radius of the zeroth hyper-sphere.

        :param coordinate_1: The coordinates of the first hyper-sphere.

        :param radius_1: The radius of the first hyper-sphere.

        :param check: A boolean flag that indicates if checking the vector
         dimensionality is required. True if checking needs to be done; False,
         otherwise. True by default.

        :return: A boolean flag indicating if two hyper-spheres intersect, given
         their position in space and their hyper-radius; i.e., their separation
         is greater than the sum of their radii.
    """
    # Check if needed.
    if check:
        # Both arrays need to have the same dimensions.
        vparameters.is_shape_matrix(coordinate_0, (len(coordinate_0),))
        vparameters.is_shape_matrix(coordinate_1, (len(coordinate_0),))

    # Difference between the two spheres.
    difference = coordinate_0 - coordinate_1
    sradius = (radius_0 + radius_1) ** 2

    return dot(difference, difference) < sradius


# ------------------------------------------------------------------------------
# Rotate Functions
# ------------------------------------------------------------------------------


def rotate_vector(
    vector: ndarray, around: ndarray, angle: float, about: ndarray = None
) -> ndarray:
    """
        Rotates the "vector" around the "around" vector the given "angle" with
        respect to the given "about" point; if no "about" point which to rotate
        the vector is given, it will be assumed it's about (0, 0, 0)
    """

    # Check that the amount is a float.
    vparameters.is_float(angle)

    # Define the about vector.
    about = array([0.0] * 3, dtype=float) if about is None else about

    # Check the types.
    vparameters.is_shape_matrix(about, (3,))
    vparameters.is_shape_matrix(vector, (3,))
    vparameters.is_shape_matrix(around, (3,))

    # Convert into numpy arrays.
    tvector, taround, tabout = vector, around, about

    # Fix vectors.
    tvector -= about
    taround /= norm(taround)

    # Trigonometric values.
    cosa = cos(angle)
    sina = sin(angle)

    # Rotate about the required vector.
    rvector = tvector * cosa
    rvector += cross(taround, tvector) * sina
    rvector += taround * dot(taround, tvector) * (float64(1.0) - cosa)

    return rvector + about


# ------------------------------------------------------------------------------
# Rotate Functions
# ------------------------------------------------------------------------------


def translate_vector(vector: ndarray, translation: ndarray) -> ndarray:
    """
        Translates the "vector" by the "translation" vector.
    """

    # Check that the amounts match.
    vparameters.is_shape_matrix(vector, (len(vector),))
    vparameters.is_shape_matrix(translation, (len(vector),))

    return vector + translation


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
    # Must be a square matrix.
    vparameters.is_shape_matrix(matrix, (len(matrix), len(matrix)))

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
