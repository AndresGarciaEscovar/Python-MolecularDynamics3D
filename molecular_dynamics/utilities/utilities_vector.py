"""
    File that contains special utilities for vector operations.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import copy
import numpy

import timeit

from numpy import ndarray, float64

# User defined.
import molecular_dynamics.utilities.utilities_molecule as um

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Rotate Functions
# ------------------------------------------------------------------------------


def rotate_about(
        coordinate: ndarray, axis: ndarray, point: ndarray, angle: float64,
) -> ndarray:
    """
        Rotates, counter-clockwise the given coordinate about an axis, with
        respect to the given point, the given amount.
        
        :param coordinate: The coordinate to be rotated.
        
        :param axis: The axis about which the coordinate will be rotated.
        
        :param point: The point with respect to which the coordinate will be
         rotated.
        
        :param angle: The angle, in radians, for the coordinate to be rotated,
         anti-clowise; i.e., following the 'right-hand' rule for cross products.
        
        :param check: Boolean flag that indicates if the arrays must be checked
         for the proper dimensionality. True, if the arrays must be checked for
         the proper dimensionality; False, otherwise.
        
        :return: The rotated vector about the given axis, with respect to the
         given point.
        
        :raise ValueError: If any of the arrays has the wrong number of
         dimensions, and the check flag is set to True.

        :raise TypeError: If any of the arrays is NOT a numpy array or the type
         of values in the array are the wrong type, and the check flag is set to
         True.
    """

    # //////////////////////////////////////////////////////////////////////////
    # Auxiliary Functions
    # //////////////////////////////////////////////////////////////////////////

    def validate_vectors_0() -> None:
        """
            Validates that the vectors have the proper dimensionality.

            :raise ValueError: If any of the arrays has the wrong number of
             dimensions.

            :raise TypeError: If any of the arrays is NOT a numpy array or the
             type of values in the array are the wrong type.
        """

        # Validate the arrays.
        um.validate_array(coordinate, exception=True)
        um.validate_array(axis, exception=True)
        um.validate_array(point, exception=True)

    # //////////////////////////////////////////////////////////////////////////
    # Implementation
    # //////////////////////////////////////////////////////////////////////////

    # Check the proper dimensionality of the vectors.
    validate_vectors_0()

    # Attempting to rotate about the zero vector.
    if numpy.isclose(numpy.linalg.norm(axis), float64(0.0)):
        raise ValueError(
            f"Attempting to rotate about the zero axis: {axis}."
        )

    # Get copies of the vectors involved.
    caxis = copy.deepcopy(axis)
    ccoordinate = copy.deepcopy(coordinate) - point

    # Normalize the axis vector.
    caxis = caxis / numpy.linalg.norm(caxis)

    # Get the rotation cosines.
    cosa = float64(numpy.cos(angle))
    sina = float64(numpy.sin(angle))

    # Get the individual terms.
    term0 = ccoordinate * cosa
    term1 = numpy.cross(caxis, ccoordinate) * sina
    term2 = caxis * (numpy.dot(caxis, ccoordinate)) * (float64(1.0) - cosa)

    # Get the new coordinate
    result = term0 + term1 + term2

    # Translate back.
    result += point

    return result


if __name__ == "__main__":

    crdnt = numpy.array([1, 0, 9], dtype=float64)
    pnt = numpy.array([0., 0, 0], dtype=float64)
    xs = numpy.array([0, 1, 0], dtype=float64)
    ngl = float64(numpy.pi)


    # get the start time
    numbers = 1000_000
    result = timeit.timeit(
        stmt='rotate_about(crdnt, xs, pnt, ngl)', globals=globals(),
        number=numbers
    )

    print(result/numbers)
