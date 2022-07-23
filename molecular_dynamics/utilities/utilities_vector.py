"""
    File that contains special utilities for vector operations.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import copy
import numpy

from numpy import ndarray, float64

# User defined.
import molecular_dynamics.utilities.utilities_molecule as um

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Orthogonal Functions
# ------------------------------------------------------------------------------


def orthogonalize(basis: ndarray, normal: bool = False) -> ndarray:
    """
        Orthogonalizes the given basis.

        :param basis: The vectors to be orthogonalized.

        :param normal: Boolean flag that indicates if the resultant vectors
         should be normalized. True, if the resultant vectors should be
         normalized; False, otherwise. False, by default.

        :return: The new orthogonalized basis.
    """

    # Stores the new basis.
    new_basis = []

    for i, v0 in enumerate(basis):
        # Use the first vector as the basis vector.
        if i == 0:
            # Normalize.
            v0 /= numpy.linalg.norm(v0) if normal else float64(1.0)
            new_basis.append(v0)
            continue

        # Get the next vector.
        v2 = copy.deepcopy(v0)
        new_v = copy.deepcopy(v0)

        # Orthogonalize.
        for j, v1 in enumerate(new_basis):
            norm = numpy.linalg.norm(v1) * numpy.linalg.norm(v2)
            new_v -= v1 * numpy.dot(v2, v1) / norm

        # Normalize.
        new_v /= numpy.linalg.norm(new_v) if normal else float64(1.0)

        # Add the new vector to the basis.
        new_basis.append(new_v)

    return numpy.array(new_basis, dtype=float64)


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
    pass
