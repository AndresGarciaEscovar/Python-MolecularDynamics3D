"""
    File that contains the functions to calculate the diffusion tensor of a
    3D molecule whose atoms are solid spheres. Based on the algorithm in:

    B. Carraso and J. Garcia de la Torre. Hydrodynamic Properties of Rigid
    Particles: Comparison of Different Modeling and Computational Procedures.
    Biophysical Journal, V75: 3044-3057, June 1999.
"""


# ##############################################################################
# Imports
# ##############################################################################

# General.
import copy

from numpy import array, dot, identity, matmul, ndarray, outer, pi, transpose
from numpy import zeros

from numpy.linalg import norm, inv

# User defined.
import code.utilities.utilities_math as umath


# ##############################################################################
# Auxiliary Functions
# ##############################################################################

# ------------------------------------------------------------------------------
# Get Functions
# ------------------------------------------------------------------------------


def get_btensor(coordinates: ndarray, radii: ndarray) -> ndarray:
    """
        Gets the big tensor with the elements to generate the friction
        tensor.

        :param coordinates: The coordinates of the particles.

        :param radii: The radii of the spherical particles.

        :return: The big tensor to get the friction tensor.
    """

    # Create the big matrix and identity matrix.
    dimensions = 3 * len(coordinates)
    matrix = zeros((dimensions, dimensions), dtype=float)

    # Loop through the pairs.
    for i, (coordinate_0, radius_0) in enumerate(zip(coordinates, radii)):
        for j, (coordinate_1, radius_1) in enumerate(zip(coordinates, radii)):

            # Determine if the particles overlap.
            overlap = umath.intersect_hspheres(
                coordinate_0, radius_0, coordinate_1, radius_1
            )

            # Set the matrix entries.
            xpos, ypos = i * 3, j * 3

            # Self interaction.
            if i == j:
                tensor = identity(3) / (6.0 * pi * radius_0)
                matrix[xpos: xpos + 3, ypos: ypos + 3] = tensor
                continue

            # Radii are equal and spheres overlap.
            if radius_0 == radius_1 and overlap:
                matrix[xpos: xpos + 3, ypos: ypos + 3] = get_tensor_requal(
                    coordinate_0, coordinate_1, radius_0
                )
                continue

            # No sphere overlap (or spheres overlap and radii are different).
            matrix[xpos: xpos + 3, ypos: ypos + 3] = get_tensor_runequal(
                    coordinate_0, radius_0, coordinate_1, radius_1
            )

    return matrix


def get_tensor_requal(
    coordinate_0: ndarray, coordinate_1: ndarray, radius: float
) -> ndarray:
    """
        Gets the 3x3 friction tensor when the radii of the spheres are
        the same.

        :param coordinate_0: The coordinate of the zeroth particle.

        :param coordinate_1: The coordinate of the firts particle.

        :param radius: The radius of either particle; since the radius of the
         each particle are the same.

        :return: The 3x3 matrix with the friction tensor.
    """

    # Base tensors.
    difference = coordinate_0 - coordinate_1
    oproduct = outer(difference, difference)

    # Distance between the two points.
    distance = norm(difference)

    # Calculate the tensor.
    tensor = (1.0 - distance * 9.0 / (radius * 32.0)) * identity(3)
    tensor += (3.0 / (32.0 * distance * radius)) * oproduct

    return tensor / (6.0 * pi * radius)


def get_tensor_runequal(
    coordinate_0: ndarray, radius_0: float, coordinate_1: ndarray,
    radius_1: float,
) -> ndarray:
    """
        Gets the 3x3 friction tensor when the radii of the spheres are
        not the same.

        :param coordinate_0: The coordinate of the zeroth particle.

        :param radius_0: The radius of zeroth particle.

        :param coordinate_1: The coordinate of the firts particle.

        :param radius_1: The radius of first particle.

        :return: The 3x3 matrix with the friction tensor; when the radius of
        each particle is different.
    """

    # Base tensors.
    difference = coordinate_0 - coordinate_1
    oproduct = outer(difference, difference)

    # Distance between the two points.
    dnorm = norm(difference)
    norms = dot(difference, difference)

    # Sum of square of the radii.
    sradiuss = (radius_0 ** 2) + (radius_1 ** 2)

    # Calculate the tensor.
    tensor = identity(3)
    tensor += (oproduct / norms)
    tensor += ((identity(3) / 3.0) - (oproduct / norms)) * (sradiuss / norms)
    tensor /= (8.0 * pi * dnorm)

    return tensor


# ------------------------------------------------------------------------------
# Validation Functions
# ------------------------------------------------------------------------------


def validate_dimensionality(coordinates: ndarray) -> None:
    """
        Validates the coordinates are 3D before continuing.

        :param coordinates: The coordinates whose dimensionality is to be
         validated.

        :raise ValueError: If the coordinates are not 3D.
    """

    # Validate the dimensionality.
    if len(coordinates) == 3:
        return

    # Dimensionality not valid.
    raise ValueError(
        "The dimensionality of the coordinates must be 3 for the diffusion "
        "tensor to be calculated."
    )


# ##############################################################################
# Main Functions
# ##############################################################################


def get_diffusion_tensor(coordinates: ndarray, radii: ndarray) -> ndarray:
    """
        Get the diffusion tensor with tt-tr-rr coupling; including the volume
        correction.

        :param coordinates: The coordinates of each atom in the molecule.

        :param radii: The radii of each atom in the molecule.

        :return: A 6x6 np array whose entries are the diffusion tensor.
    """

    # Validate the coordinate dimensionality.
    validate_dimensionality(coordinates[0])

    # Get the big matrix.
    matrix = get_btensor(coordinates, radii)

    # Invert the matrix, and symmetrize, to get the BIG C matrix.
    matrix = inv(matrix)
    matrix = umath.symmetrize(matrix, passes=2)

    #
    # # Get the different coupling tensors.
    # tt = DiffusionTensor._get_tensor_tt(matrix, coordinates)
    # tr = DiffusionTensor._get_tensor_tr(matrix, coordinates)
    # rr = DiffusionTensor._get_tensor_rr(matrix, coordinates)
    #
    # # Free memory.
    # del matrix
    #
    # # Get the volume correction and the friction tensor.
    # rr = DiffusionTensor._correction_rr(rr, radii)
    # ft = DiffusionTensor._get_ftensor(tt, tr, rr)
    #
    # # Free memory.
    # del rr
    #
    # # Diffusion tensor is related to the inverse of the friction tensor.
    # function = DiffusionTensor._math_matrix_symmetrize
    # dtensor = function(inv(ft), passes=2)
    #
    # # return array(dtensor, dtype=float)
    return array([0.0], dtype=float)


class DiffusionTensor:
    """
    Class that contains the static functions to calculate the diffusion tensor.
    """

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Correction Methods.
    # --------------------------------------------------------------------------

    @staticmethod
    def _correction_rr(rr_tensor: ndarray, radii: ndarray) -> ndarray:
        """
            Gets the corrected rotation-rotation coupling tensor by adding the
            volume correction.

            :param rr_tensor: The tensor to be corrected.

            :param radii: The radius of all the atoms.

            :return: The volume corrected rotation-rotation coupling tensor.
        """

        # Get the volume correction.
        rr = copy.deepcopy(rr_tensor)
        volume = sum([radius**3 for radius in radii]) * (8.0 * pi)

        rr += volume * identity(3)

        return rr

    # --------------------------------------------------------------------------
    # Get Methods.
    # --------------------------------------------------------------------------

    @staticmethod
    def _get_ftensor(tt: ndarray, tr: ndarray, rr: ndarray):
        """
            Gets the friction tensor from the translation-translation,
            translation- rotation and rotation-rotation tensors.

            :param tt: The translation-translation coupling tensor.

            :param tr: The translation-rotation coupling tensor.

            :param rr: The rotation-rotation coupling tensor.

            :return: The 6x6 friction tensor.
        """

        # Friction tensor.
        friction = zeros((6, 6))

        # Append the translation tensor.
        friction[0: 3, 0: 3] = tt

        # Append the translation-rotation tensor in the lower part.
        friction[3:, 0: 3] = tr

        # Append the translation-rotation transpose tensor in the upper part.
        friction[0: 3, 3:] = transpose(tr)

        # Append the rotation-rotation transpose tensor in the upper part.
        friction[3:, 3:] = rr

        return DiffusionTensor._math_matrix_symmetrize(friction, passes=2)

    @staticmethod
    def _get_tensor_rr(matrix: ndarray, coordinates: ndarray) -> ndarray:
        """
            Gets the rotation-rotation coupling tensor from the matrix.

            :param matrix: The inverse matrix of the B matrix.

            :param coordinates: The coordinates of the particles.

            :return: The 3x3 rotation-rotation coupling friction tensor.
        """

        # //////////////////////////////////////////////////////////////////////
        # Auxiliary Functions
        # //////////////////////////////////////////////////////////////////////

        def get_asmatrix_0(vector_0: ndarray) -> ndarray:
            """
                Gets the 3D anti-symmetric matrix.

                :return: The 3D anti-symmetric matrix.
            """
            return array(
                [
                    [0.0, -vector_0[2], vector_0[1]],
                    [vector_0[2], 0.0, -vector_0[0]],
                    [-vector_0[1], vector_0[0], 0.0]
                ], dtype=float
            )

        # //////////////////////////////////////////////////////////////////////
        # Implementation Functions
        # //////////////////////////////////////////////////////////////////////

        # Auxiliary matrix.
        amatrix = zeros((3, 3))

        # Go through the 3x3 blocks.
        for i, c0 in enumerate(coordinates):
            # The anti-symmetric matrix for coordinate 0.
            cmatrix0 = get_asmatrix_0(c0)

            for j, c1 in enumerate(coordinates):
                # The anti-symmetric matrix for coordinate 1.
                cmatrix1 = transpose(get_asmatrix_0(c1))

                # Get the matrix term.
                temp_matrix = matrix[i * 3: i * 3 + 3, j * 3: j * 3 + 3]
                mat1 = matmul(cmatrix0, matmul(temp_matrix, cmatrix1))
                mat2 = matmul(matmul(cmatrix0, temp_matrix), cmatrix1)

                # Add to the accumulated matrix.
                amatrix += (mat1 + mat2) * 0.5

        return amatrix

    @staticmethod
    def _get_tensor_tr(matrix: ndarray, coordinates: ndarray) -> ndarray:
        """
            Gets the translation-rotation coupling tensor from the matrix.

            :param matrix: The inverse matrix of the B matrix.

            :param coordinates: The coordinates of the particles.

            :return: The 3x3 translation-rotation coupling friction tensor.
        """

        # //////////////////////////////////////////////////////////////////////
        # Auxiliary Functions
        # //////////////////////////////////////////////////////////////////////

        def get_asmatrix_0(vector_0: ndarray) -> ndarray:
            """
                Gets the 3D anti-symmetric matrix.

                :return: The 3D anti-symmetric matrix.
            """
            return array(
                [
                    [0.0, -vector_0[2], vector_0[1]],
                    [vector_0[2], 0.0, -vector_0[0]],
                    [-vector_0[1], vector_0[0], 0.0]
                ], dtype=float
            )

        # //////////////////////////////////////////////////////////////////////
        # Implementation Functions
        # //////////////////////////////////////////////////////////////////////

        # Auxiliary matrix.
        amatrix = zeros((3, 3))

        # Go through the 3x3 blocks.
        for i, c0 in enumerate(coordinates):
            # The anti-symmetric matrix for coordinate 0.
            cmatrix0 = get_asmatrix_0(c0)

            for j, c1 in enumerate(coordinates):

                # Get the matrix term.
                temp_matrix = matrix[i * 3: i * 3 + 3, j * 3: j * 3 + 3]

                # Add to the accumulated matrix.
                amatrix += matmul(cmatrix0, temp_matrix)

        return amatrix

    @staticmethod
    def _get_tensor_tt(matrix: ndarray, coordinates: ndarray) -> ndarray:
        """
            Gets the translation-translation coupling tensor from the matrix.

            :param matrix: The inverse matrix of the B matrix.

            :param coordinates: The coordinates of the particles.

            :return: The 3x3 translation-translation coupling friction tensor.
        """

        # Auxiliary matrix.
        amatrix = zeros((3, 3))

        # Go through the 3x3 blocks.
        for i, c0 in enumerate(coordinates):
            for j, c1 in enumerate(coordinates):

                # Add to the accumulated matrix.
                amatrix += matrix[i * 3: i * 3 + 3, j * 3: j * 3 + 3]

        return amatrix

    # --------------------------------------------------------------------------
    # Math Methods.
    # --------------------------------------------------------------------------

    @staticmethod
    def _math_overlap(
        coord0: ndarray, radius0: float, coord1: ndarray, radius1: float
    ) -> bool:
        """
            Validates if two spheres overlap each other.

            :param coord0: The coordinate of the zeroth sphere.

            :param radius0: The radius of the zeroth sphere.

            :param coord1: The coordinate of the first sphere.

            :param radius1: The radius of the first sphere.
        """

        # Sum of the radii, squared.
        sradius = (radius0 + radius1) ** 2

        # The distance between the two particles, squared.
        difference = coord0 - coord1
        sdistance = dot(difference, difference)

        return sdistance < sradius

