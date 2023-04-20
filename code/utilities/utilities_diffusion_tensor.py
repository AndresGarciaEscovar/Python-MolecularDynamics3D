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
import code.validation.validation_parameters as vparameters

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


def get_correction_rr(rr_tensor: ndarray, radii: ndarray) -> ndarray:
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

    # Make the correction.
    rr += volume * identity(3, dtype=float)

    return rr


def get_coupling_tensor_tt(matrix: ndarray, coordinates: ndarray) -> ndarray:
    """
        Gets the translation-translation coupling tensor from the matrix.

        :param matrix: The inverse matrix of the B matrix.

        :param coordinates: The coordinates of the particles.

        :return: The 3x3 translation-translation coupling friction tensor.
    """

    # Auxiliary variables.
    amatrix = zeros((3, 3))
    length = len(coordinates)

    # Add the 3x3 blocks.
    for i in range(length):
        for j in range(length):
            amatrix += matrix[i * 3: i * 3 + 3, j * 3: j * 3 + 3]

    return amatrix


def get_coupling_tensor_rr(matrix: ndarray, coordinates: ndarray) -> ndarray:
    """
        Gets the rotation-rotation coupling tensor from the matrix.

        :param matrix: The inverse matrix of the B matrix.

        :param coordinates: The coordinates of the particles.

        :return: The 3x3 rotation-rotation coupling friction tensor.
    """

    # Auxiliary matrix.
    amatrix = zeros((3, 3))

    # Go through the 3x3 blocks.
    for i, coordinate_0 in enumerate(coordinates):
        # The anti-symmetric matrix for coordinate 0.
        cmatrix0 = umath.get_skew_symmetric_matrix(coordinate_0)

        for j, coordinate_1 in enumerate(coordinates):
            # The anti-symmetric matrix for coordinate 1.
            cmatrix1 = transpose(
                umath.get_skew_symmetric_matrix(coordinate_1)
            )

            # Get the matrix term.
            tmatrix = matrix[i * 3: i * 3 + 3, j * 3: j * 3 + 3]

            # Multiplication order might introduce bias.
            matrix0 = matmul(cmatrix0, matmul(tmatrix, cmatrix1))
            matrix1 = matmul(matmul(cmatrix0, tmatrix), cmatrix1)

            # Add to the accumulated matrix.
            amatrix += (matrix0 + matrix1) * 0.5

    return amatrix


def get_coupling_tensor_tr(matrix: ndarray, coordinates: ndarray) -> ndarray:
    """
        Gets the translation-rotation coupling tensor from the matrix.

        :param matrix: The inverse matrix of the B matrix.

        :param coordinates: The coordinates of the particles.

        :return: The 3x3 translation-rotation coupling friction tensor.
    """

    # Auxiliary matrix.
    amatrix = zeros((3, 3))
    length = len(coordinates)

    # Go through the 3x3 blocks.
    for i, coordinate_0 in enumerate(coordinates):
        # The anti-symmetric matrix for the given coordinate.
        cmatrix = umath.get_skew_symmetric_matrix(coordinate_0)

        for j in range(length):
            # Get the corresponding 3x3 matrix.
            tmatrix = matrix[i * 3: i * 3 + 3, j * 3: j * 3 + 3]

            # Add to the accumulated matrix.
            amatrix += matmul(cmatrix, tmatrix)

    return amatrix


def get_friction_tensor(tt: ndarray, tr: ndarray, rr: ndarray):
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

    return umath.symmetrize(friction, passes=2)


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


# ##############################################################################
# Main Function(s)
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
    vparameters.is_shape_matrix(coordinates[0], (3,))

    # Get the big matrix.
    matrix = get_btensor(coordinates, radii)

    # Invert the matrix, and symmetrize, to get the BIG C matrix.
    matrix = inv(matrix)
    matrix = umath.symmetrize(matrix, passes=2)

    # Get the different coupling tensors.
    tt = get_coupling_tensor_tt(matrix, coordinates)
    tr = get_coupling_tensor_tr(matrix, coordinates)
    rr = get_coupling_tensor_rr(matrix, coordinates)

    # Free memory.
    del matrix

    # Get the volume correction and the friction tensor.
    rr = get_correction_rr(rr, radii)
    tfriction = get_friction_tensor(tt, tr, rr)

    # Free memory.
    del rr

    return array(umath.symmetrize(inv(tfriction), passes=2), dtype=float)
