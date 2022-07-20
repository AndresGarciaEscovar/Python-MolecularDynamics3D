""" File that contains the DiffusionTensor class. """

# ------------------------------------------------------------------------------
# Imports.
# ------------------------------------------------------------------------------

# Imports: General.
import copy
import numpy as np
import numpy.linalg
from numpy import ndarray, float64

from typing import Union

# ------------------------------------------------------------------------------
# Classes.
# ------------------------------------------------------------------------------


class DiffusionTensor:
    """
    Class that contains the static functions to calculate the diffusion tensor.
    """

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Public Interface
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Get Methods
    # --------------------------------------------------------------------------

    @staticmethod
    def get_diffusion_tensor(coordinates: ndarray, radii: ndarray) -> ndarray:
        """
            Get the diffusion tensor with tt-tr-rr coupling.

            :param coordinates: The coordinates of each atom in the molecule.

            :param radii: The radii of each atom in the molecule.

            :return: A 6x6 np array whose entries are the diffusion tensor.
        """

        # Get the big matrix.
        matrix = DiffusionTensor._get_btensor(coordinates, radii)

        # # Invert the matrix to get the BIG C matrix.
        # matrix = np.linalg.inv(matrix)
        # matrix = Mathematics.get_symmetric(matrix, passes=2)
        #
        # # Get the different coupling tensors.
        # tt = DiffusionTensor.get_tensor_translation_translation(matrix, np.array(coordinates, dtype=float))
        # tr = DiffusionTensor.get_tensor_translation_rotation(matrix, np.array(coordinates, dtype=float))
        # rr = DiffusionTensor.get_tensor_rotation_rotation(matrix, np.array(coordinates, dtype=float))
        #
        # # Get the volume correction.
        # rr = DiffusionTensor.correction_rotation_rotation(rr, np.array(radii, dtype=float))
        #
        # # Diffusion tensor is related to the inverse of the friction tensor.
        # dt = np.linalg.inv(DiffusionTensor.get_friction_tensor(tt, tr, rr))
        # dt = Mathematics.get_symmetric(dt, passes=2)
        #
        return np.array([0.0])

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Private Interface
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Correction Methods.
    # --------------------------------------------------------------------------

    @ staticmethod
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
        volume = sum([radius**3 for radius in radii]) * (8.0 * np.pi)

        rr += volume * np.identity(3)

        return rr

    # --------------------------------------------------------------------------
    # Get Methods.
    # --------------------------------------------------------------------------

    @staticmethod
    def _get_btensor(coordinates: ndarray, radii: ndarray) -> ndarray:
        """
            Gets the big tensor with the elements to generate the friction
            tensor.

            :param coordinates: The coordinates of the particles.

            :param radii: The radii of the spherical particles.

            :return: The big tensor to get the friction tensor.
        """

        # Create the big matrix and identity matrix.
        dim = 3 * len(coordinates)
        identity = np.identity(dim)
        matrix = np.zeros((dim, dim), dtype=float)

        # Get the generators to loop.
        generator0 = zip(coordinates, radii)
        generator1 = zip(coordinates, radii)

        # Loop through the pairs.
        for i, (crd0, r0) in enumerate(generator0):
            for j, (crd1, r1) in enumerate(generator1):

                # Determine if the particles overlap.
                overlap = DiffusionTensor._math_overlap(crd0, r0, crd1, r1)

                # Choose the adequate case.
                if i == j:
                    tensor = np.identity(dim) / (6.0 * np.pi * r0)

                elif r0 == r1 and overlap:
                    function = DiffusionTensor.get_tensor_radii_equal
                    tensor = function(crd0, crd1, r0)

                else:
                    function = DiffusionTensor.get_tensor_radii_unequal
                    tensor = function(crd0, r0, crd1, r1)

                # Set the proper entries.
                xpos, ypos = i * dim, j * dim
                matrix[xpos : xpos + dim, ypos : ypos + dim] = tensor

        return matrix

    @staticmethod
    def get_friction_tensor(tt: np.ndarray, tr: np.ndarray, rr: np.ndarray):
        """
        Gets the friction tensor from the translation-translation, translation-
        rotation and rotation-rotation tensors.
        :param tt: The translation-translation coupling tensor.
        :param tr: The translation-rotation coupling tensor.
        :param rr: The rotation-rotation coupling tensor.
        :return: The 6x6 friction tensor, properly symmetrized.
        """

        # Friction tensor.
        friction = np.zeros((6, 6))

        # Append the translation tensor.
        friction[0: 3, 0: 3] = tt

        # Append the translation-rotation tensor in the lower part.
        friction[3:, 0: 3] = tr

        # Append the translation-rotation transpose tensor in the upper part.
        friction[0: 3, 3:] = np.transpose(tr)

        # Append the rotation-rotation transpose tensor in the upper part.
        friction[3:, 3:] = rr

        return Mathematics.get_symmetric(friction, passes=2)

    @staticmethod
    def get_tensor_radii_equal(coordinate_0: tuple, coordinate_1: tuple, radius: float) -> np.ndarray:
        """
        Gets the Rotne-Prager tensor for beads of equal radius.
        :param coordinate_0: The coordinates of the zeroth particle.
        :param coordinate_1: The coordinates of the first particle.
        :param radius: The radius of the particles, i.e., radius of both
         particles must be the same.
        :return: A np square matrix with the Rotne-Prager tensor for beads of
         equal radius.
        """

        # Base tensors.
        identity = np.identity(len(coordinate_0))
        difference = Mathematics.operation_subtract_vectors(coordinate_0, coordinate_1)
        direct = Mathematics.operation_direct_product(list(difference), list(difference))

        # Distance between the two points.
        distance = Mathematics.operation_get_norm(difference, squared=False)

        # Calculate the tensor.
        tensor = (1.0 - distance * 9.0 / (radius * 32.0)) * identity + (3.0 / (32.0 * distance * radius)) * direct
        tensor = tensor / (6.0 * np.pi * radius)

        return tensor

    @staticmethod
    def get_tensor_radii_unequal(
            coordinate_0: tuple, radius_0: float, coordinate_1: tuple, radius_1: float
    ) -> np.ndarray:
        """
        Gets the Rotne-Prager tensor for beads of unequal radius. If the beads
        intersect, for any reason, a WARNING message will be printed.
        :param coordinate_0: The coordinates of the zeroth particle.
        :param coordinate_1: The coordinates of the first particle.
        :param radius_0: The radius of the zeroth particle.
        :param radius_1: The radius of the first particle.
        :return: A np square matrix with the Rotne-Prager tensor for beads of
         equal radius.
        """

        # Base tensors.
        identity = np.identity(len(coordinate_0))
        difference = Mathematics.operation_subtract_vectors(coordinate_0, coordinate_1)
        direct = Mathematics.operation_direct_product(list(difference), list(difference))

        # Distance between the two points.
        distance = Mathematics.operation_get_norm(difference, squared=False)
        sdistance = Mathematics.operation_get_norm(difference, squared=True)

        # Sum of square of the radii.
        radii_squared = radius_0 * radius_0 + radius_1 * radius_1

        # Calculate the tensor.
        tensor = identity
        tensor += (direct / sdistance) + ((identity / 3.0) - (direct / sdistance)) * (radii_squared / sdistance)
        tensor = tensor / (8.0 * np.pi * distance)

        return tensor

    @staticmethod
    def get_tensor_rotation_rotation(matrix: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """
        Gets the rotation-rotation coupled friction tensor.
        :param matrix: The big C matrix from where to extract the information.
        :param coordinates: The coordinates of each particle.
        :return: The rotation-rotation coupled friction tensor.
        """

        # Auxiliary variables.
        particles = len(coordinates)
        dimensions = len(coordinates[0])
        rr = np.zeros((dimensions, dimensions))

        # Go through all the 3x3 matrices.
        for i in range(particles):
            # Get the 3x3 anti-symmetric mixed matrix.
            asymm_0 = Mathematics.get_anti_symmetric_mixed(coordinates[i])

            # Get the 3x3 coupling matrix.
            for j in range(particles):
                # Get the 3x3 anti-symmetric mixed matrix.
                asymm_1 = Mathematics.get_anti_symmetric_mixed(coordinates[j], transpose=True)
                mat = matrix[dimensions * i: dimensions * i + 3, dimensions * j: dimensions * j + 3]
                rr += (np.matmul(np.matmul(asymm_0, mat), asymm_1) + np.matmul(asymm_0, np.matmul(mat, asymm_1))) / 2.0

        return rr

    @staticmethod
    def get_tensor_translation_rotation(matrix: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """
        Gets the translation-rotation coupled friction tensor.
        :param matrix: The big C matrix from where to extract the information.
        :param coordinates: The coordinates of each particle.
        :return: The translation-rotation coupled friction tensor.
        """

        # Auxiliary variables.
        particles = len(coordinates)
        dimensions = len(coordinates[0])
        tr = np.zeros((dimensions, dimensions))

        # Go through all the 3x3 matrices.
        for i in range(particles):
            # Get the 3x3 anti-symmetric mixed matrix.
            asymm = Mathematics.get_anti_symmetric_mixed(coordinates[i])

            # Get the 3x3 coupling matrix.
            for j in range(particles):
                mat = matrix[dimensions * i: dimensions * i + 3, dimensions * j: dimensions * j + 3]
                tr += np.matmul(asymm, mat)

        return tr

    @staticmethod
    def get_tensor_translation_translation(matrix: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """
        Gets the translation-translation coupled friction tensor.
        :param matrix: The big C matrix from where to extract the information.
        :param coordinates: The coordinates of each particle.
        :return: The translation-translation coupled friction tensor.
        """

        # Auxiliary variables.
        particles = len(coordinates)
        dimensions = len(coordinates[0])
        tt = np.zeros((dimensions, dimensions))

        # Go through all the 3x3 matrices.
        for i in range(particles):
            for j in range(particles):
                tt += matrix[dimensions * i: dimensions * i + 3, dimensions * j: dimensions * j + 3]

        return tt

    # --------------------------------------------------------------------------
    # Math Methods.
    # --------------------------------------------------------------------------


if __name__ == "__main__":

    crdts = np.array([
        [1.0, 2.0, 3.0], [1.0, 2.0, 4.0]], dtype=np.float64
    )
    rad = np.array([0.5, 0.5], dtype=np.float64)

    dt = DiffusionTensor.get_diffusion_tensor(crdts, rad)