""" File that contains the DiffusionTensor class. """

# ------------------------------------------------------------------------------
# Imports.
# ------------------------------------------------------------------------------

# General.
import copy
import numpy as np
from numpy import ndarray, float64

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

        # Invert the matrix, and symmetrize, to get the BIG C matrix.
        matrix = np.linalg.inv(matrix)
        matrix = DiffusionTensor._math_matrix_symmetrize(matrix, passes=2)

        # Get the different coupling tensors.
        tt = DiffusionTensor._get_tensor_tt(matrix, coordinates)
        tr = DiffusionTensor._get_tensor_tr(matrix, coordinates)
        rr = DiffusionTensor._get_tensor_rr(matrix, coordinates)

        # Free memory.
        del matrix

        # Get the volume correction and the friction tensor.
        rr = DiffusionTensor._correction_rr(rr, radii)
        ft = DiffusionTensor._get_ftensor(tt, tr, rr)

        # Free memory.
        del rr

        # Diffusion tensor is related to the inverse of the friction tensor.
        function = DiffusionTensor._math_matrix_symmetrize
        dtensor = function(np.linalg.inv(ft), passes=2)

        return np.array(dtensor, dtype=float)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Private Interface
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

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

        # //////////////////////////////////////////////////////////////////////
        # Auxiliary Functions
        # //////////////////////////////////////////////////////////////////////

        def get_tensor_requal_0(
                crd0_0: ndarray, crd1_0: ndarray, r_0: float64
        ) -> ndarray:
            """
                Gets the 3x3 friction tensor when the radii of the spheres are
                the same.

                :param crd0_0: The coordinate of the zeroth particle.

                :param crd1_0: The coordinate of the firts particle.

                :param r_0: The radius of either particle.

                :return: The 3x3 matrix with the friction tensor.
            """

            # Base tensors.
            diff_0 = crd0_0 - crd1_0
            direct_0 = np.outer(diff_0, diff_0)

            # Distance between the two points.
            d_0 = np.linalg.norm(diff_0)

            # Calculate the tensor.
            tensor_0 = (1.0 - d_0 * 9.0 / (r_0 * 32.0)) * np.identity(3)
            tensor_0 += (3.0 / (32.0 * d_0 * r_0)) * direct_0

            tensor_0 = tensor_0 / (6.0 * np.pi * r_0)

            return tensor_0

        def get_tensor_runequal_0(
                crd0_0: ndarray, crd1_0: ndarray, r0_0: float64, r1_0: float64
        ) -> ndarray:
            """
                Gets the 3x3 friction tensor when the radii of the spheres are
                not the same.

                :param crd0_0: The coordinate of the zeroth particle.

                :param crd1_0: The coordinate of the firts particle.

                :param r0_0: The radius of zeroth particle.

                :param r1_0: The radius of first particle.

                :return: The 3x3 matrix with the friction tensor.
            """

            # Base tensors.
            diff_0 = crd0_0 - crd1_0
            direct_0 = np.outer(diff_0, diff_0)

            # Distance between the two points.
            d_0 = np.linalg.norm(diff_0)
            d2_0 = np.dot(diff_0, diff_0)

            # Sum of square of the radii.
            rsq_0 = (r0_0 ** 2) + (r1_0 ** 2)

            # Identity matrices.
            ident0_0 = np.identity(3)

            # Calculate the tensor.
            tensor_0 = np.identity(3)
            tensor_0 += (direct_0 / d2_0)
            tensor_0 += ((ident0_0 / 3.0) - (direct_0 / d2_0)) * (rsq_0 / d2_0)
            tensor_0 /= (8.0 * np.pi * d_0)

            return tensor_0

        # //////////////////////////////////////////////////////////////////////
        # Implementation Functions
        # //////////////////////////////////////////////////////////////////////

        # Create the big matrix and identity matrix.
        dim = 3 * len(coordinates)
        matrix = np.zeros((dim, dim), dtype=float64)

        # Loop through the pairs.
        for i, (crd0, r0) in enumerate(zip(coordinates, radii)):
            for j, (crd1, r1) in enumerate(zip(coordinates, radii)):

                # Determine if the particles overlap.
                overlap = DiffusionTensor._math_overlap(crd0, r0, crd1, r1)

                # Choose the adequate case.
                if i == j:
                    tensor = np.identity(3) / (6.0 * np.pi * r0)

                elif r0 == r1 and overlap:
                    tensor = get_tensor_requal_0(crd0, crd1, r0)

                else:
                    tensor = get_tensor_runequal_0(crd0, crd1, r0, r1)

                # Set the proper entries.
                xpos, ypos = i * 3, j * 3

                matrix[xpos: xpos + 3, ypos: ypos + 3] = tensor

        return matrix

    @staticmethod
    def _get_ftensor(tt: np.ndarray, tr: np.ndarray, rr: np.ndarray):
        """
            Gets the friction tensor from the translation-translation,
            translation- rotation and rotation-rotation tensors.

            :param tt: The translation-translation coupling tensor.

            :param tr: The translation-rotation coupling tensor.

            :param rr: The rotation-rotation coupling tensor.

            :return: The 6x6 friction tensor.
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

        return DiffusionTensor._math_matrix_symmetrize(friction, passes=2)

    @staticmethod
    def _get_tensor_rr(matrix: np.ndarray, coordinates: np.ndarray) -> ndarray:
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
            return np.array(
                [
                    [0, -vector_0[2], vector_0[1]],
                    [vector_0[2], 0, -vector_0[0]],
                    [-vector_0[1], vector_0[0], 0]
                ], dtype=float64
            )

        # //////////////////////////////////////////////////////////////////////
        # Implementation Functions
        # //////////////////////////////////////////////////////////////////////

        # Auxiliary matrix.
        amatrix = np.zeros((3, 3))

        # Go through the 3x3 blocks.
        for i, c0 in enumerate(coordinates):
            # The anti-symmetric matrix for coordinate 0.
            cmatrix0 = get_asmatrix_0(c0)

            for j, c1 in enumerate(coordinates):
                # The anti-symmetric matrix for coordinate 1.
                cmatrix1 = np.transpose(get_asmatrix_0(c1))

                # Get the matrix term.
                temp_matrix = matrix[i * 3: i * 3 + 3, j * 3: j * 3 + 3]
                mat1 = np.matmul(cmatrix0, np.matmul(temp_matrix, cmatrix1))
                mat2 = np.matmul(np.matmul(cmatrix0, temp_matrix), cmatrix1)

                # Add to the accumulated matrix.
                amatrix += (mat1 + mat2) * 0.5

        return amatrix

    @staticmethod
    def _get_tensor_tr(matrix: np.ndarray, coordinates: np.ndarray) -> ndarray:
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
            return np.array(
                [
                    [0, -vector_0[2], vector_0[1]],
                    [vector_0[2], 0, -vector_0[0]],
                    [-vector_0[1], vector_0[0], 0]
                ], dtype=float64
            )

        # //////////////////////////////////////////////////////////////////////
        # Implementation Functions
        # //////////////////////////////////////////////////////////////////////

        # Auxiliary matrix.
        amatrix = np.zeros((3, 3))

        # Go through the 3x3 blocks.
        for i, c0 in enumerate(coordinates):
            # The anti-symmetric matrix for coordinate 0.
            cmatrix0 = get_asmatrix_0(c0)

            for j, c1 in enumerate(coordinates):

                # Get the matrix term.
                temp_matrix = matrix[i * 3: i * 3 + 3, j * 3: j * 3 + 3]

                # Add to the accumulated matrix.
                amatrix += np.matmul(cmatrix0, temp_matrix)

        return amatrix

    @staticmethod
    def _get_tensor_tt(matrix: np.ndarray, coordinates: np.ndarray) -> ndarray:
        """
            Gets the translation-translation coupling tensor from the matrix.

            :param matrix: The inverse matrix of the B matrix.

            :param coordinates: The coordinates of the particles.

            :return: The 3x3 translation-translation coupling friction tensor.
        """

        # Auxiliary matrix.
        amatrix = np.zeros((3, 3))

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
        coord0: ndarray, radius0: float64, coord1: ndarray, radius1: float64
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
        sdistance = np.dot(difference, difference)

        return sdistance < sradius

    @staticmethod
    def _math_matrix_symmetrize(matrix: np.ndarray, passes: int = 1):
        """
            Symmetrizes the given square matrix.

            :param matrix: The matrix to be symmetrized.

            :return: The np array with the matrix symmetrized.
        """

        # Get a copy of the original matrix.
        mat = copy.deepcopy(matrix)

        # Only positive numbers allowed.
        if int(passes) <= 0:
            return mat

        # Symmetrize the elements, the given number of times.
        for k in range(int(passes)):

            # The last pass is to symmetrize and assign.
            for i in range(len(mat)):
                for j in range(len(mat)):
                    if i < j or (k < (passes - 1) and i != j):
                        mat[i, j] = (mat[i, j] + mat[j, i]) * 0.5
                        continue

                    mat[i, j] = mat[j, i]

        return mat
