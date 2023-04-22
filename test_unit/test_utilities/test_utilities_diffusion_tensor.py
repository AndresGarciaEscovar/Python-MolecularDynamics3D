"""
    File that contains the tests for the math utilities.
"""
import copy
# ##############################################################################
# Imports
# ##############################################################################

# General
import unittest
from numpy import array, dot, identity, matmul, outer, pi, transpose, zeros
from numpy.linalg import inv, norm

# User defined.
import code.utilities.utilities_diffusion_tensor as udtensor
import code.utilities.utilities_math as umath

# ##############################################################################
# Classes
# ##############################################################################


class TestUtilitiesDiffusionTensor(unittest.TestCase):

    def test_get_btensor(self):
        """
            Tests that the get_btensor function is working properly; for a
            simple two atom case.
        """

        # ----------------------- Atoms don't intersect ---------------------- #

        # Try to setup a 3D molecule whose atoms don't intersect.
        coordinates = array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, +0.5]
            ], dtype=float
        )
        radii = array([1.0, 0.5])

        # Manually calculate the tensor.
        entry_00 = identity(3) / (6.0 * pi * radii[0])
        entry_11 = identity(3) / (6.0 * pi * radii[1])

        # Get the unequal tensor.
        difference = coordinates[0] - coordinates[1]
        oproduct = outer(difference, difference)

        dnorm = norm(difference)
        norms = dot(difference, difference)
        sradiuss = (radii[0] ** 2) + (radii[1] ** 2)

        # Calculate the tensor.
        entry_01 = identity(3)
        entry_01 += (oproduct / norms)
        entry_01 += ((identity(3) / 3.0) - (oproduct / norms)) * (sradiuss / norms)
        entry_01 /= (8.0 * pi * dnorm)

        # Get the unequal tensor.
        difference = coordinates[1] - coordinates[0]
        oproduct = outer(difference, difference)

        dnorm = norm(difference)
        norms = dot(difference, difference)
        sradiuss = (radii[0] ** 2) + (radii[1] ** 2)

        # Calculate the tensor.
        entry_10 = identity(3)
        entry_10 += (oproduct / norms)
        entry_10 += ((identity(3) / 3.0) - (oproduct / norms)) * (sradiuss / norms)
        entry_10 /= (8.0 * pi * dnorm)

        # This is a very symmetric molecule, so there should be no coupling.
        expected_btensor = array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ], dtype=float
        )

        # Set the entries.
        expected_btensor[0: 3, 0: 3] = entry_00
        expected_btensor[0: 3, 3: 6] = entry_01
        expected_btensor[3: 6, 0: 3] = entry_10
        expected_btensor[3: 6, 3: 6] = entry_11

        # Get the tensor.
        actual_btensor = udtensor.get_btensor(coordinates, radii)

        # Compare the dimensions.
        self.assertEqual(expected_btensor.shape, actual_btensor.shape)

        # Compare all the entries.
        for i in range(len(expected_btensor)):
            for j in range(len(expected_btensor)):
                self.assertEqual(expected_btensor[i, j], actual_btensor[i, j])

        # ------------- Atoms intersect and have different radii ------------- #

        # Try to setup a 3D molecule whose atoms intersect.
        coordinates = array(
            [
                [0.0, 0.0, -0.5],
                [0.0, 0.0, +0.5]
            ], dtype=float
        )
        radii = array([1.0, 0.5])

        # Manually calculate the tensor.
        entry_00 = identity(3) / (6.0 * pi * radii[0])
        entry_11 = identity(3) / (6.0 * pi * radii[1])

        # Get the unequal tensor.
        difference = coordinates[0] - coordinates[1]
        oproduct = outer(difference, difference)

        dnorm = norm(difference)
        norms = dot(difference, difference)
        sradiuss = (radii[0] ** 2) + (radii[1] ** 2)

        # Calculate the tensor.
        entry_01 = identity(3)
        entry_01 += (oproduct / norms)
        entry_01 += ((identity(3) / 3.0) - (oproduct / norms)) * (sradiuss / norms)
        entry_01 /= (8.0 * pi * dnorm)

        # Get the unequal tensor.
        difference = coordinates[1] - coordinates[0]
        oproduct = outer(difference, difference)

        dnorm = norm(difference)
        norms = dot(difference, difference)
        sradiuss = (radii[0] ** 2) + (radii[1] ** 2)

        # Calculate the tensor.
        entry_10 = identity(3)
        entry_10 += (oproduct / norms)
        entry_10 += ((identity(3) / 3.0) - (oproduct / norms)) * (sradiuss / norms)
        entry_10 /= (8.0 * pi * dnorm)

        # This is a very symmetric molecule, so there should be no coupling.
        expected_btensor = array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ], dtype=float
        )

        # Set the entries.
        expected_btensor[0: 3, 0: 3] = entry_00
        expected_btensor[0: 3, 3: 6] = entry_01
        expected_btensor[3: 6, 0: 3] = entry_10
        expected_btensor[3: 6, 3: 6] = entry_11

        # Get the tensor.
        actual_btensor = udtensor.get_btensor(coordinates, radii)

        # Compare the dimensions.
        self.assertEqual(expected_btensor.shape, actual_btensor.shape)

        # Compare all the entries.
        for i in range(len(expected_btensor)):
            for j in range(len(expected_btensor)):
                self.assertEqual(expected_btensor[i, j], actual_btensor[i, j])

        # ----------- Atoms don't intersect and have the same radii ---------- #

        # Try to setup a 3D molecule whose atoms don't intersect.
        coordinates = array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, +1.0]
            ], dtype=float
        )
        radii = array([1.0, 1.0])

        # Manually calculate the tensor.
        entry_00 = identity(3) / (6.0 * pi * radii[0])
        entry_11 = identity(3) / (6.0 * pi * radii[1])

        # Get the unequal tensor.
        difference = coordinates[0] - coordinates[1]
        oproduct = outer(difference, difference)

        dnorm = norm(difference)
        norms = dot(difference, difference)
        sradiuss = (radii[0] ** 2) + (radii[1] ** 2)

        # Calculate the tensor.
        entry_01 = identity(3)
        entry_01 += (oproduct / norms)
        entry_01 += ((identity(3) / 3.0) - (oproduct / norms)) * (sradiuss / norms)
        entry_01 /= (8.0 * pi * dnorm)

        # Get the unequal tensor.
        difference = coordinates[1] - coordinates[0]
        oproduct = outer(difference, difference)

        dnorm = norm(difference)
        norms = dot(difference, difference)
        sradiuss = (radii[0] ** 2) + (radii[1] ** 2)

        # Calculate the tensor.
        entry_10 = identity(3)
        entry_10 += (oproduct / norms)
        entry_10 += ((identity(3) / 3.0) - (oproduct / norms)) * (sradiuss / norms)
        entry_10 /= (8.0 * pi * dnorm)

        # This is a very symmetric molecule, so there should be no coupling.
        expected_btensor = array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ], dtype=float
        )

        # Set the entries.
        expected_btensor[0: 3, 0: 3] = entry_00
        expected_btensor[0: 3, 3: 6] = entry_01
        expected_btensor[3: 6, 0: 3] = entry_10
        expected_btensor[3: 6, 3: 6] = entry_11

        # Get the tensor.
        actual_btensor = udtensor.get_btensor(coordinates, radii)

        # Compare the dimensions.
        self.assertEqual(expected_btensor.shape, actual_btensor.shape)

        # Compare all the entries.
        for i in range(len(expected_btensor)):
            for j in range(len(expected_btensor)):
                self.assertEqual(expected_btensor[i, j], actual_btensor[i, j])

        # ------------- Atoms have the same radius and intersect ------------- #

        # Try to setup a 3D molecule whose atoms intersect.
        coordinates = array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, +1.0]
            ], dtype=float
        )
        radii = array([1.5, 1.5])

        # Manually calculate the tensor.
        entry_00 = identity(3) / (6.0 * pi * radii[0])
        entry_11 = identity(3) / (6.0 * pi * radii[1])

        # Get the unequal tensor.
        difference = coordinates[0] - coordinates[1]
        oproduct = outer(difference, difference)

        dnorm = norm(difference)
        sradiuss = radii[0]

        # Calculate the tensor.
        entry_01 = (1.0 - dnorm * 9.0 / (sradiuss * 32.0)) * identity(3)
        entry_01 += (3.0 / (32.0 * dnorm * sradiuss)) * oproduct
        entry_01 /= (6.0 * pi * sradiuss)

        # Get the unequal tensor.
        difference = coordinates[1] - coordinates[0]
        oproduct = outer(difference, difference)

        dnorm = norm(difference)
        sradiuss = radii[1]

        # Calculate the tensor.
        entry_10 = (1.0 - dnorm * 9.0 / (sradiuss * 32.0)) * identity(3)
        entry_10 += (3.0 / (32.0 * dnorm * sradiuss)) * oproduct
        entry_10 /= (6.0 * pi * sradiuss)

        # This is a very symmetric molecule, so there should be no coupling.
        expected_btensor = array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ], dtype=float
        )

        # Set the entries.
        expected_btensor[0: 3, 0: 3] = entry_00
        expected_btensor[0: 3, 3: 6] = entry_01
        expected_btensor[3: 6, 0: 3] = entry_10
        expected_btensor[3: 6, 3: 6] = entry_11

        # Get the tensor.
        actual_btensor = udtensor.get_btensor(coordinates, radii)

        # Compare the dimensions.
        self.assertEqual(expected_btensor.shape, actual_btensor.shape)

        # Compare all the entries.
        for i in range(len(expected_btensor)):
            for j in range(len(expected_btensor)):
                self.assertEqual(expected_btensor[i, j], actual_btensor[i, j])

    def test_get_correction_rr(self):
        """
            Tests that the get_correction_rr function is working properly; for a
            simple two atom case.
        """

        # Define a 3x3 matrix, whichever, and two real numbers.
        matrix = array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]],
            dtype=float
        )
        radii = array([3.0, 2.0], dtype=float)

        # The expected correction.
        expected_tensor = copy.deepcopy(matrix)
        expected_tensor += (radii[0]**3) * (8.0 * pi) * identity(3, dtype=float)
        expected_tensor += (radii[1]**3) * (8.0 * pi) * identity(3, dtype=float)

        # Get the correction.
        actual_tensor = udtensor.get_correction_rr(matrix, radii)

        # Dimensions should be equal.
        self.assertEqual(expected_tensor.shape, actual_tensor.shape)

        # Numbers should be equal.
        for i in range(len(expected_tensor)):
            for j in range(len(expected_tensor)):
                self.assertEqual(expected_tensor[i, j], actual_tensor[i, j])

    def test_get_coupling_tensor_tt(self):
        """
            Tests that the get_coupling_tensor_tt function is working properly;
            for a simple two atom case.
        """

        # Define a 6x6 matrix with some entries, irrelevant which ones they are.
        matrix = array(
            [[1.00, 2.00, 3.00, 4.00, 5.00, 6.00],
             [7.00, 8.00, 9.00, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
             [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
             [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
             [31.0, 32.0, 33.0, 34.0, 35.0, 36.0]],
            dtype=float
        )

        # Some coordinates..
        coordinates = array(
            [[3.0, 2.0, 4.0],
             [5.0, 6.0, 7.0]],
            dtype=float
        )

        # The expected correction.
        expected_tensor = zeros((3, 3), dtype=float)
        expected_tensor += matrix[0: 3, 0: 3]
        expected_tensor += matrix[0: 3, 3: 6]
        expected_tensor += matrix[3: 6, 0: 3]
        expected_tensor += matrix[3: 6, 3: 6]

        # Get the correction.
        actual_tensor = udtensor.get_coupling_tensor_tt(matrix, coordinates)

        # Dimensions should be equal.
        self.assertEqual(expected_tensor.shape, actual_tensor.shape)

        # Numbers should be equal.
        for i in range(len(expected_tensor)):
            for j in range(len(expected_tensor)):
                self.assertEqual(expected_tensor[i, j], actual_tensor[i, j])

    def test_get_coupling_tensor_rr(self):
        """
            Tests that the get_coupling_tensor_rr function is working properly;
            for a simple two atom case.
        """

        # Define a 6x6 matrix with some entries, irrelevant which ones they are.
        matrix = array(
            [[1.00, 2.00, 3.00, 4.00, 5.00, 6.00],
             [7.00, 8.00, 9.00, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
             [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
             [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
             [31.0, 32.0, 33.0, 34.0, 35.0, 36.0]],
            dtype=float
        )

        # Some coordinates..
        coordinates = array(
            [[3.0, 2.0, 4.0],
             [5.0, 6.0, 7.0]],
            dtype=float
        )

        # Define the skew anti-symmetric matrices.
        skew_0 = umath.get_skew_symmetric_matrix(coordinates[0])
        skew_1 = umath.get_skew_symmetric_matrix(coordinates[1])

        # The expected correction.
        term00 = matmul(matmul(skew_0, matrix[0: 3, 0: 3]), transpose(skew_0))
        term00 += matmul(skew_0, matmul(matrix[0: 3, 0: 3], transpose(skew_0)))
        term00 = term00 * 0.5

        term01 = matmul(matmul(skew_0, matrix[0: 3, 3: 6]), transpose(skew_1))
        term01 += matmul(skew_0, matmul(matrix[0: 3, 3: 6], transpose(skew_1)))
        term01 = term01 * 0.5

        term10 = matmul(matmul(skew_1, matrix[3: 6, 0: 3]), transpose(skew_0))
        term10 += matmul(skew_1, matmul(matrix[3: 6, 0: 3], transpose(skew_0)))
        term10 = term10 * 0.5

        term11 = matmul(matmul(skew_1, matrix[3: 6, 3: 6]), transpose(skew_1))
        term11 += matmul(skew_1, matmul(matrix[3: 6, 3: 6], transpose(skew_1)))
        term11 = term11 * 0.5

        expected_tensor = term00 + term01 + term10 + term11

        # Get the correction.
        actual_tensor = udtensor.get_coupling_tensor_rr(matrix, coordinates)

        # Dimensions should be equal.
        self.assertEqual(expected_tensor.shape, actual_tensor.shape)

        # Numbers should be equal.
        for i in range(len(expected_tensor)):
            for j in range(len(expected_tensor)):
                self.assertEqual(expected_tensor[i, j], actual_tensor[i, j])

    def test_get_coupling_tensor_tr(self):
        """
            Tests that the get_coupling_tensor_tr function is working properly;
            for a simple two atom case.
        """

        # Define a 6x6 matrix with some entries, irrelevant which ones they are.
        matrix = array(
            [[1.00, 2.00, 3.00, 4.00, 5.00, 6.00],
             [7.00, 8.00, 9.00, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
             [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
             [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
             [31.0, 32.0, 33.0, 34.0, 35.0, 36.0]],
            dtype=float
        )

        # Some coordinates..
        coordinates = array(
            [[3.0, 2.0, 4.0],
             [5.0, 6.0, 7.0]],
            dtype=float
        )

        # Define the skew anti-symmetric matrices.
        skew_0 = umath.get_skew_symmetric_matrix(coordinates[0])
        skew_1 = umath.get_skew_symmetric_matrix(coordinates[1])

        # The expected correction.
        term00 = matmul(skew_0, matrix[0: 3, 0: 3])
        term01 = matmul(skew_0, matrix[0: 3, 3: 6])
        term10 = matmul(skew_1, matrix[3: 6, 0: 3])
        term11 = matmul(skew_1, matrix[3: 6, 3: 6])

        expected_tensor = term00 + term01 + term10 + term11

        # Get the correction.
        actual_tensor = udtensor.get_coupling_tensor_tr(matrix, coordinates)

        # Dimensions should be equal.
        self.assertEqual(expected_tensor.shape, actual_tensor.shape)

        # Numbers should be equal.
        for i in range(len(expected_tensor)):
            for j in range(len(expected_tensor)):
                self.assertEqual(expected_tensor[i, j], actual_tensor[i, j])

    def test_get_friction_tensor(self):
        """
            Tests that the get_friction_tensor function is working properly; for
            a simple two atom case.
        """

        # Define a 6x6 matrix with some entries, irrelevant which ones they are.
        matrix = array(
            [[1.00, 2.00, 3.00, 19.0, 25.0, 31.00],
             [7.00, 8.00, 9.00, 20.0, 26.0, 32.0],
             [13.0, 14.0, 15.0, 21.0, 27.0, 33.0],
             [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
             [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
             [31.0, 32.0, 33.0, 34.0, 35.0, 36.0]],
            dtype=float
        )

        # Extract the relevant matrices.
        tt = matrix[0: 3, 0: 3]
        tr = matrix[3: 6, 0: 3]
        rr = matrix[3: 6, 3: 6]

        # The actual tensor.
        actual_tensor = udtensor.get_friction_tensor(tt, tr, rr)

        # Symmetrize twice.
        matrix = umath.symmetrize(matrix, passes=2)

        # Validate the entries.
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                self.assertEqual(matrix[i, j], actual_tensor[i, j])

    def test_get_tensor_requal(self):
        """
            Tests that the get_tensor_requal function is working properly; for
            a pair of coordinates and radii.
        """

        # Define a pair of coordinates and a radius.
        coordinates = array(
            [[0.0, 0.0, 0.5],
             [0.0, 0.0, -0.5]],
            dtype=float
        )
        radius = 3.0

        # Base tensors.
        difference = coordinates[0] - coordinates[1]
        oproduct = outer(difference, difference)

        # Distance between the two points.
        distance = norm(difference)

        # Calculate the tensor.
        tensor = (1.0 - distance * 9.0 / (radius * 32.0)) * identity(3)
        tensor += (3.0 / (32.0 * distance * radius)) * oproduct
        tensor /= (6.0 * pi * radius)

        # Get the expected tensor.
        expected_tensor = udtensor.get_tensor_requal(
            coordinates[0], coordinates[1], radius
        )

        # Shapes must be the same.
        self.assertEqual(tensor.shape, expected_tensor.shape)

        # Validate the entries.
        for i in range(len(tensor)):
            for j in range(len(tensor)):
                self.assertEqual(expected_tensor[i, j], tensor[i, j])

    def test_get_tensor_runequal(self):
        """
            Tests that the get_tensor_runequal function is working properly; for
            a pair of coordinates and a pair of radii.
        """

        # Define a pair of coordinates and a radius.
        coordinates = array(
            [[0.0, 0.0, 0.5],
             [0.0, 0.0, -0.5]],
            dtype=float
        )
        radii = array([0.5, 1.0], dtype=float)

        # Base tensors.
        difference = coordinates[0] - coordinates[1]
        oproduct = outer(difference, difference)

        # Distance between the two points.
        dnorm = norm(difference)
        norms = dot(difference, difference)

        # Sum of square of the radii.
        sradii = (radii[0] ** 2) + (radii[1] ** 2)

        # Calculate the tensor.
        tensor = identity(3)
        tensor += (oproduct / norms)
        tensor += ((identity(3) / 3.0) - (oproduct / norms)) * (sradii / norms)
        tensor /= (8.0 * pi * dnorm)

        # Get the expected tensor.
        expected_tensor = udtensor.get_tensor_runequal(
            coordinates[0], radii[0], coordinates[1], radii[1]
        )

        # Shapes must be the same.
        self.assertEqual(tensor.shape, expected_tensor.shape)

        # Validate the entries.
        for i in range(len(tensor)):
            for j in range(len(tensor)):
                self.assertEqual(expected_tensor[i, j], tensor[i, j])

    # ##########################################################################
    # Test Main Function(s)
    # ##########################################################################

    def test_get_diffusion_tensor(self):
        """
            Tests that the get_diffusion_tensor function is working properly;
            for a pair of coordinates and a pair of radii.
        """

        # ----------------------- Define the parameters ---------------------- #

        # Define a pair of coordinates and a radius.
        coordinates = array(
            [[0.0, 0.0, 0.5],
             [0.0, 0.0, -0.5]],
            dtype=float
        )
        radii = array([0.5, 1.0], dtype=float)

        # -------------------- Procedure to get the tensor ------------------- #

        # Get the big matrix.
        matrix = udtensor.get_btensor(coordinates, radii)

        # Invert the matrix, and symmetrize, to get the BIG C matrix.
        matrix = inv(matrix)
        matrix = umath.symmetrize(matrix, passes=2)

        # Get the different coupling tensors.
        tt = udtensor.get_coupling_tensor_tt(matrix, coordinates)
        tr = udtensor.get_coupling_tensor_tr(matrix, coordinates)
        rr = udtensor.get_coupling_tensor_rr(matrix, coordinates)

        # Free memory.
        del matrix

        # Get the volume correction and the friction tensor.
        rr = udtensor.get_correction_rr(rr, radii)
        tfriction = udtensor.get_friction_tensor(tt, tr, rr)

        # Free memory.
        del rr

        # Get the expected tensor.
        tensor = array(umath.symmetrize(inv(tfriction), passes=2), dtype=float)

        # Free memory.
        del tfriction

        # ---------------------- Get the expected tensor --------------------- #

        # Get the expected tensor.
        expected_tensor = udtensor.get_diffusion_tensor(coordinates, radii)

        # ----------------------------- Validate ----------------------------- #

        # Test the shape is correct.
        self.assertEqual(expected_tensor.shape, (6,6))
        self.assertEqual(expected_tensor.shape, tensor.shape)

        # Validate the entries.
        for i in range(len(tensor)):
            for j in range(len(tensor)):
                self.assertEqual(expected_tensor[i, j], tensor[i, j])


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
