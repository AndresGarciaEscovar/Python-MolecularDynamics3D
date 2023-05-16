"""
    File that contains the tests for the math utilities.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General
import unittest

from numpy import array, pi

# User defined.
import code.utilities.utilities_math as umath

# ##############################################################################
# Classes
# ##############################################################################


class TestUtilitiesMath(unittest.TestCase):

    def test_get_skew_symmetric_matrix(self):
        """
            Tests that the get_skew_symmetric_matrix method is working properly.
        """

        # Loop through the different types of arrays.
        for mtype in (list, tuple):
            tarray = mtype([1, 2, 3])
            with self.assertRaises(TypeError):
                umath.get_skew_symmetric_matrix(tarray)

            tarray = array(tarray, dtype=float)
            rmatrix = umath.get_skew_symmetric_matrix(tarray)

            compmatrix = array(
                [
                    [0.0, -tarray[2], tarray[1]],
                    [tarray[2], 0.0, -tarray[0]],
                    [-tarray[1], tarray[0], 0.0]
                ], dtype=float
            )

            # Matrices must be equal, element by element.
            for i, row in enumerate(compmatrix):
                for j, column in enumerate(row):
                    self.assertEqual(column, rmatrix[i, j])

            tarray = array([1, 2, 3, 4], dtype=float)
            with self.assertRaises(ValueError):
                umath.get_skew_symmetric_matrix(tarray)

    def test_get_projection(self):
        """
            Tests that the get_projection method is working properly.
        """
        # Define two vectors.
        vector_0 = array([0, 3, 2], dtype=float)
        vector_1 = array([0, 0, 2], dtype=float)

        # Expected and result.
        expected = array([0, 0, 2], dtype=float)
        result = umath.get_projection(vector_0, vector_1)

        # Compare shapes.
        self.assertEqual(expected.shape, result.shape)

        # ------------------------- Division by Zero ------------------------- #

        # Define two vectors.
        vector_0 = array([0, 3, 2], dtype=float)
        vector_1 = array([0, 0, 0], dtype=float)

        # Expected and result.
        with self.assertRaises(RuntimeWarning):
            umath.get_projection(vector_0, vector_1)

        # Define two vectors.
        vector_0 = array([0, 3, 0], dtype=float)
        vector_1 = array([0, 0, 0], dtype=float)

        # Must throw an exception.
        with self.assertRaises(RuntimeWarning):
            umath.get_projection(vector_0, vector_1)

        # ----------------------- Different Size Arrays ---------------------- #

        # Define two vectors.
        vector_0 = array([0, 3, 0], dtype=float)
        vector_1 = array([0, 1], dtype=float)

        # Must throw an exception.
        with self.assertRaises(ValueError):
            umath.get_projection(vector_0, vector_1)

    def test_intersect_spheres(self):
        """
            Tests that the function intersect_spheres is working properly.
        """
        # Define the radii.
        radius_0 = 1.0
        radius_1 = 1.0

        # Define two coordinates.
        coordinates_0 = array([0, 0, 1], dtype=float)
        coordinates_1 = [0, 0, 0]

        # Wrong type for the first array.
        with self.assertRaises(TypeError):
            umath.intersect_hspheres(
                coordinates_0, radius_0, coordinates_1, radius_1
            )

        # Define two coordinates.
        coordinates_0 = [0, 0, 0]
        coordinates_1 = array([0, 0, 1], dtype=float)

        # Wrong type for the zeroth array.
        with self.assertRaises(TypeError):
            umath.intersect_hspheres(
                coordinates_0, radius_0, coordinates_1, radius_1
            )

        # Define two coordinates.
        coordinates_0 = array([0, 0, 1, 0], dtype=float)
        coordinates_1 = array([0, 0, 1], dtype=float)

        # Arrays of different lengths.
        with self.assertRaises(ValueError):
            umath.intersect_hspheres(
                coordinates_0, radius_0, coordinates_1, radius_1
            )

        # Define two coordinates.
        coordinates_0 = array([0, 0, 0], dtype=float)
        coordinates_1 = array([0, 0, 1], dtype=int)

        # Arrays wrong numerical types.
        with self.assertRaises(TypeError):
            umath.intersect_hspheres(
                coordinates_0, radius_0, coordinates_1, radius_1
            )

        # Define two coordinates.
        radius_0 = 1.0
        coordinates_0 = array([0, 0, 0], dtype=float)

        radius_1 = 1.0
        coordinates_1 = array([0, 0, 1], dtype=float)

        # Spheres must intersect.
        self.assertTrue(
            umath.intersect_hspheres(
                coordinates_0, radius_0, coordinates_1, radius_1
            )
        )

        # Spheres must not intersect; i.e., barely touching.
        radius_0 = 0.5
        radius_1 = 0.5

        self.assertFalse(
            umath.intersect_hspheres(
                coordinates_0, radius_0, coordinates_1, radius_1
            )
        )

        # Spheres must not intersect; i.e., far from touching.
        radius_0 = 0.2
        radius_1 = 0.2

        self.assertFalse(
            umath.intersect_hspheres(
                coordinates_0, radius_0, coordinates_1, radius_1
            )
        )

    def test_rotate_vector(self):
        """
            Tests that the function symmetrize is working properly.
        """

        #  This one will NOT change.
        about = array([0, 0, 0], dtype=float)

        # ------------------------- About the x-axis ------------------------- #

        # Define the vectors.
        vector = array([+1, 0, 0], dtype=float)
        around = array([+1, 0, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([1.0, 0.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, +1, 0], dtype=float)
        around = array([+1, 0, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, 0.0, 1.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, 0, +1], dtype=float)
        around = array([+1, 0, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, -1.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # --------------------- About the negative x-axis -------------------- #

        # Define the vectors.
        vector = array([+1, 0, 0], dtype=float)
        around = array([-1, 0, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([1.0, 0.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, +1, 0], dtype=float)
        around = array([-1, 0, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, 0.0, -1.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, 0, +1], dtype=float)
        around = array([-1, 0, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, 1.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # ------------------------- About the y-axis ------------------------- #

        # Define the vectors.
        vector = array([+1, 0, 0], dtype=float)
        around = array([0, +1, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, 0.0, -1.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, +1, 0], dtype=float)
        around = array([0, +1, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, 1.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, 0, +1], dtype=float)
        around = array([0, +1, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([1.0, 0.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # --------------------- About the negative y-axis -------------------- #

        # Define the vectors.
        vector = array([+1, 0, 0], dtype=float)
        around = array([0, -1, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, 0.0, 1.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, +1, 0], dtype=float)
        around = array([0, -1, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, 1.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, 0, +1], dtype=float)
        around = array([0, -1, 0], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([-1.0, 0.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # ------------------------- About the z-axis ------------------------- #

        # Define the vectors.
        vector = array([+1, 0, 0], dtype=float)
        around = array([0, 0, +1], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, 1.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, +1, 0], dtype=float)
        around = array([0, 0, +1], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([-1.0, 0.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, 0, +1], dtype=float)
        around = array([0, 0, +1], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, 0.0, 1.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # --------------------- About the negative z-axis -------------------- #

        # Define the vectors.
        vector = array([+1, 0, 0], dtype=float)
        around = array([0, 0, -1], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, -1.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, +1, 0], dtype=float)
        around = array([0, 0, -1], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([1.0, 0.0, 0.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

        # Define the vectors.
        vector = array([0, 0, +1], dtype=float)
        around = array([0, 0, -1], dtype=float)

        # Amount to be rotated.
        amount = pi * 0.5

        # Expected result.
        expected = array([0.0, 0.0, 1.0], dtype=float)

        # Get the result.
        result = umath.rotate_vector(vector, around, amount, about)

        # Shapes must be the same.
        self.assertEqual(result.shape, expected.shape)

        for i in range(result.shape[0]):
            self.assertAlmostEqual(expected[i], result[i], 10)

    def test_translate_vector(self):
        """
            Tests that the function translate_vector is working properly.
        """

        # Two vectors of the wrong type.
        vector0 = [1, 2, 3]
        vector1 = [1, 2, 3]

        with self.assertRaises(TypeError):
            umath.translate_vector(vector0, vector1)

        # Two vectors, one of the wrong type.
        vector0 = array([1, 2, 3], dtype=float)
        vector1 = [1, 2, 3]

        with self.assertRaises(TypeError):
            umath.translate_vector(vector0, vector1)

        # Two vectors, one with a different length.
        vector0 = array([1, 2, 3, 4], dtype=float)
        vector1 = array([1, 2, 3], dtype=float)

        with self.assertRaises(ValueError):
            umath.translate_vector(vector0, vector1)

        # Two vectors, one with a different length.
        vector0 = array([1, 2, 4], dtype=float)
        vector1 = array([1, 2, 3], dtype=float)

        vector2 = umath.translate_vector(vector0, vector1)

        self.assertEqual(vector0.shape, vector2.shape)
        for value0, value1, value2 in zip(vector0, vector1, vector2):
            self.assertEqual(value0 + value1, value2)

    def test_symmetrize(self):
        """
            Tests that the function symmetrize is working properly.
        """
        # Define a non-square matrix.
        matrix = array([
            [1, 2, 3],
            [4, 5, 6]
        ], dtype=float)

        # Wrong dimensions.
        with self.assertRaises(ValueError):
            umath.symmetrize(matrix, passes=3)

        # Define square matrix.
        matrix = array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=int)

        with self.assertRaises(TypeError):
            umath.symmetrize(matrix)

        # Wrong type of matrix.
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        with self.assertRaises(TypeError):
            umath.symmetrize(matrix)

        # Define square matrix.
        matrix = array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=float)

        # Correct dimensions.
        rmatrix = umath.symmetrize(matrix, passes=3)

        # Must be the same dimensions.
        self.assertEqual(matrix.shape, rmatrix.shape)

        for i in range(len(matrix)):
            for j in range(len(matrix)):
                self.assertEqual(rmatrix[i, j], rmatrix[j, i])


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
