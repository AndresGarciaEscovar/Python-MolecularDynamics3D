"""
    File that contains the tests for the molecule utilities.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General
import unittest
from numpy import array, identity, sqrt

# User defined.
import code.utilities.utilities_molecule as umolecule

# ##############################################################################
# Classes
# ##############################################################################


class TestUtilitiesMolecule(unittest.TestCase):

    def test_get_bounding_radius(self):
        """
            Tests that the get_bounding_radius function is working properly.
        """
        self.assertTrue(False)

    def test_get_cod(self):
        """
            Tests that the get_cod function is working properly; where cod is
            center of diffusion.
        """
        self.assertTrue(False)

    def test_get_cog(self):
        """
            Tests that the get_cog function is working properly; where cog is
            center of geometry.
        """
        """
                    Tests that the get_com function is working properly; where cog is
                    center of mass.
                """

        # ------------------------------ 1D Case ----------------------------- #

        # Set some coordinates.
        coordinates = array([
            [1],
            [2],
        ], dtype=float)
        radii = array([3, 4], dtype=float)

        # Get the center of geometry.
        expected_max = max([
            coordinates[0][0] + radii[0], coordinates[1][0] + radii[1]
        ])
        expected_min = min([
            coordinates[0][0] - radii[0], coordinates[1][0] - radii[1]
        ])
        expected = array([(expected_max + expected_min) * 0.5], dtype=float)
        current = umolecule.get_cog(coordinates, radii)

        # Dimensions must match.
        self.assertEqual(expected.shape, current.shape)

        # Values must match.
        for expected0, current0 in zip(expected, current):
            self.assertEqual(expected0, current0)

        # ------------------------------ 2D Case ----------------------------- #

        # Set some coordinates.
        coordinates = array([
            [1, 2],
            [3, 4],
        ], dtype=float)
        radii = array([3, 4], dtype=float)

        # Get the center of geometry.
        expected_max_0 = max([
            coordinates[0][0] + radii[0], coordinates[1][0] + radii[1]
        ])
        expected_min_0 = min([
            coordinates[0][0] - radii[0], coordinates[1][0] - radii[1]
        ])

        expected_max_1 = max([
            coordinates[0][1] + radii[0], coordinates[1][1] + radii[1]
        ])
        expected_min_1 = min([
            coordinates[0][1] - radii[0], coordinates[1][1] - radii[1]
        ])
        expected = array(
            [
                (expected_max_0 + expected_min_0) * 0.5,
                (expected_max_1 + expected_min_1) * 0.5,
            ], dtype=float
        )
        current = umolecule.get_cog(coordinates, radii)

        # Dimensions must match.
        self.assertEqual(expected.shape, current.shape)

        # Values must match.
        for expected0, current0 in zip(expected, current):
            self.assertEqual(expected0, current0)

        # ------------------------------ 3D Case ----------------------------- #

        # Set some coordinates.
        coordinates = array([
            [1, 2, 3],
            [4, 5, 6],
        ], dtype=float)
        radii = array([3, 4], dtype=float)

        # Get the center of geometry.
        expected_max_0 = max([
            coordinates[0][0] + radii[0], coordinates[1][0] + radii[1]
        ])
        expected_min_0 = min([
            coordinates[0][0] - radii[0], coordinates[1][0] - radii[1]
        ])

        expected_max_1 = max([
            coordinates[0][1] + radii[0], coordinates[1][1] + radii[1]
        ])
        expected_min_1 = min([
            coordinates[0][1] - radii[0], coordinates[1][1] - radii[1]
        ])

        expected_max_2 = max([
            coordinates[0][2] + radii[0], coordinates[1][2] + radii[1]
        ])
        expected_min_2 = min([
            coordinates[0][2] - radii[0], coordinates[1][2] - radii[1]
        ])

        expected = array(
            [
                (expected_max_0 + expected_min_0) * 0.5,
                (expected_max_1 + expected_min_1) * 0.5,
                (expected_max_2 + expected_min_2) * 0.5,
            ], dtype=float
        )
        current = umolecule.get_cog(coordinates, radii)

        # Dimensions must match.
        self.assertEqual(expected.shape, current.shape)

        # Values must match.
        for expected0, current0 in zip(expected, current):
            self.assertEqual(expected0, current0)

    def test_get_com(self):
        """
            Tests that the get_com function is working properly; where cog is
            center of mass.
        """

        # ------------------------------ 1D Case ----------------------------- #

        # Set some coordinates.
        coordinates = array([
            [1],
            [2],
        ], dtype=float)
        masses = array([3, 4], dtype=float)

        # Get the center of mass.
        expected = masses[0] * coordinates[0] + masses[1] * coordinates[1]
        expected /= (masses[0] + masses[1])
        current = umolecule.get_com(coordinates, masses)

        # Dimensions must match.
        self.assertEqual(expected.shape, current.shape)

        # Values must match.
        for expected0, current0 in zip(expected, current):
            self.assertEqual(expected0, current0)

        # ------------------------------ 2D Case ----------------------------- #

        # Set some coordinates.
        coordinates = array([
            [1, 2],
            [3, 4],
        ], dtype=float)
        masses = array([3, 4], dtype=float)

        # Get the center of mass.
        expected = masses[0] * coordinates[0] + masses[1] * coordinates[1]
        expected /= (masses[0] + masses[1])
        current = umolecule.get_com(coordinates, masses)

        # Dimensions must match.
        self.assertEqual(expected.shape, current.shape)

        # Values must match.
        for expected0, current0 in zip(expected, current):
            self.assertEqual(expected0, current0)

        # ------------------------------ 3D Case ----------------------------- #

        # Set some coordinates.
        coordinates = array([
            [1, 2, 3],
            [4, 5, 6],
        ],dtype=float)
        masses = array([3, 4], dtype=float)

        # Get the center of mass.
        expected = masses[0] * coordinates[0] + masses[1] * coordinates[1]
        expected /= (masses[0] + masses[1])
        current = umolecule.get_com(coordinates, masses)

        # Dimensions must match.
        self.assertEqual(expected.shape, current.shape)

        # Values must match.
        for expected0, current0 in zip(expected, current):
            self.assertEqual(expected0, current0)

    def test_get_dtensor(self):
        """
            Tests that the get_dtensor function is working properly.
        """
        self.assertTrue(False)

    def test_dtensor_and_orientation(self):
        """
            Tests that the dtensor_and_orientation function is working properly.
        """
        # Define a dictionary that contains a diffusion tensor + orientation.
        information = {
            "diffusion_tensor": array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ], dtype=float),

            "orientation":  array([
                [+sqrt(2), +sqrt(2), 0],
                [+sqrt(2), -sqrt(2), 0],
                [0, 0, 1],
            ], dtype=float),
        }

        # Get these from the information.
        dtensor, orient = umolecule.get_dtensor_and_orientation(information, 3)

        # Compare the diffusion tensor.
        self.assertEqual(information["diffusion_tensor"].shape, dtensor.shape)

        for row0, row1 in zip(information["diffusion_tensor"], dtensor):
            for col0, col1 in zip(row0, row1):
                self.assertEqual(col0, col1)

        # Compare the orientation.
        self.assertEqual(information["orientation"].shape, orient.shape)

        for row0, row1 in zip(information["orientation"], orient):
            for col0, col1 in zip(row0, row1):
                self.assertEqual(col0, col1)

        # -------------------- Diffusion Tensor is Missing ------------------- #

        # Define a dictionary that contains a diffusion tensor + orientation.
        information = {
            "orientation": array([
                [+sqrt(2), +sqrt(2), 0],
                [+sqrt(2), -sqrt(2), 0],
                [0, 0, 1],
            ], dtype=float),
        }

        # Get these from the information.
        dtensor, orient = umolecule.get_dtensor_and_orientation(information, 3)

        # Compare the diffusion tensor.
        self.assertIsNone(dtensor)

        # Compare the orientation.
        self.assertEqual(information["orientation"].shape, orient.shape)

        for row0, row1 in zip(information["orientation"], orient):
            for col0, col1 in zip(row0, row1):
                self.assertEqual(col0, col1)

        # ---------------------- Orientation is Missing ---------------------- #

        # Define a dictionary that contains a diffusion tensor + orientation.
        information = {
            "diffusion_tensor": array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ], dtype=float),
        }

        # Get these from the information.
        dtensor, orient = umolecule.get_dtensor_and_orientation(information, 3)

        # Compare the diffusion tensor.
        self.assertEqual(information["diffusion_tensor"].shape, dtensor.shape)

        for row0, row1 in zip(information["diffusion_tensor"], dtensor):
            for col0, col1 in zip(row0, row1):
                self.assertEqual(col0, col1)

        # Compare the orientation.
        self.assertEqual(identity(3).shape, orient.shape)

        for row0, row1 in zip(identity(3), orient):
            for col0, col1 in zip(row0, row1):
                self.assertEqual(col0, col1)

# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
