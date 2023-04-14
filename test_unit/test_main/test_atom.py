"""
    File that contains the unit test for setting up the atom.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General

import mendeleev
import numpy as np
import random as rd
import string
import unittest

# User defined.
import code.main.atom as atom

# ##############################################################################
# Global Constants
# ##############################################################################

ITERATIONS = 100

# ##############################################################################
# Classes
# ##############################################################################


class TestAtom(unittest.TestCase):

    def test_change_aname(self):
        """
            Tests the creation of the atom and changing its name.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = np.array(
            [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Try to change the name to an empty string.
        with self.assertRaises(ValueError):
            matom.aname = ""

        # Try to change the name to None.
        with self.assertRaises(ValueError):
            matom.aname = None

        # Try to change the name to anything else.
        matom.aname = 6

        # Result must be a string.
        self.assertIsInstance(matom.aname, (str,))
        self.assertEqual(matom.aname, str(f"{6}"))

    def test_change_atype(self):
        """
            Tests changing the type of the atom.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = np.array(
            [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Try to change the name to an empty string.
        with self.assertRaises(ValueError):
            matom.atype = ""

        # Try to change the name to None.
        with self.assertRaises(ValueError):
            matom.atype = None

        # Try to change the name to anything else.
        matom.atype = 6

        # Result must be a string.
        self.assertIsInstance(matom.atype, (str,))
        self.assertEqual(matom.atype, str(f"{6}"))

    def test_change_coordinates(self):
        """
            Tests the creation of the atom and changing its coordinates.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = np.array(
                [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Try to set the coordinates to a non-numpy array.
        with self.assertRaises(TypeError):
            matom.coordinates = [1.0, 2.0, 3.0]

        # Try to set the coordinates to a numpy array with the wrong dimensions.
        for i in [x for x in range(1, 11) if x != len(coordinate)]:
            with self.assertRaises(ValueError):
                matom.coordinates = np.array(
                    [rd.uniform(0.0, 1.0)] * i, dtype=float
                )

        # Try to set the coordinates to a numpy array with the coordinate types.
        with self.assertRaises(TypeError):
            length = len(coordinate)
            matom.coordinates = np.array(
                [rd.choice([i for i in range(10)]) for _ in range(length)],
                dtype=int
            )

        # Change the coordinates to something valid.
        length = len(coordinate)
        tcoordinates = np.array(
            [rd.uniform(-10.0, 10.0) for _ in range(length)],
            dtype=float
        )
        matom.coordinates = tcoordinates

        # Coordinates should have changed.
        self.assertEqual(len(matom.coordinates), len(tcoordinates))
        for crd0, crd1 in zip(matom.coordinates, tcoordinates):
            self.assertEqual(type(crd0), type(crd1))
            self.assertEqual(crd0, crd1)

    def test_change_mass(self):
        """
            Tests the creation of the atom and changing its mass.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = np.array(
                [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Try to set a zero or negative mass.
        for tmass in [0.0, -rd.uniform(0.01, 10.0)]:
            with self.assertRaises(ValueError):
                matom.mass = tmass

        # Try to set with the wrong type.
        with self.assertRaises(ValueError):
            matom.mass = f"{tmass}"

    def test_change_radius(self):
        """
            Tests the creation of the atom and changing its radius.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = np.array(
                [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Try to set a zero or negative mass.
        for tradius in [0.0, -rd.uniform(0.01, 10.0)]:
            with self.assertRaises(ValueError):
                matom.radius = tradius

        # Try to set with the wrong type.
        with self.assertRaises(ValueError):
            matom.radius = f"{tradius}"

    def test_creation(self):
        """
            Tests the creation of the atom and the basic quantities.
        """

        # Test several times.
        for _ in range(ITERATIONS):
            # Choose a random mass and radius
            mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
            coordinate = np.array(
                [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
            )

            # Create an atom.
            matom = atom.Atom(radius, mass, coordinate)

            # Check that the atom is properly created.
            self.assertEqual(mass, matom.mass)
            self.assertEqual(radius, matom.radius)

            # Check the remaining variables.
            self.assertEqual("---", matom.aname)
            self.assertEqual("---", matom.atype)

    def test_setup_from_atype(self):
        """
            Tests changing the type of the atom.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = np.array(
            [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Set the type to a random string.
        matom.atype = "".join(rd.choices(string.ascii_letters, k=5))

        # It must raise a warning.
        with self.assertWarns(UserWarning):
            matom.set_from_elements()

        # Change the type to carbon.
        element = mendeleev.element("C")
        matom.atype = "C"

        # Set the atom and test.
        matom.set_from_elements()
        self.assertEqual(element.mass, matom.mass)
        self.assertEqual(element.vdw_radius / 100.0, matom.radius)


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
