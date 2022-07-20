"""
    File that contains the Molecule class and its methods.
"""

# ##############################################################################
# Imports
# ##############################################################################
import copy
import numpy
import os


# ##############################################################################
# Classes
# ##############################################################################


class Molecule:
    """
        Class that represents a rigid molecule made of spheres. A file with the
        molecule information can be provided to load the molecule.
    """

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Public Interface
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # ##########################################################################
    # Constructor
    # ##########################################################################

    def __init__(self, filename: str = None):
        """
            Constructs a new instance of the a molecule. If the name of the file
            is not provided, it will create a single-sphere molecule with a
            radius of 0.5A and mass of 1.0amu.

            A = Angstrom, amu = Atomic Mass Unit.

            :param filename: The name of the file from where the molecule must
             be loaded.
        """

        # Set the file name.
        self.filename = filename
        self.name = "molecule name"

        # Define the number type
        dtype = numpy.float64

        # Create the basic quantities.
        self.atoms = 1
        self.bounding_radius = 0.5
        self.coordinates = numpy.array([[0.0 for _ in range(3)]], dtype=dtype)
        self.masses = numpy.array([1.0 for _ in range(1)], dtype=dtype)
        self.radii = numpy.array([0.5 for _ in range(1)], dtype=dtype)

        # Validate masses and radii.
        self._validate_masses()
        self._validate_radii()

        # Load the molecule from the given file.
        if filename is not None:
            self.load(filename)

        # Get the center of mass and the center of geometry.
        self.center_of_mass = self._get_center_of_mass()
        self.center_of_geometry = self._get_center_of_geometry()

        # Translate to the center of mass.
        self._translate_to_center_of_mass()

        # Get the bounding radius.
        self.bounding_radius = self._get_bounding_radius()

    # ##########################################################################
    # Dunder Methods
    # ##########################################################################

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Load Methods
    # --------------------------------------------------------------------------

    def load(self, filename: str) -> None:
        """
            Loads the given values.

            :param filename: The name of the file where the data is stored.
        """

        # Auxiliary variables.
        atom_number = 0
        coordinates = []
        masses = []
        radii = []

        # Open the file and load the values.
        with open(filename, newline="\n", mode="r") as file:
            # Read the generator to the lines.
            lines = file.readlines()

            for i, line in enumerate(lines):
                # Read the line and tokenize it.
                line = line.strip().split(",")

                # Read the molecule name.
                if i == 0:
                    self.name = line[1]
                    continue

                # get the number of atoms in the molecule.
                if i == 1:
                    atom_number = int(line[1])
                    continue

                # Go up to the number of atoms.
                if i in range(2, 2 + atom_number):
                    coordinates.append([float(c) for c in line[1:4]])
                    radii.append(float(line[4]))
                    masses.append(float(line[5]))

        # Set the number of atoms.
        self.atoms = atom_number

        # Set the coordinates, masses and radii.
        self.coordinates = numpy.array(coordinates, dtype=numpy.float64)
        self.masses = numpy.array(masses, dtype=numpy.float64)
        self.radii = numpy.array(radii, dtype=numpy.float64)

    # --------------------------------------------------------------------------
    # Translate Methods
    # --------------------------------------------------------------------------

    def translate_to(self, vector: numpy.ndarray, check: bool = False) -> None:
        """
            Translates the molecule by the given vector.

            :param vector: The 3D vector by which the molecule should be
             translated.

            :param check: Boolean flag that indicates whether the translation
             vector should be checked for the dimensionality. True, if the
             vector should be checked; False, otherwise and set to as the
             default.
        """

        # //////////////////////////////////////////////////////////////////////
        # Auxiliary Functions
        # //////////////////////////////////////////////////////////////////////

        def validate_coordinate_0() -> None:
            """
                Validates the dimensionality and type of array that represents
                the translation.
            """

            # Set the data type.
            d_0 = numpy.float64

            # Check for a 3D numpy array of numpy floats.
            i_0 = isinstance(vector, (numpy.ndarray,))
            i_0 = i_0 and all(map(lambda x: isinstance(x, (d_0,)), vector))
            if not i_0 or len(vector) != 3:
                raise TypeError(
                    "The given array must be a 3D numpy array of numpy.float64 "
                    f"values.\nArray: {vector}\nType: {type(vector)}"
                    f"\nTypes: {tuple(map(type, vector))}."
                )

        # //////////////////////////////////////////////////////////////////////
        # Implementation
        # //////////////////////////////////////////////////////////////////////

        # Validate the translation.
        validate_coordinate_0() if check else None

        # Translate the coordinates to the center of mass.
        for i, _ in enumerate(self.coordinates):
            self.coordinates[i] += vector

        # Translate the center of geometry and center of mass.
        self.center_of_geometry += vector
        self.center_of_mass += vector

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Private Interface
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Get Methods
    # --------------------------------------------------------------------------

    def _get_bounding_radius(self) -> numpy.float64:
        """
            Gets the bounding radius of the molecule, with respect to the center
            of mass.

            :return: The sphere with the shortest radius that encloses the
             molecule, with respect to the center of mass.
        """

        # Auxiliary variables.
        bradius = 0.0

        # Translate the molecule, temporarily, to the center of mass.
        translation = copy.deepcopy(self.center_of_mass)
        self.translate_to(-translation)

        # Get the maximum bounding radius.
        for c, r in zip(self.coordinates, self.radii):
            bradius = max(bradius, numpy.linalg.norm(c) + r)

        # Translate back to the original center of mass.
        self.translate_to(translation)

        return numpy.float64(bradius)

    def _get_center_of_geometry(self) -> numpy.ndarray:
        """
            From the radii and the positions, gets the center of geometry.

            :return: The center of geometry of the molecule.
        """

        # Get the radius and mass generator.
        generator = zip(self.coordinates, self.radii)

        maximum = numpy.array(
            [position + radius for position, radius in generator],
            dtype=numpy.float64
        )

        maximum = numpy.array(
            [max(maximum[:, i]) for i in range(3)],
            dtype=numpy.float64
        )

        # Get the radius and mass generator.
        generator = zip(self.coordinates, self.radii)

        minimum = numpy.array(
            [position - radius for position, radius in generator],
            dtype=numpy.float64
        )

        minimum = numpy.array(
            [min(minimum[:, i]) for i in range(3)],
            dtype=numpy.float64
        )

        return (maximum + minimum) * 0.5

    def _get_center_of_mass(self) -> numpy.ndarray:
        """
            From the masses and the positions, gets the center of mass.

            :return: The center of mass of the molecule.
        """

        # Get the total mass.
        mass = self.masses.sum()

        return numpy.dot(self.masses, self.coordinates) / mass

    # --------------------------------------------------------------------------
    # Translate Methods
    # --------------------------------------------------------------------------

    def _translate_to_center_of_geometry(self) -> None:
        """
            Translates the molecule to the center of geometry.
        """

        # Translate the coordinates to the center of mass.
        for i, _ in enumerate(self.coordinates):
            self.coordinates[i] -= self.center_of_geometry

        # Translate the center of geometry and center of mass.
        self.center_of_mass -= self.center_of_geometry
        self.center_of_geometry -= self.center_of_geometry

    def _translate_to_center_of_mass(self) -> None:
        """
            Translates the molecule to the center of mass.
        """

        # Translate the coordinates to the center of mass.
        for i, _ in enumerate(self.coordinates):
            self.coordinates[i] -= self.center_of_mass

        # Translate the center of geometry and center of mass.
        self.center_of_geometry -= self.center_of_mass
        self.center_of_mass -= self.center_of_mass

    # --------------------------------------------------------------------------
    # Validate Methods
    # --------------------------------------------------------------------------

    def _validate_masses(self):
        """
            Validates that the given masses are all definite positive, i.e., are
            all greater than zero.

            :raise ValueError: If there is a mass that is less than, or equal
            to, zero.
        """

        # Check if there are negative masses.
        if any(map(lambda x: x <= numpy.float64(0.0), self.masses)):
            raise ValueError(
                "There is a negative, or zero, mass present. All masses must "
                "be positive definite."
            )

    def _validate_radii(self):
        """
            Validates that the given radii are all definite positive, i.e., are
            all greater than zero.

            :raise ValueError: If there is a radius that is less than, or equal
            to, zero.
        """

        # Check if there are negative masses.
        if any(map(lambda x: x <= numpy.float64(0.0), self.radii)):
            raise ValueError(
                "There is a negative, or zero, radius present. All radii must "
                "be positive definite."
            )


if __name__ == "__main__":
    file_location = "../../data/product.csv"
    mol = Molecule(file_location)
