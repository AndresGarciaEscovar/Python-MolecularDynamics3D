"""
    File that contains the Molecule class and its methods.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import copy
import numpy

# User defined.
import molecular_dynamics.main.diffusion_tensor as dt

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

        # Load the molecule from the given file.
        if filename is not None:
            self.load(filename)

        # Validate masses and radii.
        self._validate_masses()
        self._validate_radii()

        # Get the center of diffusion, center of mass and center of geometry.
        self.center_of_diffusion = numpy.array(
            [0.0 for _ in range(3)], dtype=dtype
        )
        self.center_of_mass = self._get_center_of_mass()
        self.center_of_geometry = self._get_center_of_geometry()

        # Translate to the center of mass.
        self._translate_to_center_of_mass()

        # Get the diffusion tensor and center of diffusion.
        dtens = dt.DiffusionTensor.get_diffusion_tensor
        self.dtensor = dtens(self.coordinates, self.radii)
        self.center_of_diffusion = self._get_center_of_diffusion()

        # Get the bounding radius.
        self.bounding_radius = self._get_bounding_radius()

    # ##########################################################################
    # Dunder Methods
    # ##########################################################################

    def __repr__(self):
        """
            Returns a string with a quick represenation of the molecule, i.e.,
            the current coordinate, radius and mass of each atom.
        """

        # Extract the variables.
        c = list(map(tuple, self.coordinates))
        m = self.masses
        r = self.radii

        # Create a string with the atom number, coordinate, radius and mass.
        string = [("Atom Number", "Coordinate (x, y, z)", "Radius", "Mass")]
        for i, e in enumerate(zip(c, r, m)):
            string.append(tuple([i + 1, *e]))

        # Create a string.
        string = "\n".join(map(str, string))

        return string

    def __str__(self):
        """
            Returns a more sophisticated string representation of the molecule
            to include a better looking table and more information such as the
            center of diffusion, center of geometry, center of mass and
            diffusion tensor; the latter with respect to the center of mass.
        """

        # //////////////////////////////////////////////////////////////////////
        # Auxiliary Functions
        # //////////////////////////////////////////////////////////////////////

        # --------------------------- Get Functions -------------------------- #

        def get_diffusion_tensor_0() -> str:
            """

            :return:
            """
            # TODO: CONTINUE HERE.

        def get_table_string_0() -> str:
            """
                Gets the string that represents the table.

                :return: The string that represents the table.
            """

            # Start with an empty string.
            string_0 = ""

            # Write to the file.
            for i_0, entry_0 in enumerate(string):
                string_0 += "   ".join(entry_0) + "\n"

            return string_0

        def get_widths_0() -> tuple:
            """
                Gets a tuple with the widths of the column.

                :return: A tuple with the maximum width of the columns.
            """

            # The list that contains the widths.
            widths_0 = []

            # Go through each row.
            for i_0, strng_0 in enumerate(string):
                # Get the width of the rows.
                if i_0 == 0:
                    widths_0 = list(map(lambda x_0: len(f"{x_0}"), strng_0))
                    continue

                # Get the widths.
                widths_0 = list(
                    map(
                        lambda x_0, y_0: max(len(f"{x_0}"), y_0),
                        strng_0, widths_0
                    )
                )

            return tuple(widths_0)

        # ------------------------- Format Functions ------------------------- #

        def format_entries_0() -> None:
            """
                Formats the coordinates to reflect a reasonable value.

                :return: Formats coordinates, so they all show, at most, 7
                 significant figures.
            """

            # For each entry.
            for i_0, element_0 in enumerate(string):
                # Ignore the first element
                if i_0 == 0:
                    continue

                # Format the atom number.
                string[i_0][0] = f"{int(string[i_0][0])}"

                # Format the coordinates.
                coords_0 = str(tuple(f"{x_0:+.8e}" for x_0 in element_0[1]))
                string[i_0][1] = coords_0

                # Format the radii.
                string[i_0][2] = f"{string[i_0][2]:.8e}"

                # # Format the masses.
                string[i_0][3] = f"{string[i_0][3]:.8e}"

        def format_entries_fix_0() -> None:
            """
                Formats the strings so that they have a fixed length.

                :return: Fix the strings for them to have the proper length.
            """

            # Fix the entries.
            for i_0, entry0 in enumerate(string):
                if i_0 == 0:
                    string[i_0] = [
                        f"{x_0:^{y_0}}" for x_0, y_0 in zip(entry0, widths)
                    ]
                    continue

                string[i_0] = [
                    f"{x_0:<{y_0}}" for x_0, y_0 in zip(entry0, widths)
                ]

        # //////////////////////////////////////////////////////////////////////
        # Implementation
        # //////////////////////////////////////////////////////////////////////

        # Extract the variables.
        c = list(map(tuple, self.coordinates))
        m = self.masses
        r = self.radii

        # Create a string with the atom number, coordinate, radius and mass.
        string = [
            (
                "#",
                "Coordinate (x, y, z) (Angstrom)", "Radius (Angstroms)",
                "Mass (AMUs)"
            )
        ]
        for i, e in enumerate(zip(c, r, m)):
            string.append(list([i + 1, *e]))

        # Format the entries as strings.
        format_entries_0()
        widths = list(get_widths_0())

        # Fix the strings.
        format_entries_fix_0()

        return get_table_string_0()

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

    def _get_center_of_diffusion(self) -> numpy.ndarray:
        """
            From the masses and the positions, gets the center of mass.

            :return: The center of mass of the molecule.
        """

        # Get the appropriate tensors.
        rr = self.dtensor[3:, 3:]
        tr = self.dtensor[3:, :3]

        # Matrix with rotation-rotation coupling.
        matrix = numpy.linalg.inv(numpy.array(
             [
                 [rr[1, 1] + rr[2, 2], -rr[0, 1], -rr[0, 2]],
                 [-rr[0, 1], rr[0, 0] + rr[2, 2], -rr[1, 2]],
                 [-rr[0, 2], -rr[1, 2], rr[0, 0] + rr[1, 1]]
             ], dtype=numpy.float64
        ))

        # The vector with the assymetric translation-rotation entries.
        vector = numpy.array(
            [
                [tr[1, 2] - tr[2, 1]],
                [tr[2, 0] - tr[0, 2]],
                [tr[0, 1] - tr[1, 0]]
            ], dtype=numpy.float64
        )

        return numpy.transpose(numpy.matmul(matrix, vector))[0]

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

    def _translate_to_center_of_diffusion(self) -> None:
        """
            Translates the molecule to the center of diffusion.
        """

        # Translate the coordinates to the center of diffusion.
        for i, _ in enumerate(self.coordinates):
            self.coordinates[i] -= self.center_of_diffusion

        # Now, the center of geometry, center of mass and center of diffusion.
        self.center_of_mass -= self.center_of_diffusion
        self.center_of_geometry -= self.center_of_diffusion
        self.center_of_diffusion -= self.center_of_diffusion

    def _translate_to_center_of_geometry(self) -> None:
        """
            Translates the molecule to the center of geometry.
        """

        # Translate the coordinates to the center of geomery.
        for i, _ in enumerate(self.coordinates):
            self.coordinates[i] -= self.center_of_geometry

        # Now, the center of geometry, center of mass and center of diffusion.
        self.center_of_mass -= self.center_of_geometry
        self.center_of_diffusion -= self.center_of_geometry
        self.center_of_geometry -= self.center_of_geometry

    def _translate_to_center_of_mass(self) -> None:
        """
            Translates the molecule to the center of mass.
        """

        # Translate the coordinates to the center of mass.
        for i, _ in enumerate(self.coordinates):
            self.coordinates[i] -= self.center_of_mass

        # Now, the center of geometry, center of mass and center of diffusion.
        self.center_of_geometry -= self.center_of_mass
        self.center_of_diffusion -= self.center_of_mass
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

    print(str(mol))
