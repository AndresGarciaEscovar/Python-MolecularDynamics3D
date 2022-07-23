"""
    File that contains the Molecule class and its methods.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import copy
import numpy

from numpy import ndarray, float64

# User defined.
import molecular_dynamics.main.diffusion_tensor as dt
import molecular_dynamics.utilities.utilities_molecule as um
import molecular_dynamics.utilities.utilities_strings as us
import molecular_dynamics.utilities.utilities_vector as uv


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

        # Create the basic quantities.
        self.atoms = 1
        self.boundingr = 0.5
        self.coordinates = numpy.zeros((1, 3), dtype=float64)
        self.masses = numpy.zeros((1, 1), dtype=float64)[0] + 1
        self.orientation = numpy.identity(3, dtype=float64)
        self.radii = numpy.zeros((1, 1), dtype=float64)[0] + 0.5

        # Load the molecule from the given file.
        if filename is not None:
            self.load(filename)

        # Validate masses and radii.
        self._validate_masses()
        self._validate_radii()

        # Get the center of diffusion, center of mass and center of geometry.
        self.cod = numpy.zeros((1, 3), dtype=float64)[0]
        self.com = um.get_center_of_mass(self.coordinates, self.masses)
        self.cog = um.get_center_of_geometry(self.coordinates, self.radii)

        # Translate to the center of mass.
        self._translate_to_com()

        # Get the diffusion tensor and center of diffusion.
        dtens = dt.DiffusionTensor.get_diffusion_tensor
        self.dtensor = dtens(self.coordinates, self.radii)
        self.cod = um.get_center_of_diffusion(self.dtensor)

        # Get the bounding radius.
        self.boundingr = self._get_boundingr()

    # ##########################################################################
    # Dunder Methods
    # ##########################################################################

    def __repr__(self):
        """
            Returns a string with a quick represenation of the molecule, i.e.,
            the current coordinate, radius and mass of each atom.
        """
        return us.get_string_molecule(self.coordinates, self.radii, self.masses)

    def __str__(self):
        """
            Returns a more sophisticated string representation of the molecule
            to include a better looking table and more information such as the
            center of diffusion, center of geometry, center of mass and
            diffusion tensor; the latter with respect to the center of mass.
        """

        # Name of the molecule.
        mname = "Molecule name: " + self.name + "\n"

        # Get the atom information.
        minf = us.get_string_molecule(self.coordinates, self.radii, self.masses)
        minf += "\n"

        # Get the center information.
        ci = f"Center of mass (x,y,z): {us.get_string_array(self.com)}\n"
        ci += f"Center of geometry (x,y,z): {us.get_string_array(self.cog)}\n"
        ci += f"Center of diffusion (x,y,z): {us.get_string_array(self.cod)}\n"

        # Get the diffusion tensor.
        dtensor = "Diffusion tensor in the center of mass along, the body-fixed"
        dtensor += " axes, x', y' and z':\n"
        dtensor += us.get_string_matrix(self.dtensor) + "\n"

        # Get the orientation.
        orientation = "Orientation of the x', y', z' axes:\n"
        orientation += f"\tx': {us.get_string_array(self.orientation[0])}\n"
        orientation += f"\ty': {us.get_string_array(self.orientation[1])}\n"
        orientation += f"\tz': {us.get_string_array(self.orientation[2])}\n"

        return mname + minf + ci + dtensor + orientation

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
        self.coordinates = numpy.array(coordinates, dtype=float64)
        self.masses = numpy.array(masses, dtype=float64)
        self.radii = numpy.array(radii, dtype=float64)

    # --------------------------------------------------------------------------
    # Rotate Methods
    # --------------------------------------------------------------------------

    def rotate_wr_cod(
            self, axis: ndarray, angle: float64, rads: bool = True
    ) -> None:
        """
            Rotates the molecule, with respect to the center of geometry, about
            the given axis, the given angle; where the default measure for
            the angle is in radians.

            :param axis: The axis about which the molecule should be rotated.

            :param angle: The angle about which the molecule will be rotated.

            :param rads: Boolean flag that indicates the units in which the
             angle is given. True, if the angle is given in radians; False,
             if the angle is given in degrees. No other unit of angle
             measurement is supported.
        """
        # TODO: WRITE THIS FUNCTION, ROTATIONS TO ORIENTATION FIRST.

    def rotate_wr_com(
            self, axis: ndarray, angle: float64, rads: bool = True
    ) -> None:
        """
            Rotates the molecule, with respect to the center of mass, about
            the given axis, the given angle; where the default measure for
            the angle is in radians.

            :param axis: The axis about which the molecule should be rotated.

            :param angle: The angle about which the molecule will be rotated.

            :param rads: Boolean flag that indicates the units in which the
             angle is given. True, if the angle is given in radians; False,
             if the angle is given in degrees. No other unit of angle
             measurement is supported.
        """
        # TODO: WRITE THIS FUNCTION, CONSIDERING ROTATIONS TO ORIENTATION.

    # --------------------------------------------------------------------------
    # Translate Methods
    # --------------------------------------------------------------------------

    def translate_to(self, vector: ndarray, check: bool = False) -> None:
        """
            Translates the molecule by the given vector.

            :param vector: The 3D vector by which the molecule should be
             translated.

            :param check: Boolean flag that indicates whether the translation
             vector should be checked for the dimensionality. True, if the
             vector should be checked; False, otherwise and set to as the
             default.
        """

        # Validate the translation and abort, if needed.
        um.validate_array(vector, exception=True) if check else None

        # Translate the coordinates to the center of mass.
        for i, _ in enumerate(self.coordinates):
            self.coordinates[i] += vector

        # Translate the center of geometry, mass and diffusion.
        self.cog += vector
        self.com += vector
        self.cod += vector

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Private Interface
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Get Methods
    # --------------------------------------------------------------------------

    def _get_boundingr(self) -> float64:
        """
            Gets the bounding radius of the molecule, with respect to the center
            of mass.

            :return: The sphere with the shortest radius that encloses the
             molecule, with respect to the center of mass.
        """

        # Auxiliary variables.
        bradius = float64(0.0)

        # Translate the molecule, temporarily, to the center of mass.
        translation = copy.deepcopy(self.com)
        self.translate_to(-translation)

        # Get the maximum bounding radius.
        for c, r in zip(self.coordinates, self.radii):
            bradius = float64(max(bradius, numpy.linalg.norm(c) + r))

        # Translate back to the original center of mass.
        self.translate_to(translation)

        return float64(bradius)

    # --------------------------------------------------------------------------
    # Translate Methods
    # --------------------------------------------------------------------------

    def _translate_to_cod(self) -> None:
        """
            Translates the molecule to the center of diffusion.
        """

        # Translate the coordinates to the center of diffusion.
        for i, _ in enumerate(self.coordinates):
            self.coordinates[i] -= self.cod

        # Now, the center of geometry, center of mass and center of diffusion.
        self.com -= self.cod
        self.cog -= self.cod
        self.cod -= self.cod

    def _translate_to_cog(self) -> None:
        """
            Translates the molecule to the center of geometry.
        """

        # Translate the coordinates to the center of geomery.
        for i, _ in enumerate(self.coordinates):
            self.coordinates[i] -= self.cog

        # Now, the center of geometry, center of mass and center of diffusion.
        self.com -= self.cog
        self.cod -= self.cog
        self.cog -= self.cog

    def _translate_to_com(self) -> None:
        """
            Translates the molecule to the center of mass.
        """

        # Translate the coordinates to the center of mass.
        for i, _ in enumerate(self.coordinates):
            self.coordinates[i] -= self.com

        # Now, the center of geometry, center of mass and center of diffusion.
        self.cog -= self.com
        self.cod -= self.com
        self.com -= self.com

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
        if any(map(lambda x: x <= float64(0.0), self.masses)):
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

        # Check if there are negative radii.
        if any(map(lambda x: x <= float64(0.0), self.radii)):
            raise ValueError(
                "There is a negative, or zero, radius present. All radii must "
                "be positive definite."
            )


if __name__ == "__main__":
    file_location = "../../data/product.csv"
    mol = Molecule(file_location)

    xs = numpy.array([0, 0, 1], dtype=float64)
    ng = numpy.pi * 0.5

    # print(str(mol))
    mol.rotate_wr_com(xs, ng)
    # print(str(mol))
