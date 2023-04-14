"""
    File that contains the Molecule class and its methods.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import os
import numpy

from pathlib import Path
from typing import Any

# User defined.
import code.main.atom as atom

# import molecular_dynamics.main.diffusion_tensor as dt
# import molecular_dynamics.utilities.utilities_strings as us
# import molecular_dynamics.utilities.utilities_vector as uv

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
    # Properties
    # ##########################################################################

    @property
    def atoms(self) -> list:
        """
            Returns the list of atoms in the molecule.

            :return: The list of the atoms in the molecule.
        """
        return self.__atoms

    @atoms.setter
    def atoms(self, atoms: list) -> None:
        """
            Initially sets the list of atoms. Atoms to be added must be added
            using the add atom function.

            :param atoms: The initial list of atoms.
        """

        # Cannot change the atoms if they already exist.
        if "_Molecule__atoms" in self.__dict__:
            raise AttributeError(
                "The atoms can only be initialized once. If atoms must be "
                "added/removed/edited, it must be done through the add_atom or "
                "remove_atom functions."
            )

        self.__atoms = atoms

    # ------------------------------------------------------------------------ #

    @property
    def dimensions(self) -> int:
        """
            The number of coordinates used to described the position of a
            molecule in space.

            :return: The number of coordinates used to described the position of
             a molecule in space.
        """
        return self.atoms[0].dimensions

    # ------------------------------------------------------------------------ #

    @property
    def masses(self) -> numpy.ndarray:
        """
            Returns the masses  of all the atoms in the molecule.

            :return: The numpy array of the masses of the atoms in the molecule,
             in the order in which the atoms are stored.
        """
        return numpy.array([x.mass for x in self.atoms], dtype=float)

    # ------------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        """
            Returns the string that represents the name of the molecule.

            :return: The string that represents the name of the molecule.
        """
        return self.__name

    @name.setter
    def name(self, name: Any) -> None:
        """
            Sets the name of the molecule to the given strings.

            :param name: The object or string whose string representation is the
             name of the molecule.
        """

        # The molecule must have a name.
        if name is None or f"{name}".strip() == "":
            raise ValueError(
                "A non-empty name must be provided to the molecule."
            )

        self.__name = f"{name}"

    # ------------------------------------------------------------------------ #

    @property
    def names(self) -> tuple:
        """
            Returns the names of all the atoms in the molecule.

            :return: The tuple that contains all of the names of the atoms in
             the molecule, in the order in which the atoms are stored.
        """
        return tuple(x.name for x in self.atoms)

    # ------------------------------------------------------------------------ #

    @property
    def radii(self) -> numpy.ndarray:
        """
            Returns the radius of all the atoms in the molecule.

            :return: The numpy array of the radius of the atoms in the molecule,
             in the order in which the atoms are stored.
        """
        return numpy.array([x.radius for x in self.atoms], dtype=float)

    # ##########################################################################
    # Constructor
    # ##########################################################################

    def __init__(self, filename: str = None):
        """
            Constructs a new instance of the a molecule. If the name of the file
            is not provided, it will create a single-sphere molecule with a
            radius of 1.0 \u212B and mass of 1.0 AMU.

            :param filename: The name of the file from where the molecule must
             be loaded.
        """

        # Set the file name.
        self.filename = f"{Path(filename).resolve()}"
        self.name = "<Unnamed molecule>"

        # Create the basic quantities.
        self.atoms = list()

        # Get the center of mass.
        # self.com = um.get_com(self.atoms)
        # Get the center of diffusion, center of mass and center of geometry.

    # ##########################################################################
    # Dunder Methods
    # ##########################################################################

    def __len__(self):
        """
            Returns the number of atoms in the molecule, i.e., the length of
            the atoms array.

            :return: Number of atoms in the molecule.
        """
        return len(self.atoms)

    def __repr__(self):
        """
            Returns a string with a quick represenation of the molecule, i.e.,
            the current coordinate, radius and mass of each atom.
        """
        return ""

    def __str__(self):
        """
            Returns a more sophisticated string representation of the molecule
            to include a better looking table and more information such as the
            center of diffusion, center of geometry, center of mass and
            diffusion tensor; the latter with respect to the center of mass.
        """

        return ""

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Atom Methods
    # --------------------------------------------------------------------------

    def atom_add(
        self, radius: float, mass: float, coordinates: numpy.ndarray
    ) -> None:
        """
            Adds an atom to the atom list.

            :param radius: The floating point number that represents the radius
             of the atom.

            :param mass: The floating point number that represents the mass of
             the atom.

            :param coordinates: The numpy array of floating point numbers that
             that represent the coordinates of the atom.
        """
        self.atoms.append(
            atom.Atom(radius=radius, mass=mass, coordinates=coordinates)
        )

    def atom_remove(self, index: int) -> None:
        """
            Removes the atom at the given index.

            :param index: The index of the atom to be removed.
        """

        # Validate the index.
        if index < 0 or index >= len(self):
            raise ValueError(
                "The index of the given atom is out of range; i.e., must be a "
                f"number between 0 and {len(self)}. Requested index: {index}"
            )

        del self.atoms[index]

    # --------------------------------------------------------------------------
    # Load Methods
    # --------------------------------------------------------------------------

    def load(self, dimensions: int, filename: str) -> None:
        """
            Loads the given values.

            :param filename: The name of the file where the data is stored.
        """

        # Remove all the atoms.
        while len(self) > 0:
            self.atom_remove(0)

        # Load the molecule.
        with open(filename, mode='r', newline="\n") as molecule:
            # Load the molecule.
            self.name = molecule.readline().split(",")[1]

            # Number of atoms.
            natoms = int(molecule.readline().split(",")[1])

            # Load the atoms.
            for i in range(natoms):
                # Read the lines.
                line = molecule.readline().split(",")

                # Extract the values.
                coordinates = numpy.array(
                    [float(x) for x in line[1:dimensions + 1]], dtype=float
                )
                mass = float(line[dimensions + 2])
                radius = float(line[dimensions + 1])

                # Add the atom.
                self.atom_add(
                    radius=radius, mass=mass, coordinates=coordinates
                )

                # Label the atom.
                self.atoms[i].name = f"Atom {i + 1}"

    # --------------------------------------------------------------------------
    # Rotate Methods
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Translate Methods
    # --------------------------------------------------------------------------

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Private Interface
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Get Methods
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Translate Methods
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Validate Methods
    # --------------------------------------------------------------------------


# ##############################################################################
#
# ##############################################################################

if __name__ == "__main__":
    # Get the absolute path.
    mpath = f"{Path(os.getcwd(), '..', '..', 'data', 'product.txt').resolve()}"

    # Get the absolute path.
    molecule = Molecule(mpath)
