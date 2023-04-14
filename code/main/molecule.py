"""
    File that contains the Molecule class and its methods.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import os
import numpy as np
import yaml

from pathlib import Path
from typing import Any

# User defined.
import code.main.atom as atom
import code.utilities.utilities_molecule as umolecule

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
    def masses(self) -> np.ndarray:
        """
            Returns the masses  of all the atoms in the molecule.

            :return: The numpy array of the masses of the atoms in the molecule,
             in the order in which the atoms are stored.
        """
        return np.array([x.mass for x in self.atoms], dtype=float)

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

        self.__name = f"{name}".strip()

    # ------------------------------------------------------------------------ #

    @property
    def anames(self) -> tuple:
        """
            Returns the names of all the atoms in the molecule.

            :return: The tuple that contains all of the names of the atoms in
             the molecule, in the order in which the atoms are stored.
        """
        return tuple(x.aname for x in self.atoms)

    # ------------------------------------------------------------------------ #

    @property
    def radii(self) -> np.ndarray:
        """
            Returns the radius of all the atoms in the molecule.

            :return: The numpy array of the radius of the atoms in the molecule,
             in the order in which the atoms are stored.
        """
        return np.array([x.radius for x in self.atoms], dtype=float)

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Atom Methods
    # --------------------------------------------------------------------------

    def atom_add(
        self, radius: float, mass: float, coordinates: np.ndarray,
        aname: str = None, atype: str = None
    ) -> None:
        """
            Adds an atom to the atom list.

            :param radius: The floating point number that represents the radius
             of the atom.

            :param mass: The floating point number that represents the mass of
             the atom.

            :param coordinates: The numpy array of floating point numbers that
             that represent the coordinates of the atom.

            :param aname: The name of the atom; this is an optional parameter.

            :param atype: The type of the atom; this is an optional parameter.
        """

        # Make into strings.
        aname = "---" if aname is None else f"{aname}"
        atype = "---" if atype is None else f"{atype}"

        # Check the atom names are unique.
        if aname.strip() in self.anames:
            raise ValueError(
                f"There is already an atom with the name: '{aname}', in this "
                f"molecule: {self.name}. Please make sure all the atoms have "
                f"UNIQUE names."
            )

        # Add the atom.
        self.atoms.append(atom.Atom(radius, mass, coordinates, atype, aname))

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
    # Clean Methods
    # --------------------------------------------------------------------------

    def clean_atoms(self) -> None:
        """
            Removes all the atoms and leaves an empty list.
        """
        while len(self) > 0:
            self.atom_remove(0)

    # --------------------------------------------------------------------------
    # Load an Save Methods
    # --------------------------------------------------------------------------

    def load(self) -> None:
        """
            Loads the given values.

            :param filename: The name of the file where the data is stored.
        """

        # Remove all the atoms.
        self.clean_atoms()

        # Load the file.
        with open(self.filename, "r") as stream:
            info = yaml.safe_load(stream)

        # Get the molecule name.
        self.name = info["molecule_name"]

        # Load the atoms.
        for name, iatom in info["atoms"].items():
            # Extract the information.
            atype = f"{iatom['atype']}"
            crds = np.array(iatom["coordinates"], dtype=float)
            mass = float(iatom["mass"])
            radius = float(iatom["radius"])

            # Add the atom to the system.
            self.atom_add(radius, mass, crds, name, atype)

    def save(self, path: str) -> None:
        """
            Saves the configuration of the molecule to a yaml file, with the
            given name.

            :param path: The path where to save the yaml file; must have a yaml
             extension for this to work.
        """

        # Auxiliary variables.
        indent = "    "

        # Remove leading and trailing spaces.
        path = path.strip()

        # Check it is a yaml file.
        if not path.endswith(".yaml"):
            raise TypeError(
                f"The given file location: {path}, doesn't correspond to a "
                f"yaml file. The file must have a '.yaml' extension."
            )

        # Basic structure.
        temp = [
            "# Coordinates are in Angstrom.",
            "# Mass is in AMU.",
            "# Radius is in Angstrom.",
            "# Atom type (atype) is not a mandatory field; set to '---' by "
            "default.",
            f"molecule_name: {self.name}",
            f"atoms:",
        ]

        # Ge the information of the atoms:
        for matom in self.atoms:
            # Get a LIST of the coordinates.
            coordinates = ','.join([f"{float(x)}" for x in matom.coordinates])

            # Add the variables.
            temp.append(f"{indent * 1}{matom.aname}:")
            temp.append(f"{indent * 2}coordinates: [{coordinates}]")
            temp.append(f"{indent * 2}mass: {matom.mass}")
            temp.append(f"{indent * 2}radius: {matom.radius}")
            temp.append(f"{indent * 2}atype: {matom.atype}")

        # Write the file.
        with open(path, mode="w", newline="\n") as fl:
            for line in temp:
                fl.write(line + "\n")

    # --------------------------------------------------------------------------
    # Rotate Methods
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Translate Methods
    # --------------------------------------------------------------------------

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

        # Load the molecule.
        self.load()

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


# ##############################################################################
#
# ##############################################################################

if __name__ == "__main__":
    # Get the absolute path.
    mp0 = f"{Path(os.getcwd(), '..', '..', 'data', 'product.yaml').resolve()}"
    mp1 = f"{Path(os.getcwd(), '..', '..', 'data', 'reactant.yaml').resolve()}"

    mp2 = f"{Path(os.getcwd(), '..', '..', 'data', 'product_1.yaml').resolve()}"
    mp3 = f"{Path(os.getcwd(), '..', '..', 'data', 'reactant_1.yaml').resolve()}"

    tcoordinates = np.array([0, 0, 0], dtype=float)

    # Load using the absolute path.
    molecule0 = Molecule(mp0)
    molecule0.save(mp2)

    molecule1 = Molecule(mp1)
