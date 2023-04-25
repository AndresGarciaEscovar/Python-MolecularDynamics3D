"""
    File that contains the Molecule class and its methods.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import os
import yaml

from numpy import array, ndarray
from pathlib import Path
from typing import Any

# User defined.
import code.main.atom as atom
import code.utilities.utilities_molecule as umolecule
import code.validation.validation_parameters as vparameters
import code.validation.validation_molecule as vmolecule

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
        vparameters.is_not_in_dictionary("_Molecule__atoms", self.__dict__)

        self.__atoms = atoms

    # ------------------------------------------------------------------------ #

    @property
    def coordinates(self) -> ndarray:
        """
            Returns the coordinates of all the atoms in the molecule.

            :return: The numpy array of the coordinates of the atoms in the
             molecule, in the order in which the atoms are stored.
        """
        return array([x.coordinates for x in self.atoms], dtype=float)

    # ------------------------------------------------------------------------ #

    @property
    def com(self) -> ndarray:
        """
            Returns the string that represents the name of the molecule.

            :return: The string that represents the name of the molecule.
        """
        return self.__com

    @com.setter
    def com(self, com: ndarray) -> None:
        """
            Sets the center of mass.

            :param com: The numpy array that represents the center of mass of
             the molecule.
        """
        self.__com = com

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
    def masses(self) -> ndarray:
        """
            Returns the masses  of all the atoms in the molecule.

            :return: The numpy array of the masses of the atoms in the molecule,
             in the order in which the atoms are stored.
        """
        return array([x.mass for x in self.atoms], dtype=float)

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
        # The molecule must have a non-empty name.
        vparameters.is_string(name, strip=True, empty=True)

        self.__name = name.strip()

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
    def radii(self) -> ndarray:
        """
            Returns the radius of all the atoms in the molecule.

            :return: The numpy array of the radius of the atoms in the molecule,
             in the order in which the atoms are stored.
        """
        return array([x.radius for x in self.atoms], dtype=float)

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Atom Methods
    # --------------------------------------------------------------------------

    def atom_add(
        self, aname: str, radius: float, mass: float, coordinates: ndarray,
        atype: str = None
    ) -> None:
        """
            Adds an atom to the atom list.

            :param radius: The floating point number that represents the radius
             of the atom.

            :param mass: The floating point number that represents the mass of
             the atom.

            :param coordinates: The numpy array of floating point numbers that
             that represent the coordinates of the atom.

            :param aname: The name of the atom.

            :param atype: The type of the atom; this is an optional parameter.
        """

        # Validate both strings, if needed.
        vparameters.is_string(aname, empty=False)
        if atype is not None:
            vparameters.is_string(atype, empty=False)

        # Format the strings properly.
        aname = "---" if aname.strip() == "" else aname
        atype = "---" if atype is None or atype.strip() == "" else atype

        # Check the atom names are unique.
        vmolecule.is_unique_name(aname, self.anames)

        # Create and add the atom.
        matom = atom.Atom(
            radius, mass, coordinates, atype.strip(), aname.strip()
        )
        self.atoms.append(matom)

    def atom_remove(self, index: int) -> None:
        """
            Removes the atom at the given index.

            :param index: The index of the atom to be removed.
        """

        # Validate the index.
        if index < 0 or index >= len(self):
            raise ValueError(
                "The index of the given atom is out of range; i.e., must be a "
                f"number between 0 and {len(self)}. Requested index: {index}."
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
            Loads the molecule from the self.filename variable.
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
            crds = array(iatom["coordinates"], dtype=float)
            mass = float(iatom["mass"])
            radius = float(iatom["radius"])

            # Add the atom to the system.
            self.atom_add(name, radius, mass, crds, atype)

    def save(self, path: str) -> None:
        """
            Saves the configuration of the molecule to a yaml file, with the
            given name.

            :param path: The path where to save the yaml file; must have a yaml
             extension for this to work.
        """

        # Auxiliary variables.
        indent = "    "

        # Validate it's a yaml file.
        path = path.strip()
        vparameters.is_yaml(path)

        # Ge the information of the atoms:
        with open(path, mode="w", newline="\n") as fl:
            # Write the file header.
            fl.writelines([
                "# Coordinates are in Angstrom.\n",
                "# Mass is in AMU.\n",
                "# Radius is in Angstrom.\n",
                "# Atom type (atype) is not a mandatory field; set to '---' by "
                "default.\n",
                "# The diffusion tensor must ALWAYS be given with respect to "
                "the center of mass.\n",
                f"molecule_name: {self.name}\n",
                f"atoms:\n",
            ])

            # For all the atoms.
            for matom in self.atoms:
                # Get a LIST of the coordinates.
                coordinates = ','.join([f"{float(x)}" for x in matom.coordinates])

                # Add the atom properties.
                fl.write(f"{indent * 1}{matom.aname}:\n")
                fl.write(f"{indent * 2}coordinates: [{coordinates}]\n")
                fl.write(f"{indent * 2}mass: {matom.mass}\n")
                fl.write(f"{indent * 2}radius: {matom.radius}\n")
                fl.write(f"{indent * 2}atype: {matom.atype}\n")

            # Add the diffusion tensor.
            fl.write("diffusion_tensor:\n")
            for row in self.dtensor:
                fl.write(f"{indent * 1}- [{','.join(list(map(str, row)))}]\n")

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
            radius of 1.0 Angstom (\u212B) and mass of 1.0 Atomic Mass Units
            AMU.

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

        # Get the center of mass of the molecule.
        self.cog = umolecule.get_cog(self.coordinates, self.radii)
        self.com = umolecule.get_com(self.coordinates, self.masses)

        # Calculate with respect to the center of mass.
        self.dtensor = umolecule.get_dtensor(
            self.coordinates, self.masses, -self.com
        )

        print(self.dtensor)
        print(self.coordinates)
        print(self.cog)
        print(self.com)

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

    # Path from where the molecules are loaded.
    mp0 = f"{Path(os.getcwd(), '..', '..', 'data', 'product.yaml').resolve()}"
    mp1 = f"{Path(os.getcwd(), '..', '..', 'data', 'reactant.yaml').resolve()}"

    # Path to where the molecules are saved.
    mp2 = f"{Path(os.getcwd(), '..', '..', 'data', 'product_1.yaml').resolve()}"
    mp3 = f"{Path(os.getcwd(), '..', '..', 'data', 'reactant_1.yaml').resolve()}"

    # Load using the absolute path.
    molecule0 = Molecule(mp0)
    molecule0.save("test.yaml")
    # molecule1 = Molecule(mp1)
