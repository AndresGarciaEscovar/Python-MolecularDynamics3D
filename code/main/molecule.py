"""
    File that contains the Molecule class and its methods.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import numpy as np
import sys
import yaml

from pathlib import Path
from typing import Any, Union

# User defined.
import code.main.atom as atom
import code.utilities.utilities_molecule as umolecule
import code.validation.validation_parameters as vparameters

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

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Atom Methods
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Clean Methods
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Load Methods
    # --------------------------------------------------------------------------

    def load(self) -> None:
        """
            Loads the molecule from the file.
        """
        # Auxiliary variables.
        atoms = None
        
        # Get the parameters from the yaml file.
        parameters = umolecule.get_parameters(self.filename)

        # Get the molecule parameters.
        self.dtensor = np.array(parameters["diffusion_tensor"], dtype=float)
        self.orientation = np.array(parameters["orientation"], dtype=float)
        atoms = parameters["atoms"]

        # Load the atoms.
        for name, information in atoms.items():
            coords = np.array(information["coordinates"], dtype=float)
            atype = information["atype"]
            mass = information["mass"]
            radius = information["radius"]
            self.atoms.append(atom.Atom(name, atype, mass, radius, coords))
        
        # Validate the matrices.
        vparameters.is_matrix(self.dtensor, (6, 6), "diffusion tensor")
        vparameters.is_matrix(self.orientation, (3, 3), "orientation")

    def load_axes(self) -> None:
        """
            Loads the longest and shortest axes of the molecule.
        """
        # Get the longest and shortest axes.
        axes = umolecule.get_axes(self.atoms, step=1.0e-3)

        raise Exception("Continue here!")

    def load_centers(self) -> None:
        """
            Loads the different centers of the molecule.
        """
        self.cod = umolecule.get_cod(self.dtensor)
        self.cog = umolecule.get_cog(self.atoms)
        self.com = umolecule.get_com(self.atoms)

    # --------------------------------------------------------------------------
    # Rotate Methods
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Translate Methods
    # --------------------------------------------------------------------------

    # ##########################################################################
    # Constructor
    # ##########################################################################

    def __init__(self, filename: str, working: str):
        """
            Constructs a new instance of the a molecule. If the name of the file
            is not provided, it will create a single-sphere molecule with a
            radius of 1.0 Angstom (\u212B) and mass of 1.0 Atomic Mass Units
            AMU.

            :param filename: The name of the file from where the molecule must
             be loaded.
            
            :param working: The name of the directory where the simulation
             output is being saved.
        """
        # Initialize the molecule properties.
        self.dtensor = None
        self.orientation = None
        self.atoms = []

        # Molecule centers, cod = center of diffusion, cog = center of geometry
        # and com = center of mass.
        self.cod = None
        self.cog = None
        self.com = None

        self.longest_axis = None
        self.longest_axis_lenght = None

        self.shortest_axis = None
        self.shortest_axis_lenght = None

        # Set the file name and get the working directory.
        self.filename = f"{Path(filename).resolve()}"
        self.working = f"{Path(working).resolve()}"

        # Load the molecule.
        self.load()
        self.load_centers()

        # Load the long and short axes.
        self.load_axes()

    # ##########################################################################
    # Dunder Methods
    # ##########################################################################

    def __len__(self):
        """
            Returns the number of atoms in the molecule, i.e., the length of
            the atoms array.

            :return: Number of atoms in the molecule.
        """
        return 0

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
