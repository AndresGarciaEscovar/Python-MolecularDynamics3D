"""
    File that contains the Molecule class and its methods.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import numpy as np
import yaml

from pathlib import Path
from typing import Any, Union

# User defined.
import code.main.atom as atom

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
    # Load an Save Methods
    # --------------------------------------------------------------------------

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
        # Set the file name and get the working directory.
        self.filename = f"{Path(filename).resolve()}"
        self.working = f"{Path(working).resolve()}"
        

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
