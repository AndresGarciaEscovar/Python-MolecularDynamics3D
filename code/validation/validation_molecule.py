"""
    File that contains the validation functions for different features of
    molecules.
"""


# ##############################################################################
# Imports
# ##############################################################################

# General.
from numpy import dtype, float64, ndarray
from typing import Any

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Is Functions
# ------------------------------------------------------------------------------


def is_unique_name(name: str, names: tuple) -> None:
    """
        Checks that the given name to an atom in the molecule is unique.

        :param name: The intended name of the atom.

        :param names: The current names of the atoms in the molecule.

        :raise ValueError: If the name is in the given tuple.
    """
    if name in names:
        raise ValueError(
            f"There is already a '{name}' string in the tuple: '{names}', in "
            f"this molecule: {name}. Please make sure all the atoms have "
            f"UNIQUE names."
        )
