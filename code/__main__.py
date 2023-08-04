"""
    This file is used to run the code as a module.
"""

# ##############################################################################
# Imports
# ##############################################################################


# General.
import argparse
import numpy as np
import yaml

# User defined.
import code.validation.validation_parameters as vparameters

# Temporary.
from code.main.molecule import Molecule

# ##############################################################################
# Functions
# ##############################################################################


def get_args() -> str:
    """
        Returns function is used to get the command line arguments.
    """

    # Create the parser.
    parser = argparse.ArgumentParser(
        description="Program that runs a molecular dynamics simulation.",
    )

    # Add the arguments.
    parser.add_argument(
        "filename",
        type=str,
        help="The full path of the file containing the simulation parameters.",
    )

    # Parse the arguments.
    args = parser.parse_args()

    # Validate the yaml file path.
    vparameters.is_yaml(args.filename)

    # Return the arguments.
    return args.filename


def get_parameters(filename: str) -> tuple:
    """
        This function is used to get the simulation and molecule parameters.

        :param filename: The full path of the yaml file containing the
         simulation and molecule parameters.

        :return: The simulation and molecule parameters.
    """
    # Get the dictionary of parameters from the yaml file.
    with open(filename, "r") as file:
        parameters = yaml.safe_load(file)
    
    # Get the simulation parameters.    
    working = parameters["directory"]["working"]
    simulation = parameters["simulation"]
    molecules = tuple(value for value in parameters["molecule"].values())
    
    return simulation, molecules, working


# ##############################################################################
# Main Function
# ##############################################################################


def run_main() -> None:
    """
        This function is used to run the main program.
    """
    # Get the command line arguments.
    args = get_args()

    # Get the molecule and simulation parameters from the yaml file.
    simulation, molecules, working = get_parameters(args)

    molecule_0 = Molecule(molecules[0], working)
    molecule_1 = Molecule(molecules[1], working)


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == "__main__":
    """
        This is the main program.
    """
    run_main()
