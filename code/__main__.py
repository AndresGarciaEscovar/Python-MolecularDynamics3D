"""
    This file is used to run the code as a module.
"""

# ##############################################################################
# Imports
# ##############################################################################


# General.
import argparse
import numpy as np

# User defined.
import code.validation.validation_parameters as vparameters

# Temporary.
from code.main.atom import Atom

# ##############################################################################
# Functions
# ##############################################################################


def get_args() -> list:
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


# ##############################################################################
# Main Function
# ##############################################################################


def run_main() -> None:
    """
        This function is used to run the main program.
    """
    # Get the command line arguments.
    args = get_args()

    atom = Atom("C1", "C", 1.0, 12.0, np.array([1.0, 2.0, 3.0]))

    print(atom)
    


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == "__main__":
    """
        This is the main program.
    """
    run_main()
