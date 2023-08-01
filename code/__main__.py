"""
    This file is used to run the code as a module.
"""

# ##############################################################################
# Imports
# ##############################################################################


# General.
import argparse

# User defined.
import code.validation.validation_parameters as vparameters

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

    print(args)


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == "__main__":
    """
        This is the main program.
    """
    run_main()
