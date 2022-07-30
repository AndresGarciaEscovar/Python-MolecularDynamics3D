""" File that contains the DiffusionTensor class. """

# ##############################################################################
# Imports
# ##############################################################################
import os

import molecular_dynamics.main.molecule as molecule

# ##############################################################################
# Main Function
# ##############################################################################


def main() -> None:
    """
        Runs the main program.
    """
    filename = os.getcwd() + "/data/product.csv"
    pnb = molecule.Molecule(filename)


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == "__main__":
    """
        Runs the program.
    """
    main()
