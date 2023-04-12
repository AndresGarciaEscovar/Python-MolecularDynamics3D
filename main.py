""" File that contains the DiffusionTensor class. """

# ##############################################################################
# Imports
# ##############################################################################
import os

import code.main.molecule as molecule

# ##############################################################################
# Main Function
# ##############################################################################


def main() -> None:
    """
        Runs the main program.
    """
    filename = os.getcwd() + "/data/product.csv"
    pnb = molecule.Molecule(filename)

    filename = os.getcwd() + "/data/reactant.csv"
    pnb44 = molecule.Molecule(filename)

    print(pnb)
    print(pnb44)


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == "__main__":
    """
        Runs the program.
    """
    main()
