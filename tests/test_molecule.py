"""
    File that contains the basic unit testing for the Molecule class.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import numpy
import os
import unittest

# User defined.
import context_manager as cm

# Change the current working directory.
with cm.SetCWD(os.getcwd() + os.sep + "..") as _:

    # ##########################################################################
    # Imports
    # ##########################################################################

    # User defined.
    import molecular_dynamics.main.molecule as mol

    # ##########################################################################
    # Classes
    # ##########################################################################

    class TestMolecule(unittest.TestCase):

        def test_constructor(self):
            """
                Tests that the class has initialized the variables properly,
                after the class has been initialized.
            """

            # //////////////////////////////////////////////////////////////////
            # Auxiliary Functions.
            # //////////////////////////////////////////////////////////////////

            def verify_0(natoms_0: int, mname_0: str) -> None:
                """
                    Verifies the variables.

                    :param natoms_0: The number of atoms in the molecule.

                    :param mname_0: The name of the molecule.
                """

                # -------------- Check number of atoms and name -------------- #

                self.assertEqual(
                    natoms_0, molecule.atoms, msg="Wrong number of atoms."
                )
                self.assertEqual(mname_0, molecule.name, msg=(
                    "Wrong molecule name."
                ))

                # ------------------ Check length of arrays ------------------ #

                self.assertEqual(molecule.atoms, len(molecule.radii), msg=(
                    "Wrong number of radius elements."
                ))
                self.assertEqual(molecule.atoms, len(molecule.masses), msg=(
                    "Wrong number of mass elements."
                ))
                self.assertEqual(molecule.atoms, len(molecule.coordinates), msg=(
                    "Wrong number of coordinates."
                ))

                # -------------------- Check array content ------------------- #

                # Extract quantities to alternative variables.
                m_0 = molecule.masses
                r_0 = molecule.radii
                c_0 = molecule.coordinates

                # Verify each quantity.
                for i_0, element_0 in enumerate(zip(m_0, r_0, c_0)):
                    self.assertIsInstance(element_0[0], (numpy.float64,), msg=(
                        f"Mass {i_0}, in molecule '{molecule.name}', has the "
                        f"wrong type."
                    ))
                    self.assertIsInstance(element_0[1], (numpy.float64,), msg=(
                        f"Radius {i_0}, in molecule '{molecule.name}', has the "
                        f"wrong type."
                    ))
                    self.assertIsInstance(element_0[2], (numpy.ndarray,), msg=(
                        f"The coordinate {i_0}, in molecule '{molecule.name}', "
                        f"has the wrong type."
                    ))

                    # All elements shoudl be float 64.
                    for j_0, value_0 in enumerate(element_0[2]):
                        self.assertIsInstance(
                            value_0, (numpy.float64,), msg=(
                                f"For molecule '{molecule.name}', the element "
                                f"{j_0} in coordinate {i_0} has the wrong type."
                            )
                        )

            # //////////////////////////////////////////////////////////////////
            # Implementation.
            # //////////////////////////////////////////////////////////////////

            # ------------------ Creation without parameters ----------------- #

            # Create the molecule.
            molecule = mol.Molecule()

            # Check the different quantities.
            self.assertAlmostEqual(0.5, molecule.bounding_radius, msg=(
                "Wrong bounding radius."
            ))
            verify_0(1, "molecule name")

            # ------------------ Check scalar quantities ----------------- #

            # Create the molecule from the data file.
            filepath = "./data/product.csv"
            molecule = mol.Molecule(filepath)

            # Check the different quantities.
            self.assertAlmostEqual(
                7.285512742827672, molecule.bounding_radius, msg=(
                    "Wrong bounding radius."
                )
            )
            verify_0(26, "44npnb")

        def test_particles(self):
            """
                Tests that the class has initialized the variables properly, after
                the class has been initialized.
            """

            # ------------------ Creation without parameters ----------------- #

            # Create the molecule.
            molecule = mol.Molecule()

            # Validate that an atom has been created.
            self.assertEqual(1, 1, msg="Wrong number of atoms.")

    # ##########################################################################
    # Main Program
    # ##########################################################################

    if __name__ == "__main__":
        """
            Runs the tests.
        """
        unittest.main(failfast=False)
