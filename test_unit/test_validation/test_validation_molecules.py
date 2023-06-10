"""
    File that contains the unit test for the different validation functions for
    molecules.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General
import unittest

from numpy import zeros

# User defined.
import code.validation.validation_molecule as vmolecules

# ##############################################################################
# Classes
# ##############################################################################


class TestValidationMolecules(unittest.TestCase):

    def test_is_diffusion_tensor(self):
        """
            Tests that the function is_diffusion_tensor is working as intended.
        """
        # Test several dimensions.
        for i in range(1, 10):
            # Number of entries.
            dims = i if i == 1 else i + 1
            dims = 2 * i if i > 2 else dims

            # A zero tensor is fine.
            dtensor = zeros(dims**2, dtype=float).reshape((dims, dims))

            # Must not trow an error.
            vmolecules.is_diffusion_tensor(i, dtensor)

            # Wrond dimensions.
            with self.assertRaises(ValueError):
                vmolecules.is_diffusion_tensor(i + 1, dtensor)

        # Negative index or zero.
        dims = 3
        dtensor = zeros(dims ** 2, dtype=float).reshape((dims, dims))
        for i in (0, -1):
            with self.assertRaises(ValueError):
                vmolecules.is_diffusion_tensor(i, dtensor)

# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
