"""
    File that contains the tests for the context manager to temporarily change
    directories.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import os
import pathlib
import unittest

# User defined.
import code.managers.context_manager_swd as swd

# ##############################################################################
# Classes
# ##############################################################################


class TestContexManagerSWD(unittest.TestCase):

    def test_change_working_directory(self):
        """
            Tests that the SetWD context manager is working properly.
        """

        # Loop through the different types of arrays.
        old_working = os.getcwd()

        # Get the path of the previous working directory.
        new_working = f"{pathlib.Path(old_working, '..').resolve()}"

        # Make sure the paths are different.
        self.assertNotEqual(old_working, new_working)

        # Test the context manager.
        with swd.SetWD(new_working) as fl:
            self.assertEqual(fl, new_working)
            self.assertEqual(fl, os.getcwd())

        # Make sure the path is restored.
        self.assertEqual(os.getcwd(), old_working)

        # Invalid new path.
        new_working = "asaslnsandaibajs"
        with self.assertRaises(FileNotFoundError):
            with swd.SetWD(new_working) as _:
                pass

# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
