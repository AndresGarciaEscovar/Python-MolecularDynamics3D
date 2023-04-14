"""
    File that contains the context manager to set the working directory and
    restore the state after exiting.
"""


# ##############################################################################
# Imports
# ##############################################################################

# General
import os
import copy

# ##############################################################################
# Classes
# ##############################################################################


class SetWD:
    """
        The context manager to setup the working directory.

        Parameters:
        __________

        self.newpath: str
         The new path to be set as the current working directory.

        self.oldpath: str
         The current working directory path.
    """

    # //////////////////////////////////////////////////////////////////////////
    # Dunder Methods
    # //////////////////////////////////////////////////////////////////////////

    def __enter__(self) -> str:
        """
            Sets the path to the desired path.

            :return: The name of the new path.
        """
        # Set the path.
        os.chdir(self.newpath)

        return self.newpath

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
            Exits the context manager, restoring the old path.

            :param exc_type: If there are exceptions, the exception types.

            :param exc_val: If there are exceptions, the exception values.

            :param exc_tb: If there are exceptions, the exception traceback.
        """

        # Set the old path back.
        os.chdir(self.oldpath)

    # //////////////////////////////////////////////////////////////////////////
    # Constructor
    # //////////////////////////////////////////////////////////////////////////

    def __init__(self, newpath: str):
        """
            Create the variables.

            :param newpath: The new path to be set as the current working
             directory.
        """

        # Save the new and old path.
        self.newpath = newpath
        self.oldpath = copy.deepcopy(os.getcwd())
