"""
    File that contains the validation functions for different features of
    parameters.
"""


# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Miscellaneous Functions
# ------------------------------------------------------------------------------


def exists_in_dict(attribute: str, dictionary: dict) -> None:
    """
        Checks that the given parameter exists in the dictionary, otherwise it
        raises an error.

        :param attribute: The attribute name to be checked.

        :param dictionary: The dictionary that potentiall contains the
         parameter.

        :raise AttributeError: If the attribute is not in the given dictionary.
    """
    if attribute in dictionary:
        raise AttributeError(
            f"The attibute '{attribute}' already exists and can only be "
            f"initialized once."
        )
