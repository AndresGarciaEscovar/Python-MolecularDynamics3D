"""
    File that contains several utility functions to calculate properties of
    molecules.
"""

# ##############################################################################
# Imports
# ##############################################################################


# General.
import numpy as np
import yaml

# User defined.
import code.validation.validation_parameters as vparameters


# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Get Functions
# ------------------------------------------------------------------------------


def get_cod(diffusion_tensor: np.ndarray) -> np.ndarray:
    """
        This function is used to get the center of diffusion from the diffusion
        tensor.

        :param diffusion_tensor: The diffusion tensor.

        :return: The center of diffusion, given the diffusion tensor.
    """
    drot = diffusion_tensor[0:3, 3:6]

    cod = np.array([
        diffusion_tensor[1, ],
    ])


def get_parameters(path: str) -> dict:
    """
        This function is used to get the simulation and molecule parameters.

        :return: A dictionary with the simulation and molecule parameters.
    """
    # Verify that the file exists and it is a YAML file.
    vparameters.is_yaml(path)

    # Get the dictionary of parameters from the yaml file.
    with open(path, "r") as file:
        parameters = yaml.safe_load(file)
    
    return parameters
