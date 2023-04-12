# ##############################################################################
# Imports
# ##############################################################################

# General.
import copy
import numpy

# ##############################################################################
# Classes
# ##############################################################################


class Atom:
    """
        Class that represents a spherical atom.

        Parameters:
        ___________
        self.coordinates: numpy.ndarray
         A 1D numpy array of n-entries that represents the position of the
         sphere in n-dimensional space.

        self.mass: float
         A positive floating point number that represents the mass of the atom.

        self.name: str
         A string that represents the name of the atom. It can be changed at
         any time.

        self.radius: float
         A positive floating point number that represents the radius of the
         spherical atom.
    """

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Public Interface
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # //////////////////////////////////////////////////////////////////////////
    # Properties
    # //////////////////////////////////////////////////////////////////////////

    @property
    def coordinates(self) -> numpy.ndarray:
        """
            Returns a copy of the coordinates of the atom.

            :return: Returns a copy of the coordinates of the atom.
        """
        return copy.deepcopy(self.__coordinates)

    @coordinates.setter
    def coordinates(self, coordinates: numpy.ndarray) -> None:
        """
            Sets the coordinates of the atom.
        """

        # Set for the first time.
        if "_Atom__coordinates" in self.__dict__:
            raise ValueError(
                "The coordinates have already been set. If they must be "
                "changed, they must be changed as an attribute."
            )

        # Set the coordinates.
        self.__coordinates = numpy.array(coordinates, dtype=float)

    # ------------------------------------------------------------------------ #

    @property
    def dimensions(self) -> int:
        """
            Returns the number of coordinates needed to completely describe the
            position of an atom in space.

            :return: The number of coordinates needed to completely describe the
             position of an atom in space.
        """
        return len(self.coordinates)

    # ------------------------------------------------------------------------ #

    @property
    def mass(self) -> float:
        """
            Returns the floating point number that represents the mass of the
            atom.

            :return: Returns the floating point number that represents the mass
             of the atom.
        """
        return self.__mass

    @mass.setter
    def mass(self, mass: float) -> None:
        """
            Sets the mass of the atom.

            :param mass: The floating point number that represents the mass of
             the atom.
        """

        if "_Atom__mass" in self.__dict__:
            raise ValueError(
                "The mass exists, once the value is set it cannot be changed."
            )
        elif mass <= 0.0:
            raise ValueError(
                f"The requested mass to be set has a negative, or zero, value. "
                f"The mass is a positive, non-zero, value. Requested: {mass}"
            )

        self.__mass = float(f"{mass}")

    # ------------------------------------------------------------------------ #

    @property
    def radius(self) -> float:
        """
            Returns the floating point number that represents the radius of the
            atom.

            :return: Returns the floating point number that represents the
             radius of the atom.
        """
        return self.__radius

    @radius.setter
    def radius(self, radius: float) -> None:
        """
            Sets the radius of the atom.

            :param radius: The floating point number that represents the
             radius of the atom.
        """

        if "_Atom__radius" in self.__dict__:
            raise ValueError(
                "The radius exists, once the value is set it cannot be changed."
            )
        elif radius <= 0.0:
            raise ValueError(
                f"The requested radius to be set has a negative, or zero, "
                f"value. The radius is a positive, non-zero, value. Requested: "
                f"{radius}"
            )

        self.__radius = float(f"{radius}")

    # //////////////////////////////////////////////////////////////////////////
    # Constructor
    # //////////////////////////////////////////////////////////////////////////

    def __init__(self, radius: float, mass: float, coordinates: numpy.ndarray):
        """
            Constructs a new instance of an atom. Once created, the mass and the
            radius of the atom cannot be changed.

            :param radius: A positive floating point number that represents the
             radius of the spherical atom.

            :param mass: A positive floating point number that represents the
             mass of the atom.

            :param coordinates: A 1D numpy array of n-entries that represents
             the position of the sphere in n-dimensional space.
        """

        # Set the atom.
        self.name = "<unnamed>"

        # Set the other parameters.
        self.coordinates = coordinates
        self.mass = mass
        self.radius = radius

    # //////////////////////////////////////////////////////////////////////////
    # Dunder Methods
    # //////////////////////////////////////////////////////////////////////////

    def __len__(self):
        """
            Returns the number of coordinates needed to completely describe the
            position of an atom in space.

            :return: The number of coordinates needed to completely describe the
             position of an atom in space.
        """
        return self.dimensions

    def __repr__(self) -> str:
        """
            Returns a string with a quick representation of the atom, i.e.,
            the name of the atom, the coordinates, the radius and mass; in that
            specific order.

            :return: A string with a quick representation of the atom.
        """

        # Get the coordinates string.
        crds = self.coordinates
        crds = [tuple(['+' if x >= 0 else '-', abs(x)]) for x in crds]
        crds = "(" + ",".join([f"{x[0]}{x[1]:.7e}" for x in crds]) + ")"

        # Mass and radius string.
        mass = f"{self.mass:.7e}"
        radius = f"{self.radius:.7e}"

        # Set the values of the string.
        string = [self.name, crds, radius, mass]

        return "    ".join(string)

    def __str__(self) -> str:
        """
            Returns a string with a detailed representation of the atom.

            :return: A string with a detailed representation of the atom.
        """

        # Get the coordinates string.
        crds = self.coordinates
        crds = [tuple(['+' if x >= 0 else '-', abs(x)]) for x in crds]
        crds = "(" + ", ".join([f"{x[0]}{x[1]:.7e}" for x in crds]) + ")"

        # Mass and radius string.
        mass = f"{self.mass:.7e}"
        radius = f"{self.radius:.7e}"

        # Set the values of the string.
        string = [
            f"Name: {self.name}",
            f"Coordinates: {crds} \u212B",
            f"Radius: {radius} \u212B",
            f"Mass: {mass} AMU"
        ]

        return "\n".join(string)
