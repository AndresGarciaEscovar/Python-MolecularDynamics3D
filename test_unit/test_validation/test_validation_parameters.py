"""
    File that contains the unit test for the different validation parameters.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General
import unittest

from numpy import array, dtype, float64, ndarray

# User defined.
import code.validation.validation_parameters as vparameters

# ##############################################################################
# Global Constants
# ##############################################################################

ITERATIONS = 100

# ##############################################################################
# Classes
# ##############################################################################


class TestValidationParameters(unittest.TestCase):

    def test_is_float(self):
        """
            Tests that the function is_float is working as intended.
        """
        # Must not throw an exception.
        number = float(1.0)
        vparameters.is_float(number)

        # Must not throw an exception.
        number = int(1.0)
        vparameters.is_float(number)

        # Must not throw an exception.
        number = dtype("float64").type(1.0)
        vparameters.is_float(number)

        # Check other types.
        ctypes = (str, complex)

        for ttype in ctypes:
            number = ttype(1.0)
            with self.assertRaises(TypeError):
                vparameters.is_float(number)

    def test_is_ndarray(self):
        """
            Tests that the function is_ndarray is working as intended.
        """
        # Always test in this order.
        self.test_is_ndarray_type()
        self.test_is_ndarray_dtype()
        self.test_is_ndarray_length()

    def test_is_ndarray_dtype(self):
        """
            Tests that the function is_ndarray_dtype is working as intended.
        """
        # Must not throw an exception.
        number = array([1.0], dtype=float)
        vparameters.is_ndarray_dtype(number, dtype("float").type)

        # Must not throw an exception.
        number = array([1.0], dtype=float64)
        vparameters.is_ndarray_dtype(number, dtype("float64").type)

        # Must not throw an exception.
        number = array([1.0], dtype=int)
        vparameters.is_ndarray_dtype(number, dtype("int").type)

        # Must throw an exception.
        number = array([1.0], dtype=float)
        with self.assertRaises(TypeError):
            vparameters.is_ndarray_dtype(number, dtype("int").type)

        # Must throw an exception.
        number = array([1.0], dtype=int)
        with self.assertRaises(TypeError):
            vparameters.is_ndarray_dtype(number, dtype("float").type)

    def test_is_ndarray_length(self):
        """
            Tests that the function is_ndarray_length is working as intended.
        """
        # For the different cases.
        for length in range(1, 5):
            # Must not throw an exception.
            number = array([1.0 for _ in range(length)], dtype=float)
            vparameters.is_ndarray_length(number, length)

            # Must throw an exception.
            with self.assertRaises(ValueError):
                vparameters.is_ndarray_length(number, 5 - length)

    def test_is_ndarray_type(self):
        """
            Tests that the function is_ndarray_type is working as intended.
        """
        # For the different cases.
        for atype in (tuple, list):
            # Must not throw an exception.
            number = atype([i for i in range(5)])
            with self.assertRaises(TypeError):
                vparameters.is_ndarray_type(number)

            number = array(number, dtype=float)
            vparameters.is_ndarray_type(number)

    def test_is_negative(self):
        """
            Tests that the function is_negative is working as intended.
        """
        # Must throw a type error.
        with self.assertRaises(TypeError):
            vparameters.is_negative("-1.0")

        # Must throw a type error.
        with self.assertRaises(TypeError):
            vparameters.is_negative(complex(-1.0))

        # Must throw a type error.
        for atype in (list, ndarray, set, tuple):
            with self.assertRaises(TypeError):
                vparameters.is_negative(atype([-1.0, 1.0]))

        # Must throw value errors.
        for value in [int(1), int(0), float(1.0), float(0.0)]:
            with self.assertRaises(ValueError):
                vparameters.is_negative(value, include=False)

        # Must not throw errors.
        for value in [int(0),int(-1), float(0.0), float(-1.0)]:
            vparameters.is_negative(value, include=True)

    def test_is_not_in_dictionary(self):
        """
            Tests that the function is_not_in_dictionary is working as intended.
        """
        # Must throw an attribute error.
        dictionary = dict((("one", 1), ("two", 2), ("three", 3)))
        for key in dictionary.keys():
            with self.assertRaises(AttributeError):
                vparameters.is_not_in_dictionary(key, dictionary)

        # Must not throw an error.
        vparameters.is_not_in_dictionary("four", dictionary)

    def test_is_not_none(self) -> None:
        """
            Tests that the function is_not_none is working as intended.
        """
        # Must throw a type error.
        with self.assertRaises(TypeError):
            vparameters.is_not_none(None)

        # Must not throw an error; cases of interest.
        vparameters.is_not_none(complex(1.0))
        vparameters.is_not_none(str(""))
        vparameters.is_not_none(float(1.0))
        vparameters.is_not_none(int(1.0))

        vparameters.is_not_none(array([], dtype=float))
        vparameters.is_not_none(list())
        vparameters.is_not_none(tuple())

    def test_is_positive(self):
        """
            Tests that the function is_positive is working as intended.
        """
        # Must throw a type error.
        with self.assertRaises(TypeError):
            vparameters.is_positive("1.0")

        # Must throw a type error.
        with self.assertRaises(TypeError):
            vparameters.is_positive(complex(1.0))

        # Must throw a type error.
        for atype in (list, ndarray, set, tuple):
            with self.assertRaises(TypeError):
                vparameters.is_positive(atype([-1.0, 1.0]))

        # Must throw value errors.
        for value in [int(-1), int(0), float(-1.0), float(0.0)]:
            with self.assertRaises(ValueError):
                vparameters.is_positive(value, include=False)

        # Must not throw errors.
        for value in [int(0),int(1), float(0.0), float(1.0)]:
            vparameters.is_positive(value, include=True)

    def test_is_not_in_dict(self):
        """
            Tests that the function is_not_in_dict is working as intended.
        """
        # Tuple from where to form the dictionary.
        dtuple = (("one", 1), ("two", 2), ("three", 3))

        # Create a dictionary.
        dictionary = dict(dtuple)

        # Must throw exceptions.
        for key in dictionary.keys():
            with self.assertRaises(AttributeError):
                vparameters.is_not_in_dictionary(key, dictionary)

        # Must NOT throw execptions.
        key = list(dictionary.keys())[0] + "w"
        vparameters.is_not_in_dictionary(key, dictionary)

    def test_is_shape_matrix(self):
        """
            Tests that the function is_string is working as intended.
        """

        # Define a matrix.
        matrix = [3, 1, 3]

        # Must be a numpy array.
        with self.assertRaises(TypeError):
            vparameters.is_shape_matrix(matrix, (0, 3))

        # Check for the wrong dimensions.
        matrix = array([3, 1, 3, 9], dtype=float)

        # Must be a numpy array.
        with self.assertRaises(ValueError):
            vparameters.is_shape_matrix(matrix, (3,))

        # Check for the right dimensions.
        vparameters.is_shape_matrix(matrix, (4,))

        # Last test case.
        matrix = array([[3, 1, 3, 9], [4, 7, 2, 6]], dtype=float)

        with self.assertRaises(ValueError):
            vparameters.is_shape_matrix(matrix, (4, 2))
        vparameters.is_shape_matrix(matrix, (2, 4))

    def test_is_string(self):
        """
            Tests that the function is_string is working as intended.
        """

        # Must not throw a value error.
        vparameters.is_string(str("aaa"), strip=True, empty=True)
        vparameters.is_string(str("aaa"), strip=False, empty=True)
        vparameters.is_string(str(" "), strip=True, empty=False)
        vparameters.is_string(str(" "), strip=False, empty=False)
        vparameters.is_string(str(""), strip=True, empty=False)
        vparameters.is_string(str(""), strip=False, empty=False)

        # Must throw a value error.
        with self.assertRaises(ValueError):
            vparameters.is_string(str(" "), strip=True, empty=True)

        # Must throw a value error.
        vparameters.is_string(str(" "), strip=False, empty=True)
        with self.assertRaises(ValueError):
            vparameters.is_string(str(" "), strip=True, empty=True)

        # Must throw a value error.
        with self.assertRaises(ValueError):
            vparameters.is_string(str(""), strip=False, empty=True)

        # Must not throw an error; cases of interest.
        with self.assertRaises(TypeError):
            vparameters.is_string(None)

        with self.assertRaises(TypeError):
            vparameters.is_string(complex(1.0))

        with self.assertRaises(TypeError):
            vparameters.is_string(float(1.0))

        with self.assertRaises(TypeError):
            vparameters.is_string(int(1.0))

        with self.assertRaises(TypeError):
            vparameters.is_string(array([], dtype=float))

        with self.assertRaises(TypeError):
            vparameters.is_string(list())

        with self.assertRaises(TypeError):
            vparameters.is_string(tuple())


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
#