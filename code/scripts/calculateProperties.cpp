/**
* Test to integrate C++ with Python.
*/


// *****************************************************************************
// Imports
// *****************************************************************************


#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


// *****************************************************************************
// Global Variables
// *****************************************************************************


// -----------------------------------------------------------------------------
// Constant Functions
// -----------------------------------------------------------------------------


// Expected number of arguments.
#define EXPECTED 4


// *****************************************************************************
// Function Prototypes
// *****************************************************************************


// ----------------------------- Read Functions ----------------------------- //


// Reads the binary file associated with the molecule.
void read_file(
    std::string const& file, int dimensionality, int nAtoms,
    std::vector<double>& radii,
    std::vector<std::vector<double>>& coordinates
);


// --------------------------- Validate Functions --------------------------- //


// Validate the number of arguments.
void validateArguments(int nParameters);


// *****************************************************************************
// Main Function
// *****************************************************************************


/**
 * Runs the program.
*/
int main(int argc, char **argv)
{
    // Where the coordinates will be stored.
    std::vector<double> radii{};
    std::vector<std::vector<double>> coordinates{};

    // Validate the number of arguments passed.
    validateArguments(argc);

    // Setup the values.
    int dimensionality = std::stoi(argv[2]);
    int nAtoms = std::stoi(argv[3]);

    // Get the file information.
    read_file(argv[1], dimensionality , nAtoms, radii, coordinates);

    std::cout << coordinates.size() << std::endl;
    std::cout << coordinates[0].size() << std::endl;


    std::cout << radii.size() << std::endl;
    for(size_t i = 0; i < radii.size(); ++i)
        std::cout << (i + 1) << ". radius: " << radii[i] << std::endl;
}

// *****************************************************************************
// Function Declaration
// *****************************************************************************


// -----------------------------------------------------------------------------
// Validate Functions
// -----------------------------------------------------------------------------


// Reads the binary file associated with the molecule.
void read_file(
    std::string const& file, int dimensionality, int nAtoms,
    std::vector<double>& radii,
    std::vector<std::vector<double>>& coordinates
)
{
    // Initialize the vectors.
    radii = std::vector<double>(nAtoms,0.0);

    coordinates = std::vector<std::vector<double>>(
        nAtoms, std::vector<double>(dimensionality, 0.0)
    );
}


// -----------------------------------------------------------------------------
// Validate Functions
// -----------------------------------------------------------------------------


/**
 * Validate the number of arguments is exactly 4.
 *
 * @param nParameters The number of read command-line arguments.
*/
void validateArguments(int nParameters)
{
    // Check the number of parameters is exactly 4.
    if(nParameters != EXPECTED)
    {
        std::string nCurrent = std::to_string(nParameters);
        std::string nExpected = std::to_string(EXPECTED);
        throw std::range_error(
            "The number of parameters is wrong; current: " + nCurrent + ", "
            "expected: " + nExpected + "."
        );
    }
}
