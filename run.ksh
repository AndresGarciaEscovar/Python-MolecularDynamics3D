#!/bin/ksh

# Run the program with the given conda environment.
if [[ $1 == "mac" ]]
then
    fpath="/Users/andres/Documents/Projects/Python/MolecularDynamics/data"
    fpath="$fpath/simulation_mac.yaml"
    conda run -n mdynamics python -m code $fpath
elif [[ $1 == "linux" ]]
then
    fpath="/home/hp/Projects/Programming/Python/MolecularDynamics/data"
    fpath="$fpath/simulation_linux.yaml"
    conda run -n mdynamics python -m code $fpath
else
    echo "Please specify the operating system: linux or mac"
fi