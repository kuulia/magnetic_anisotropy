# 1. Feb 2024
# Author: Linus Lind
# Magnetic anisotropy project
# The aim of this project is to study anisotropic properties of a ferromagnetic film sample.
# The magnetic film sample was rotated and subjected to an external magnetic field after and measured with a Kerr microscope. 
# This project is mainly for results post-processing and data visualization 
# The folders XXdeg contain measurement data from different angles with respect to the easy angle. In each folder there are plots of the data.
# saturation_gradient.py contains useful functions for reading the raw data files, removing faraday effect from the measurement data, applying scaling to the data and finding zero-intercepts via linear interpolation
# processing.py processes the results for all angles.
# .csv files are data files of intensity and applied magnetric field