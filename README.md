# pycurrents_ADCP_processing

For processing raw ADCP data in Python using the UHDAS pycurrents package -- under construction.

*ADCP_pycurrents_3.py* is based off of Jody Klymak's script (https://gist.github.com/jklymak/b39172bd0f7d008c81e32bf0a72e2f09). 
This version contains changes to variable names (following BODC conventions), and addition of variables, variable and global attributes, and dimensions to match the netCDF ADCP files produced before using R. This script is in-progress.

*read_mcdcp.py* is in-progress.

*ADCP_pycurrents_rdiraw.py* and *ADCP_pycurrents_rdiraw_reshape.py* use the `pycurrents.adcp.rdiraw.rawfile()` function instead of the `pycurrents.adcp.rdiraw.Multiread()` function to import raw ADCP files. Both are in-progress.

## Installation
To install the Python package `pycurrents`, follow the instructions at https://currents.soest.hawaii.edu/ocn_data_analysis/installation.html.

## Usage
*ADCP_pycurrents.py* uses the `pycurrents` package to open a raw ADCP file in Python and export it in netCDF file format. The script also uses a .csv metadata file for the raw ADCP file whose contents are combined with the raw data in the netCDF file, so that the netCDF file is self-describing. This metadata file is filled out by the user and a template can be found at https://github.com/hhourston/ADCP_processing_visualization/tree/master/ADCP_metadata_template. 

## Credits
*ADCP_pycurrents.py* is based off of the gist, *RdiToNetcdf.ipynb*, by Jody Klymak (https://github.com/jklymak).
