# pycurrents_ADCP_processing

For processing raw ADCP data in Python using the UHDAS `pycurrents` package.

*ADCP_pycurrents_rdiraw.py* is based off of Jody Klymak's script (https://gist.github.com/jklymak/b39172bd0f7d008c81e32bf0a72e2f09). This version contains changes to variable names (following BODC conventions), and addition of variables, variable and global attributes, and dimensions to match the netCDF ADCP files produced before using R. It uses the `pycurrents.adcp.rdiraw.rawfile()` function instead of the `pycurrents.adcp.rdiraw.Multiread()` function to import raw ADCP files. This script is in-progress. 

## Installation
To install the Python package `pycurrents`, follow the instructions at https://currents.soest.hawaii.edu/ocn_data_analysis/installation.html.

To download this script without having to clone the whole repository:
1. Open a file in a GitHub repository and click on the button "raw" in the top right corner to view the raw file in a new browser tab. Copy the url of the raw file (e.g. https://raw.githubusercontent.com/username/reponame/path/to/file).
2. To save this file to your computer, open a terminal window and enter
    
        wget https://raw.githubusercontent.com/username/reponame/path/to/file --no-check-certificate
   
   "--no-check-certificate" allows the user to connect to GitHub insecurely and download a file successfully.
    

## Usage
*ADCP_pycurrents_rdiraw.py* uses the `pycurrents` package to open a raw ADCP file in Python and export it in netCDF file format. The script also uses a .csv metadata file for the raw ADCP file whose contents are combined with the raw data in the netCDF file, so that the netCDF file is self-describing. This metadata file is filled out by the user and a template can be found at https://github.com/hhourston/ADCP_processing_visualization/tree/master/ADCP_metadata_template. 

A description of output data format can be found in the RDI Ocean Surveyor technical manual at http://www.teledynemarine.com/Documents/Brand%20Support/RD%20INSTRUMENTS/Technical%20Resources/Manuals%20and%20Guides/Ocean%20Surveyor_Observer/Ocean%20Surveyor%20Technical%20Manual_Apr18.pdf, starting on page 148.

## Credits
*ADCP_pycurrents_rdiraw.py* is based off of the gist, *RdiToNetcdf.ipynb*, by Jody Klymak (https://github.com/jklymak).
