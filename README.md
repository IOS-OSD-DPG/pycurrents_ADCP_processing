# pycurrents_ADCP_processing

For processing raw ADCP data in Python using the UHDAS `pycurrents` package (https://currents.soest.hawaii.edu/docs/adcp_doc/codas_setup/index.html).

*ADCP_pycurrents_L1.py* is based off of Jody Klymak's script (https://gist.github.com/jklymak/b39172bd0f7d008c81e32bf0a72e2f09). This version contains changes to variable names (following BODC conventions); addition of variables, variable and global attributes, and minimal, "level 1" processing.  

Level 1 (L1) processing comprises:
* Corrections for magnetic declination
* Calculation of sea surface height from pressure values and latitude
* Rotation into enu coordinates if this is not already the coordinate system of the dataset
* Flagging leading and trailing ensembles from before and after deployment and setting them to nan's
* Flagging negative pressure values

*ADCP IOS header file.py* produces an IOS Shell header file for each netCDF file that makes the netCDF file searchable on the IOS Water Properties website (https://www.waterproperties.ca/mainlogin.php?refer=/). 

## Installation
1. Before creating a virtual environment for the package, create a folder for the virtual environment and enter the folder in terminal, e.g. "test"  
2. Create a virtual environment called "adcp37" with Python version 3.7:  
        `conda create -n adcp37 python=3.7`
3. Activate the virtual environment:  
        `conda activate adcp37`
4. Add the conda-forge to your channel:  
        `conda config --add channels conda-forge`  
        `conda config --set channel_priority strict`
4. Install required packages:  
        `conda install numpy scipy pip pandas netCDF4 xarray gsw`  
        `pip install datetime`  
5. Clone pycurrents:  
        `hg clone --verbose http://currents.soest.hawaii.edu/hg/pycurrents`  
6. Install pycurrents:  
        `pip install -e ./pycurrents`  
6. Clone this `pycurrents_ADCP_processing` repository with git:  
        `git clone https://github.com/hhourston/pycurrents_ADCP_processing.git`  

### Pre-requisites
* Linux (or Unix-like) environment
* Python 2.7

## Usage
*ADCP_pycurrents_L1.py* uses the `pycurrents` package to open a raw ADCP file in Python and export it in netCDF file format. The script also uses a .csv metadata file for the raw ADCP file whose contents are combined with the raw data in the netCDF file, so that the netCDF file is self-describing. This metadata file is filled out by the user and a template can be found at https://github.com/hhourston/ADCP_processing_visualization/tree/master/ADCP_metadata_template. 

## Credits
*ADCP_pycurrents_L1.py* is based off of the gist, *RdiToNetcdf.ipynb*, by Jody Klymak (https://github.com/jklymak), and includes contributions from Di Wan (https://github.com/onedwd) and Eric Firing (https://github.com/efiring). *add_var2nc.py* was written by Di Wan. 

## Helpful links
Documentation:
* `oce` documentation: https://cran.r-project.org/web/packages/oce/oce.pdf
* `ncdf4` documentation: https://cran.r-project.org/web/packages/ncdf4/ncdf4.pdf
* netCDF documentation: https://www.unidata.ucar.edu/software/netcdf/docs/index.html 
* RDI Ocean Surveyor technical manual: http://www.teledynemarine.com/Documents/Brand%20Support/RD%20INSTRUMENTS/Technical%20Resources/Manuals%20and%20Guides/Ocean%20Surveyor_Observer/Ocean%20Surveyor%20Technical%20Manual_Apr18.pdf
    * Raw data variable terminology starts on page 148; `pycurrents` outputs variables following these names

Conventions:
* BODC SeaDataNet quality flags: https://www.bodc.ac.uk/data/documents/series/37006/#QCflags
* BODC SeaDataNet P01 vocabulary search: http://seadatanet.maris2.nl/v_bodc_vocab_v2/search.asp?lib=p01&screen=0
* GF3 codes (no longer maintained): https://www.nodc.noaa.gov/woce/woce_v3/wocedata_1/sss/documents/liste_param.htm
* CF Conventions: http://cfconventions.org/standard-names.html 