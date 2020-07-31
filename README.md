# pycurrents_ADCP_processing

For performing "level 0" (L0) and "level 1" (L1) processing on raw moored ADCP data in Python using the UHDAS `pycurrents` package.

L0 processing does not include any processing. Raw ADCP data is combined with metadata from a csv file and exported in netCDF format. 

L1 processing contains minimal processing. Raw ADCP data is also combined with metadata from a csv file and exported in netCDF format. The difference from L0 is that L1 processing comprises:
* Corrections for magnetic declination
* Calculation of sea surface height from pressure values and latitude
* Rotation into enu coordinates if this is not already the coordinate system of the dataset
* Flagging leading and trailing ensembles from before and after deployment and setting them to nan's
* Flagging negative pressure values

*add_var2nc.py* adds a geographic_area variable to a netCDF file from either the L0 or L1 process and exports a new netCDF file containing this addition.

*ADCP IOS header file.py* produces an IOS Shell header file for each netCDF file that makes the netCDF file searchable on the IOS Water Properties website (https://www.waterproperties.ca/). 

## Installation
1. Before creating a virtual environment for the package, create a folder for the virtual environment and enter the folder in terminal, e.g. "adcp"  
2. Create a virtual environment called "adcp37" with Python version 3.7:  
        `conda create -n adcp37 python=3.7`
3. Activate the virtual environment:  
        `conda activate adcp37`
4. Add the conda-forge to your channel:  
        `conda config --add channels conda-forge`  
        `conda config --set channel_priority strict`
4. Install required packages:  
        `conda install numpy scipy pip pandas netCDF4 xarray gsw matplotlib shapely`  
        `pip install datetime`  
5. Clone pycurrents with Mercurial:  
        `hg clone --verbose http://currents.soest.hawaii.edu/hg/pycurrents`  
6. Install pycurrents:  
        `pip install -e ./pycurrents`  
6. Clone this `pycurrents_ADCP_processing` repository with git:  
        `git clone https://github.com/hhourston/pycurrents_ADCP_processing.git`  
7. cd to the `pycurrents_ADCP_processing` directory and run "setup.py":  
        `sudo python setup.py install`

### Pre-requisites
* Linux (or Unix-like) environment
* Python 3.7

## Usage 
Sample usage of *ADCP_processing_L1.py*, *ADCP_processing_L0.py*, *add_var2nc.py* and *ADCP_IOS_Header_file.py* is laid out in *create_nc.py*. An example of how to create uniform time data (for replacing invalid time data in a raw ADCP file) can be found in *generate_time_range.py*.

## Credits
*ADCP_pycurrents_L1.py* is based off of the gist, *RdiToNetcdf.ipynb*, by Jody Klymak (https://github.com/jklymak), and includes contributions from Di Wan (https://github.com/onedwd) and Eric Firing (https://github.com/efiring). *add_var2nc.py* was written by Di Wan. *ADCP_IOS_Header_file.py* was written by Lu Guan (https://github.com/guanlu129).

## Helpful links
Documentation:
* `pycurrents` package and log: https://currents.soest.hawaii.edu/hgstage/pycurrents/file/tip
* netCDF documentation: https://www.unidata.ucar.edu/software/netcdf/docs/index.html
* RDI Broadband primer: https://www.comm-tec.com/Docs/Manuali/RDI/BBPRIME.pdf
* RDI Ocean Surveyor technical manual: http://www.teledynemarine.com/Documents/Brand%20Support/RD%20INSTRUMENTS/Technical%20Resources/Manuals%20and%20Guides/Ocean%20Surveyor_Observer/Ocean%20Surveyor%20Technical%20Manual_Apr18.pdf
    * Raw data variable terminology starts on page 148; `pycurrents` outputs variables with these names

Conventions:
* BODC SeaDataNet quality flags: https://www.bodc.ac.uk/data/documents/series/37006/#QCflags
* BODC SeaDataNet P01 vocabulary search: http://seadatanet.maris2.nl/v_bodc_vocab_v2/search.asp?lib=p01&screen=0
* CF Conventions standard name table: http://cfconventions.org/standard-names.html 
* GF3 codes (no longer maintained): https://www.nodc.noaa.gov/woce/woce_v3/wocedata_1/sss/documents/liste_param.htm
