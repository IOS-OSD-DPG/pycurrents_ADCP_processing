# pycurrents_ADCP_processing

Authors: Hana Hourston (@hhourston) , Di Wan (@onedwd), Lu Guan (@guanlu129), Maxim Krassovski

For performing "level 0" (L0), "level 1" (L1), and "level 2" (L2) processing on raw 
moored ADCP data in Python using the UHDAS `pycurrents` package and plotting the output netCDF data. 
Teledyne RDI Workhorse, Sentinel V, and Broadband instruments are supported.

L0 processing does not include any processing. Raw ADCP data is combined with metadata from a csv file and exported 
in netCDF format. 

L1 processing contains minimal processing. Raw ADCP data is also combined with metadata from a csv file and exported 
in netCDF format. The difference from L0 is that L1 processing comprises:
* Corrections for magnetic declination
* Calculation of sea surface height from pressure values and latitude
* Rotation into enu coordinates if this is not already the coordinate system of the dataset
* Removing leading and trailing ensembles from before and after deployment
* Flagging negative pressure values
* Optional: Splitting the dataset into segments if there are pressure changes due to mooring strikes. The user must provide the date-times or ensembles at which to split the dataset

L2 processing contains:
* Flagging data in bins where calculated pressure is negative
* Flagging data by backscatter increases in upward-facing ADCPs
* Flagging data below the depth of the sea floor in downward-facing ADCPs
* Optional: Calculation of pressure data from CTD pressure data from the same deployment (if the ADCP was missing a pressure sensor)

If the time data in the raw file are garbled, then use *generate_time_range.py* to create a csv file containing 
regularly-spaced time data.  

*ADCP_IOS_header_file.py* produces an IOS Shell header file for each netCDF file that makes the netCDF file searchable 
on the [IOS Water Properties website](https://www.waterproperties.ca/). 

*plot_westcoast_nc_LX.py* contains code for the following types of plots:
* pressure data time series (check potential ADCP depth change)
* North/East (LCNSAP01/LCEWAP01) and along-/cross-shore velocity profile time series 
* Low-pass filtered versions of the above plots (Godin or X-hour rolling mean, e.g., 30 hour)
* Eastward (LCEWAP01) bin 1 time series (check bad data close to ADCP)
* Diagnostic plots for mean backscatter, mean velocity, and principal axis (orientation)
* Feather plots for select bins: current vectors at the sea surface plotted over time
* Rotary spectra for selected bins
* Depth profile of rotary spectrum
* Depth profile of tidal ellipses
* Single bin North/East velocity plots

Reference: [New tools for ADCP processing.](https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/40993127.pdf) 
Canadian Technical Report of Hydrography and Ocean Sciences 336.

## Installation
1. Before creating a virtual environment for the package, create a folder for the virtual environment and enter the folder in terminal, e.g. "adcp"  
2. Create a virtual environment called "adcp37" with Python version 3.7:  
        `conda create -n adcp37 python=3.7`
3. Activate the virtual environment:  
        `conda activate adcp37`
4. Add the conda-forge to your channel:  
        `conda config --add channels conda-forge`  
        `conda config --set channel_priority strict`
5. Install required packages:  
        `conda install numpy scipy pip pandas netCDF4 xarray gsw matplotlib=3.5 shapely`  
        `pip install datetime ruamel.yaml`  
6. Install ttide_py from github  
        `git clone https://github.com/moflaher/ttide_py`  
        Navigate to the newly created ttide_py folder, then  
        `python setup.py install`  
7. Clone pycurrents with Mercurial:  
        `hg clone --verbose http://currents.soest.hawaii.edu/hg/pycurrents`  
8. Install pycurrents:  
        `pip install -e ./pycurrents`  
9. Clone this `pycurrents_ADCP_processing` repository with git:  
        `git clone https://github.com/IOS-OSD-DPG/pycurrents_ADCP_processing.git`  
10. cd to the `pycurrents_ADCP_processing` directory and run "setup.py":  
         `python setup.py install`

### Pre-requisites
* Linux (or Unix-like) environment
* Python 3.7

## Usage 
Sample usage of *ADCP_processing_L1.py*, *ADCP_processing_L0.py*, *add_var2nc.py* and *ADCP_IOS_Header_file.py* 
is laid out in *create_nc.py*. An example of how to create uniform time data (for replacing invalid time data 
in a raw ADCP file) can be found in *generate_time_range.py*. Sample usage of the plotting functions in 
*plot_westcoast_nc_LX.py* is given in the file *example_plot_westcoast.py*.

## Credits
*ADCP_pycurrents_L1.py* is based off of the gist, *RdiToNetcdf.ipynb*, by Jody Klymak (https://github.com/jklymak), 
and includes contributions from Di Wan (https://github.com/onedwd) and Eric Firing (https://github.com/efiring). 
*add_var2nc.py* was written by Di Wan. *ADCP_IOS_Header_file.py* was written by Lu Guan (https://github.com/guanlu129).

## Resources
A web app based off this package can be found on the [IOS Data Management Apps](https://dmapps.waterproperties.ca/en/) 
(DM Apps) page. Credits: Tom Roe.

Documentation:
* `pycurrents` package and log: https://currents.soest.hawaii.edu/hgstage/pycurrents/file/tip
* netCDF documentation: https://www.unidata.ucar.edu/software/netcdf/docs/index.html
* RDI Broadband primer: https://www.comm-tec.com/Docs/Manuali/RDI/BBPRIME.pdf
* RDI Ocean Surveyor technical manual: http://www.teledynemarine.com/Documents/Brand%20Support/RD%20INSTRUMENTS/Technical%20Resources/Manuals%20and%20Guides/Ocean%20Surveyor_Observer/Ocean%20Surveyor%20Technical%20Manual_Apr18.pdf
    * Raw data variable terminology starts on page 148; `pycurrents` outputs variables with these names

Conventions:
* BODC SeaDataNet quality flags: https://www.bodc.ac.uk/data/documents/series/37006/#QCflags
* BODC SeaDataNet P01 vocabulary search: http://seadatanet.maris2.nl/v_bodc_vocab_v2/search.asp?lib=p01&screen=0
* BODC SeaDataNet P06 data storage unit search: http://seadatanet.maris2.nl/v_bodc_vocab_v2/search.asp?lib=p06&screen=0
* CF Conventions standard name table: https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html 
* GF3 codes (no longer maintained): https://www.nodc.noaa.gov/woce/woce_v3/wocedata_1/sss/documents/liste_param.htm

Tidal analysis:  
Pawlowicz, R., B. Beardsley, and S. Lentz, "Classical Tidal Harmonic Analysis Including Error Estimates in MATLAB 
using T_TIDE", Computers and Geosciences, 28, 929-937 (2002).