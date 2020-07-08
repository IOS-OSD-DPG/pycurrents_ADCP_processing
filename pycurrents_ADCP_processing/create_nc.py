"""
Sample usage of:
    - ADCP_processing_L1.py
    - ADCP_processing_L0.py
    - add_var2nc.py
    - ADCP_IOS_Header_file.py

Outputs:
    - L0 netCDF ADCP file without geographic_area variable
    - L0 netCDF ADCP file with geographic_area variable
    - L1 netCDF ADCP file without geographic_area variable
    - L1 netCDF ADCP file with geographic_area variable
    - IOS .adcp header file (produced from L1 netCDF ADCP file with geographic_area variable)
"""

from pycurrents_ADCP_processing import ADCP_processing_L0, ADCP_processing_L1, add_var2nc, ADCP_IOS_Header_file
import os


# Remove before uploading to GitHub
os.chdir('/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/ADCP_L1/scott1_20170706_20170711_0101m/')

# Define raw ADCP file and associated metadata file
f = './sample_data/a1_20050503_20050504_0221m.000'
meta = './sample_data/a1_20050503_20050504_0221m_meta_L1.csv'

f = 'scott1_20170706_20170711_0101m.000'
meta = 'scott1_20170706_20170711_0101m_meta_L1.csv'

# Perform L0 processing on the raw data and export as a netCDF file
ncname_L0 = ADCP_processing_L0.nc_create_L0(f_adcp=f, f_meta=meta)

# Perform L1 processing on the raw data and export as a netCDF file
ncname_L1 = ADCP_processing_L1.nc_create_L1(inFile=f, file_meta=meta)

os.chdir('/home/hourstonh/Gitkraken_local/pycurrents_ADCP_processing/')

# Read in the netCDF file produced above and add a geographic_area variable
# Export as a new netCDF file
geoname_L1 = add_var2nc.add_geo(ncname_L1)

# Generate a header (.adcp) file from the newest netCDF file
ADCP_IOS_Header_file.main_header(geoname_L1)

