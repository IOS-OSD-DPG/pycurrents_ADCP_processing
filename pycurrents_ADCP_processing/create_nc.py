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

from pycurrents_ADCP_processing import ADCP_processing_L0, ADCP_processing_L1, ADCP_IOS_Header_file
from deprecated import add_var2nc

# Define raw ADCP file and associated metadata file
f = './sample_data/a1_20050503_20050504_0221m.000'
meta = './sample_data/a1_20050503_20050504_0221m_meta_L1.csv'
dest_dir = 'dest_dir'

# Perform L0 processing on the raw data and export as a netCDF file
ncname_L0 = ADCP_processing_L0.nc_create_L0(f_adcp=f, f_meta=meta, dest_dir=dest_dir)

# Perform L1 processing on the raw data and export as a netCDF file
ncname_L1 = ADCP_processing_L1.nc_create_L1(inFile=f, file_meta=meta, dest_dir=dest_dir)

# Generate a header (.adcp) file from the L1 netCDF file that has the geographic area variable
header_name = ADCP_IOS_Header_file.main_header(ncname_L1, dest_dir)

