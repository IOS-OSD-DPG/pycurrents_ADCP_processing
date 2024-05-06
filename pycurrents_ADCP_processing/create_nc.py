"""
Sample usage of:
    - ADCP_processing_L0_L1.py
    - ADCP_IOS_Header_file.py

Outputs:
    - L0 netCDF ADCP file with geographic_area variable
    - L1 netCDF ADCP file with geographic_area variable
    - IOS .adcp header file (produced from L1 netCDF ADCP file with geographic_area variable)
    - Full suite of plots for output L1 files
"""

from pycurrents_ADCP_processing import ADCP_processing_L0_L1, ADCP_IOS_Header_file
from pycurrents_ADCP_processing import plot_westcoast_nc_LX
# from deprecated import add_var2nc

# Define raw ADCP file and associated metadata file
f = './sample_data/a1_20050503_20050504_0221m.000'
meta = './sample_data/a1_20050503_20050504_0221m_metadata.csv'
dest_dir = 'dest_dir'

# Perform L0 processing on the raw data and export as a netCDF file
# ncname_L0 = ADCP_processing_L0.nc_create_L0(f_adcp=f, f_meta=meta, dest_dir=dest_dir)
ncnames_L0 = ADCP_processing_L0_L1.nc_create_L0_L1(in_file=f, file_meta=meta, dest_dir=dest_dir, level=0)


# Perform L1 processing on the raw data and export as a netCDF file
ncnames_L1 = ADCP_processing_L0_L1.nc_create_L0_L1(in_file=f, file_meta=meta, dest_dir=dest_dir, level=1)

# Generate a header (.adcp) file from the L1 netCDF file that has the geographic area variable
for n in ncnames_L1:
    header_name = ADCP_IOS_Header_file.main_header(n, dest_dir)

    plot_list = plot_westcoast_nc_LX.create_westcoast_plots(n, dest_dir, do_all_plots=True)