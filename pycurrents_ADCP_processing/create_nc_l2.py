# Example usage of ADCP_processing_L2 in a script

from pycurrents_ADCP_processing import ADCP_processing_L2

# Sample L1 netCDF ADCP file
f_adcp = './newnc/a1_20050503_20050504_0221m.adcp.L1.nc'
dest_dir = 'dest_dir'

out_files = ADCP_processing_L2.create_nc_L2(f_adcp, dest_dir)
