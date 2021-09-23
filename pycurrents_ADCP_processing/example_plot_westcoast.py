"""
author: Hana Hourston
Sample usage of functions in plot_westcoast_nc_LX.py
"""

from pycurrents_ADCP_processing import plot_westcoast_nc_LX

ncfile = './newnc/a1_20160713_20170513_0480m.adcp.L1.nc'
dest_dir = 'dest_dir'

# output_files = plot_westcoast_nc_LX.create_westcoast_plots(ncfile, dest_dir, "Godin", None)

# Show how to set time and bin limits manually
dest_dir_manual = 'dest_dir_manual'
along_shore_angle = 20  # degrees
time_start_end_ind = (0, 10000)  # indices
bin_start_end_ind = (0, -4)  # indices
colourmap_limits = (-0.7, 0.7)  # for pcolormesh() plots
output_files_manual = plot_westcoast_nc_LX.create_westcoast_plots(
    ncfile, dest_dir_manual, "30h", along_shore_angle, time_start_end_ind, bin_start_end_ind,
    colourmap_limits)
