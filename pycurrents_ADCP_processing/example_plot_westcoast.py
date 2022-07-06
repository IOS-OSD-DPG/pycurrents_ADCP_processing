"""
author: Hana Hourston
Sample usage of functions in plot_westcoast_nc_LX.py
"""

from pycurrents_ADCP_processing import plot_westcoast_nc_LX

ncfile = './newnc/a1_20160713_20170513_0480m.adcp.L1.nc'
dest_dir = 'dest_dir'

# -----------------OPTION 1-------------------
# Option 1: *do not* manually enter limits for bin and time ranges
# Let the program determine them automatically, along with the along-shore angle
# and the matplotlib.pyplot colourmap limits
output_files = plot_westcoast_nc_LX.create_westcoast_plots(
    ncfile, dest_dir, "Godin", None)

# -----------------OPTION 2-------------------
# Option 2: *do* manually enter limits for bin and time ranges
# Show how to set time and bin limits manually, along with the along-shore angle
# and the matplotlib.pyplot colourmap limits
# Also activate the resampling override, which is in place for files with size > 100 MB
#
dest_dir_manual = 'dest_dir_manual'
along_shore_angle = 115.  # degrees CCW from East
time_start_end_ind = (0, 10000)  # indices
bin_start_end_ind = (0, -4)  # indices
colourmap_limits = (-0.7, 0.7)  # for pcolormesh() plots
override_resample = True  # Use caution when turning on override for files greater than 100 MB
output_files_manual = plot_westcoast_nc_LX.create_westcoast_plots(
    ncfile, dest_dir_manual, "30h", along_shore_angle, time_start_end_ind,
    bin_start_end_ind, colourmap_limits, override_resample)
