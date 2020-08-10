"""
author: Hana Hourston
Sample usage of functions in plot_westcoast_nc_LX.py
"""

from pycurrents_ADCP_processing import plot_westcoast_nc_LX

ncfile = './newnc/a1_20160713_20170513_0480m.adcp.L1.nc'
bad_surface_bins = 3  # these bins nearest the surface are not plotted
dest_dir = 'dest_dir'
cross_angle = 25

output_files = plot_westcoast_nc_LX.create_westcoast_plots(ncfile, dest_dir, "Godin", bad_surface_bins, cross_angle)

