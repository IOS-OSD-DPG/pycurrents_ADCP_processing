"""
author: Hana Hourston
Sample usage of functions in plot_westcoast_nc_LX.py
"""

from pycurrents_ADCP_processing import plot_westcoast_nc_LX

ncfile = './newnc/a1_20160713_20170513_0480m.adcp.L1.nc'
dest_dir = 'dest_dir'

output_files = plot_westcoast_nc_LX.create_westcoast_plots(ncfile, dest_dir, "Godin", None)
