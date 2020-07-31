"""
author: Hana Hourston
Sample usage of functions in plot_westcoast_nc_LX.py
"""

from pycurrents_ADCP_processing import plot_westcoast_nc_LX
import xarray as xr

ncfile = './newnc/a1_20160713_20170513_0480m.adcp.L1.nc'
ncdata = xr.open_dataset(ncfile)

bad_surface_bins = 3  #to input by user; these bins nearest the surface are not plotted

start_end = plot_westcoast_nc_LX.get_L1_start_end(ncdata=ncdata)

time_lim, bin_depths_lim, ns_lim, ew_lim = plot_westcoast_nc_LX.limit_data(ncdata, ncdata.LCEWAP01.data,
                                                                           ncdata.LCNSAP01.data,
                                                                           ncdata.time.data, ncdata.orientation,
                                                                           ncdata.instrument_depth,
                                                                           ncdata.distance.data, bad_surface_bins)

# North/East velocity plots
plot_westcoast_nc_LX.make_pcolor_ne(ncdata, time_lim, bin_depths_lim, ns_lim, ew_lim)

# Along/Cross-shore velocity plots
plot_westcoast_nc_LX.make_pcolor_ac(ncdata, time_lim, bin_depths_lim, ns_lim, ew_lim, cross_angle=25)

# Apply Godin filter and repeat the above steps using the filtered data
ew_filt, ns_filt = plot_westcoast_nc_LX.filter_godin(ncdata)

# Limit data
time_lim, bin_depths_lim, ns_filt_lim, ew_filt_lim = plot_westcoast_nc_LX.limit_data(ncdata, ew_filt, ns_filt,
                                                                                     ncdata.time.data,
                                                                                     ncdata.orientation,
                                                                                     ncdata.instrument_depth,
                                                                                     ncdata.distance.data,
                                                                                     bad_surface_bins)

# Godin-filtered North/East velocity plots
plot_westcoast_nc_LX.make_pcolor_ne(ncdata, time_lim, bin_depths_lim, ns_filt_lim, ew_filt_lim, filter_type='Godin')

# Along/Cross-shore velocity plots
plot_westcoast_nc_LX.make_pcolor_ac(ncdata, time_lim, bin_depths_lim, ns_filt_lim, ew_filt_lim, cross_angle=25,
                                    filter_type='godin')

# Compare raw and filtered velocity in bin 1
plot_westcoast_nc_LX.binplot_compare_filt(ncdata, time_lim, ew_lim, ew_filt_lim, filter_type='Godin', direction='east')
