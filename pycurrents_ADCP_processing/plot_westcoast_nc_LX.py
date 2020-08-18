# -*- coding: utf-8 -*-

__author__ = 'diwan'

"""
This script contains functions that make several different kinds of plots of ADCP
netCDF-format data. The types of plots are:
    1. North and East current velocities (one plot containing a subplot for each)
    2. Along and cross-shelf current velocities (one plot containing a subplot for each)
    3. North and East Godin or 30h-averaged filtered current velocities
    4. Along and cross-shelf Godin or 30h-averaged filtered current velocities
    5. Bin plot for one month's worth of velocity data comparing raw and filtered (Godin or 30h-averaged) data
    
Plots can be made from L1- or L2-processed data.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd


def resolve_to_alongcross(u_true, v_true, along_angle):
    # Rotate North and East velocities to along- and cross-shore velocities given an along-shore angle
    # along_angle measured in degrees counter-clockwise from geographic East
    along_angle = np.deg2rad(along_angle)

    u_along = u_true * np.cos(along_angle) + v_true * np.sin(along_angle)
    u_cross = u_true * np.sin(along_angle) - v_true * np.cos(along_angle)

    return u_along, u_cross


def get_L1_start_end(ncdata):
    # Obtain the number of leading and trailing ensembles that were set to nans in L1 processing
    # from the processing_history global attribute.
    # Then get the index of the last ensemble cut from the beginning of the time series
    # and the index of the first ensemble cut from the end of the time series.
    # ncdata: dataset-type object created by reading in a netCDF ADCP file with the xarray package

    digits_in_process_hist = [int(s) for s in ncdata.attrs['processing_history'].split() if s.isdigit()]
    # The indices used are conditional upon the contents of the processing history remaining unchanged
    # in L1 and before L2. Appending to the end of the processing history is ok
    start_end_indices = (digits_in_process_hist[1], len(ncdata.time.data) - digits_in_process_hist[2] - 1)

    return start_end_indices


def limit_data(ncdata, ew_data, ns_data):
    if ncdata.orientation == 'up':
        bin_depths = ncdata.instrument_depth - ncdata.distance.data
    else:
        bin_depths = ncdata.instrument_depth + ncdata.distance.data
    print(bin_depths)

    # data.time should be limited to the data.time with no NA values; bins must be limited
    if 'L1' in ncdata.filename.data.tolist() or 'L2' in ncdata.filename.data.tolist():
        new_first_last = get_L1_start_end(ncdata=ncdata)
    else:
        new_first_last = (0, len(ew_data[0]))

    # Remove bins where surface backscatter occurs
    time_lim = ncdata.time.data[new_first_last[0]:new_first_last[1]]

    bin_depths_lim = bin_depths[bin_depths >= 0]
    print(bin_depths_lim)

    ew_lim = ew_data[bin_depths >= 0, new_first_last[0]:new_first_last[1]]
    ns_lim = ns_data[bin_depths >= 0, new_first_last[0]:new_first_last[1]]

    return time_lim, bin_depths_lim, ns_lim, ew_lim


def make_pcolor_ne(data, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim, level0=False, filter_type='raw'):
    # filter_type options: 'raw' (default), '30h' (or, 35h, etc, average), 'Godin' (Godin Filtered)

    magnetic = '' #specify "Magnetic (North)" for L0 files, since magnetic declination wasn't applied to them
    if level0:
        magnetic = 'Magnetic '

    vminvmax = [-0.5, 0.5] #vertical min and max of the colour bar in the plots
    fig = plt.figure(figsize=(13.75, 10))
    ax = fig.add_subplot(2, 1, 1)

    f1 = ax.pcolor(time_lim, bin_depths_lim, ns_lim[:, :], cmap='RdBu_r', vmin=vminvmax[0], vmax=vminvmax[1])
    cbar = fig.colorbar(f1, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)
    ax.set_ylabel('Depth [m]', fontsize=14)

    if filter_type == '30h':
        ax.set_title(
            'ADCP ({}North, 30h average) {}-{} {}m'.format(magnetic, data.attrs['station'],
                                                           data.attrs['deployment_number'],
                                                           str(int(data.instrument_depth))), fontsize=14)
    elif filter_type == 'Godin':
        ax.set_title(
            'ADCP ({}North, Godin Filtered) {}-{} {}m'.format(magnetic, data.attrs['station'],
                                                              data.attrs['deployment_number'],
                                                              str(int(data.instrument_depth))), fontsize=14)
    elif filter_type == 'raw':
        ax.set_title(
            'ADCP ({}North, raw) {}-{} {}m'.format(magnetic, data.attrs['station'], data.attrs['deployment_number'],
                                                   str(int(data.instrument_depth))), fontsize=14)
    else:
        ValueError('Not a recognized data type; choose one of \'raw\', \'30h\' or \'Godin\'')

    if data.orientation == 'up':
        plt.gca().invert_yaxis()

    ax2 = fig.add_subplot(2, 1, 2)

    f2 = ax2.pcolor(time_lim, bin_depths_lim, ew_lim[:, :], cmap='RdBu_r', vmin=vminvmax[0], vmax=vminvmax[1])
    cbar = fig.colorbar(f2, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)

    ax2.set_ylabel('Depth [m]', fontsize=14)
    if 'h' in filter_type:  # xxh-average; e.g. '30h', '35h'
        ax2.set_title('ADCP ({}East, {} average) {}-{} {}m'.format(magnetic, filter_type, data.attrs['station'],
                                                                   data.attrs['deployment_number'],
                                                                   str(int(data.instrument_depth))), fontsize=14)
    elif filter_type == 'Godin':
        ax2.set_title('ADCP ({}East, Godin Filtered) {}-{} {}m'.format(magnetic, data.attrs['station'],
                                                                       data.attrs['deployment_number'],
                                                                       str(int(data.instrument_depth))), fontsize=14)
    elif filter_type == 'raw':
        ax2.set_title(
            'ADCP ({}East, raw) {}-{} {}m'.format(magnetic, data.attrs['station'], data.attrs['deployment_number'],
                                                  str(int(data.instrument_depth))), fontsize=14)

    if data.orientation == 'up':
        plt.gca().invert_yaxis()

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    if 'L0' in data.filename.data.tolist():
        plot_dir = './{}/L0_Python_plots/'.format(dest_dir)
    elif 'L1' in data.filename.data.tolist():
        plot_dir = './{}/L1_Python_plots/'.format(dest_dir)
    elif 'L2' in data.filename.data.tolist():
        plot_dir = './{}/L2_Python_plots/'.format(dest_dir)
    else:
        ValueError('Input netCDF file must be a L0, L1 or L2-processed file.')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if level0:
        plot_name = plot_dir + data.attrs['station'] + '-' + data.attrs['deployment_number'] + '_{0}m'.format(
            str(int(data.instrument_depth))) + '-magn_NE_{}.png'.format(filter_type)
    else:
        plot_name = plot_dir + data.attrs['station'] + '-' + data.attrs['deployment_number'] + '_{0}m'.format(
            str(int(data.instrument_depth))) + '-NE_{}.png'.format(filter_type)
    fig.savefig(plot_name)
    plt.close()

    return os.path.abspath(plot_name)


def determine_dom_angle(u_true, v_true):
    # Determine the dominant angle in degrees
    # along_angle measured in degrees relative to geographic East, counter-clockwise
    angles = np.arange(0, 180)
    max_rms = 0.
    max_angle = 0.
    for angle in angles:
        along_angle = np.deg2rad(angle)

        u_along = u_true * np.cos(along_angle) + v_true * np.sin(along_angle)
        rms = np.sqrt(np.nanmean(u_along * u_along))
        if rms > max_rms:
            max_rms = rms
            max_angle = angle # in degrees
    along_angle = max_angle
    cross_angle = max_angle - 90.
    return along_angle, cross_angle


def make_pcolor_ac(data, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim, filter_type='raw'):
    # filter_type options: 'raw' (default), '30h' (or, 35h, etc, average), 'Godin' (Godin Filtered)
    # cross_angle in degrees; defaults to 25
    along_angle, cross_angle = determine_dom_angle(ew_lim, ns_lim)
    print(along_angle, cross_angle)
#     along_angle = cross_angle + 90  # deg

    u_along, u_cross = resolve_to_alongcross(ew_lim, ns_lim, along_angle)
    AS = u_along
    CS = u_cross

    vminvmax = [-0.5, 0.5] #vertical min and max of the colour bar in the plots
    fig = plt.figure(figsize=(13.75, 10))
    ax1 = fig.add_subplot(2, 1, 1)

    f1 = ax1.pcolor(time_lim, bin_depths_lim, AS[:, :], cmap='RdBu_r', vmin=vminvmax[0], vmax=vminvmax[1])
    cbar = fig.colorbar(f1, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)

    ax1.set_ylabel('Depth [m]', fontsize=14)
    if 'h' in filter_type:
        ax1.set_title(
            'ADCP (along, {} average) {}$^\circ$ (CCW from E) {}-{} {}m'.format(filter_type, along_angle,
                                                                                data.attrs['station'],
                                                                                data.attrs['deployment_number'],
                                                                                math.ceil(data.instrument_depth)),
            fontsize=14)
    elif filter_type == 'Godin':
        ax1.set_title(
            'ADCP (along, Godin Filtered) {}$^\circ$ (CCW from E) {}-{} {}m'.format(along_angle, data.attrs['station'],
                                                                                    data.attrs['deployment_number'],
                                                                                    math.ceil(data.instrument_depth)),
            fontsize=14)
    elif filter_type == 'raw':
        ax1.set_title('ADCP (along, raw) {}$^\circ$ (CCW from E) {}-{} {}m'.format(along_angle, data.attrs['station'],
                                                                                   data.attrs['deployment_number'],
                                                                                   math.ceil(data.instrument_depth)),
                      fontsize=14)
    else:
        ValueError('Not a recognized data type; choose one of \'raw\', \'30h\' or \'Godin\'')

    if data.orientation == 'up':
        plt.gca().invert_yaxis()

    ax2 = fig.add_subplot(2, 1, 2)

    # vminvmax = get_vminvmax(CS)

    f2 = ax2.pcolor(time_lim, bin_depths_lim, CS[:, :], cmap='RdBu_r', vmin=vminvmax[0], vmax=vminvmax[1])
    cbar = fig.colorbar(f2, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)

    ax2.set_ylabel('Depth [m]', fontsize=14)
    if 'h' in filter_type:  # xxh-average; e.g. '30h', '35h'
        ax2.set_title(
            'ADCP (cross, {} average) {}$^\circ$ (CCW from E) {}-{} {}m'.format(filter_type, cross_angle,
                                                                                data.attrs['station'],
                                                                                data.attrs['deployment_number'],
                                                                                math.ceil(data.instrument_depth)),
            fontsize=14)
    elif filter_type == 'Godin':
        ax2.set_title(
            'ADCP (cross, Godin Filtered) {}$^\circ$ (CCW from E) {}-{} {}m'.format(str(cross_angle),
                                                                                    data.attrs['station'],
                                                                                    data.attrs['deployment_number'],
                                                                                    str(math.ceil(
                                                                                        data.instrument_depth))),
            fontsize=14)
    elif filter_type == 'raw':
        ax2.set_title(
            'ADCP (cross, raw) {}$^\circ$ (CCW from E) {}-{} {}m'.format(str(cross_angle), data.attrs['station'],
                                                                         data.attrs['deployment_number'],
                                                                         str(math.ceil(data.instrument_depth))),
            fontsize=14)
    else:
        ValueError('Not a recognized data type; choose one of \'raw\', \'30h\' or \'Godin\'')

    if data.orientation == 'up':
        plt.gca().invert_yaxis()

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    if 'L0' in data.filename.data.tolist():
        plot_dir = './{}/L0_Python_plots/'.format(dest_dir)
    elif 'L1' in data.filename.data.tolist():
        plot_dir = './{}/L1_Python_plots/'.format(dest_dir)
    elif 'L2' in data.filename.data.tolist():
        plot_dir = './{}/L2_Python_plots/'.format(dest_dir)
    else:
        ValueError('Input netCDF file must be a L0, L1 or L2-processed file.')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_name = plot_dir + data.attrs['station'] + '-' + data.attrs['deployment_number'] + '_{}m'.format(
        math.ceil(data.instrument_depth)) + '-AC_{}.png'.format(filter_type)
    fig.savefig(plot_name)
    plt.close()

    return os.path.abspath(plot_name)


def num_ens_per_hr(nc):
    # Calculate the number of ensembles recorded per hour
    time_incr = float(nc.time.data[1] - nc.time.data[0])
    hr2min = 60
    min2sec = 60
    sec2nsec = 1e9
    hr2nsec = hr2min * min2sec * sec2nsec
    return int(np.round(hr2nsec / time_incr, decimals=0))


def filter_godin(nc):
    # Make North and East lowpassed plots using the simple 3-day Godin filter
    # Running average of 24 hours first, then another time with a 24 hour filter, the again with a 25 hour filter
    # Repeat with along- and cross-shore velocities
    # nc: xarray Dataset-type object from reading in a netCDF file; contains 1D numpy array of values
    # Output: 1D numpy array of values of same size as the time series within nc, padded with nan's

    # Determine the number of ensembles taken per hour
    ens_per_hr = num_ens_per_hr(nc)
    print('Time stamps per hour:', ens_per_hr, sep=' ')
    window = int(ens_per_hr)
    num_hrs = 24

    # Rolling window calculations
    # Need to take transpose so that the rolling average is taken along the time (not bin) axis
    if 'L0' in nc.filename.data.tolist():
        ew_df = pd.DataFrame(data=nc.VEL_MAGNETIC_EAST.data.transpose())
        ns_df = pd.DataFrame(data=nc.VEL_MAGNETIC_NORTH.data.transpose())
    else:
        ew_df = pd.DataFrame(data=nc.LCEWAP01.data.transpose())
        ns_df = pd.DataFrame(data=nc.LCNSAP01.data.transpose())
    # 24h rolling average
    ew_filt_temp1 = ew_df.rolling(window=num_hrs * window, min_periods=None, center=True, win_type=None).median()
    ns_filt_temp1 = ns_df.rolling(window=num_hrs * window, min_periods=None, center=True, win_type=None).median()
    # 24h rolling average
    ew_filt_temp2 = ew_filt_temp1.rolling(window=num_hrs * window, min_periods=None, center=True,
                                          win_type=None).median()
    ns_filt_temp2 = ns_filt_temp1.rolling(window=num_hrs * window, min_periods=None, center=True,
                                          win_type=None).median()
    # 25h rolling average
    ew_filt_final = ew_filt_temp2.rolling(window=(num_hrs + 1) * window, min_periods=None, center=True,
                                          win_type=None).median()
    ns_filt_final = ns_filt_temp2.rolling(window=(num_hrs + 1) * window, min_periods=None, center=True,
                                          win_type=None).median()

    # Convert to numpy arrays
    ew_filt_final = ew_filt_final.to_numpy().transpose()
    ns_filt_final = ns_filt_final.to_numpy().transpose()

    return ew_filt_final, ns_filt_final


def filter_XXh(nc, num_hrs=30):
    # Perform XXh averaging on velocity data (30-hour, 35-hour, ...)
    # nc: xarray Dataset-type object from reading in a netCDF file; contains 1D numpy array of values
    # Output: 1D numpy array of values of same size as the time series within nc, padded with nan's

    # Determine the number of ensembles taken per hour
    ens_per_hr = num_ens_per_hr(nc)
    print(ens_per_hr)
    window = int(ens_per_hr)

    # Rolling window calculations
    # Need to take transpose so that the rolling average is taken along the time (not bin) axis
    if 'L0' in nc.filename.data.tolist():
        ew_df = pd.DataFrame(data=nc.VEL_MAGNETIC_EAST.data.transpose())
        ns_df = pd.DataFrame(data=nc.VEL_MAGNETIC_NORTH.data.transpose())
    else:
        ew_df = pd.DataFrame(data=nc.LCEWAP01.data.transpose())
        ns_df = pd.DataFrame(data=nc.LCNSAP01.data.transpose())
    ew_filt_final = ew_df.rolling(window=num_hrs * window, min_periods=None, center=True, win_type=None).median()
    ns_filt_final = ns_df.rolling(window=num_hrs * window, min_periods=None, center=True, win_type=None).median()

    # Convert to numpy arrays
    ew_filt_final = ew_filt_final.to_numpy().transpose()
    ns_filt_final = ns_filt_final.to_numpy().transpose()

    return ew_filt_final, ns_filt_final


def binplot_compare_filt(nc, dest_dir, time, dat_raw, dat_filt, filter_type, direction):
    # Function to take one bin from the unfiltered (raw) data and the corresponding bin in the filtered
    # data, and plot the time series together on one plot. Restrict time series to 1 month.
    # dat_filt: data filtered using the method defined in filter_type
    # filter_type options: 'Godin' or 'xxh' (e.g., '30h', '35h')
    # direction: 'east' or 'north'

    if direction == 'magnetic_east':
        vel_code = 'VEL_MAGNETIC_EAST'
    elif direction == 'magnetic_north':
        vel_code = 'VEL_MAGNETIC_NORTH'
    elif direction == 'east':
        vel_code = 'LCEWAP01'
    elif direction == 'north':
        vel_code = 'LCNSAP01'

    # Calculate the number of ensembles in 1 month
    ens_per_hr = num_ens_per_hr(nc)
    hr2day = 24
    day2mth = 30
    ens_per_mth = ens_per_hr * hr2day * day2mth

    bin_index = 0  # which bin to plot
    if nc.orientation == 'up':
        bin_depth = nc.instrument_depth - (bin_index + 1) * nc.cellSize
    else:
        bin_depth = nc.instrument_depth + (bin_index + 1) * nc.cellSize

    fig = plt.figure(figsize=(13.75, 10))

    ax1 = fig.add_subplot(1, 1, 1)

    f1 = ax1.plot(time[:ens_per_mth], dat_raw[bin_index, :ens_per_mth], color='k', label='Raw')
    if filter_type == 'Godin':
        f2 = ax1.plot(time[:ens_per_mth], dat_filt[bin_index, :ens_per_mth], color='b', label='Godin Filtered')
    elif 'h' in filter_type:
        f2 = ax1.plot(time[:ens_per_mth], dat_filt[bin_index, :ens_per_mth], color='b',
                      label='{} average'.format(filter_type))

    ax1.set_ylabel('Velocity [m s$^{-1}$]', fontsize=14)
    ax1.legend(loc='lower left')
    ax1.set_title(
        'ADCP {}-{} {} bin {} at {}m'.format(nc.attrs['station'], nc.attrs['deployment_number'],
                                             vel_code, bin_index + 1, bin_depth), fontsize=14)

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    if 'L0' in nc.filename.data.tolist():
        plot_dir = './{}/L0_Python_plots/'.format(dest_dir)
    elif 'L1' in nc.filename.data.tolist():
        plot_dir = './{}/L1_Python_plots/'.format(dest_dir)
    elif 'L2' in nc.filename.data.tolist():
        plot_dir = './{}/L2_Python_plots/'.format(dest_dir)
    else:
        ValueError('Input netCDF file must be a L0, L1 or L2-processed file.')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_name = plot_dir + nc.attrs['station'] + '-' + nc.attrs['deployment_number'] + '_{0}m'.format(
        str(math.ceil(nc.instrument_depth))) + '-{}_bin{}_compare_{}.png'.format(vel_code, bin_index + 1,
                                                                                 filter_type)
    fig.savefig(plot_name)
    plt.close()
    return os.path.abspath(plot_name)


def create_westcoast_plots(ncfile, dest_dir, filter_type="Godin"):
    """
    Inputs:
        - ncfile: file name of netCDF ADCP file
        - dest_dir: destination directory for output files
        - filter_type: "Godin", "30h", or "35h"
    Outputs:
        - list of absolute file names of output files
    """
    ncdata = xr.open_dataset(ncfile)

    if "L0" in ncfile:
        time_lim, bin_depths_lim, ns_lim, ew_lim = limit_data(ncdata, ncdata.VEL_MAGNETIC_EAST.data, ncdata.VEL_MAGNETIC_NORTH.data)
        # North/East velocity plots
        fname_ne = make_pcolor_ne(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim, True)

        # Along/Cross-shelf velocity plots
        fname_ac = make_pcolor_ac(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim)

    else:
        time_lim, bin_depths_lim, ns_lim, ew_lim = limit_data(ncdata, ncdata.LCEWAP01.data, ncdata.LCNSAP01.data)
        # North/East velocity plots
        fname_ne = make_pcolor_ne(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim)

        # Along/Cross-shelf velocity plots
        fname_ac = make_pcolor_ac(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim)

    # Redo whole process with filtered data

    if filter_type == "Godin":
        ew_filt, ns_filt = filter_godin(ncdata)
    elif filter_type == "30h":
        ew_filt, ns_filt = filter_XXh(ncdata, num_hrs=30)
    elif filter_type == "35h":
        ew_filt, ns_filt = filter_XXh(ncdata, num_hrs=35)
    else:
        ValueError("filter_type value not understood !")

    # Limit data
    time_lim, bin_depths_lim, ns_filt_lim, ew_filt_lim = limit_data(ncdata, ew_filt, ns_filt)

    # East/North
    if "L0" in ncfile:
        fname_ne_filt = make_pcolor_ne(ncdata, dest_dir, time_lim, bin_depths_lim, ns_filt_lim, ew_filt_lim, True,
                                       filter_type)
    else:
        fname_ne_filt = make_pcolor_ne(ncdata, dest_dir, time_lim, bin_depths_lim, ns_filt_lim, ew_filt_lim, False,
                                       filter_type)

    # Along-shore/cross-shore
    fname_ac_filt = make_pcolor_ac(ncdata, dest_dir, time_lim, bin_depths_lim, ns_filt_lim, ew_filt_lim, filter_type)

    # Compare velocity in bin 1
    if 'L0' in ncfile:
        fname_binplot = binplot_compare_filt(ncdata, dest_dir, time_lim, ew_lim, ew_filt_lim, filter_type,
                                             direction='magnetic_east')
    else:
        fname_binplot = binplot_compare_filt(ncdata, dest_dir, time_lim, ew_lim, ew_filt_lim, filter_type,
                                             direction='east')

    return [fname_ne, fname_ac, fname_ne_filt, fname_ac_filt, fname_binplot]
