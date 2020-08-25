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
    
Plots can be made from L0-, L1- or L2-processed data.
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


def fmamidir(u, v):
    # Computes principal component direction of u and v

    ub = np.nanmean(u)
    vb = np.nanmean(v)
    uu = np.nanmean(u ** 2)
    vv = np.nanmean(v ** 2)
    uv = np.nanmean(u * v)
    uu = uu - (ub * ub)
    vv = vv - (vb * vb)
    uv = uv - (ub * vb)

    # Solve for the quadratic
    a = 1.0
    b = -(uu + vv)
    c = (uu * vv) - (uv * uv)
    s1 = (-b + np.sqrt((b * b) - (4.0 * a * c)) / (2.0 * a))
    s2 = (-b - np.sqrt((b * b) - (4.0 * a * c)) / (2.0 * a))
    major = s1
    minor = s2
    if minor > major:
        major = s2
        minor = s1

    # Return major and minor axes
    return major, minor


def fpcdir(x, y):
    if x.shape != y.shape:
        ValueError('u and v are different sizes!')
    else:
        # Compute major and minor axes
        major, minor = fmamidir(x, y)

        # Compute principal component direction
        u = x
        v = y
        ub = np.nanmean(u)
        vb = np.nanmean(v)
        uu = np.nanmean(u ** 2)
        uv = np.nanmean(u * v)
        uu = uu - (ub * ub)
        uv = uv - (ub * vb)

        e1 = -uv / (uu - major)
        e2 = 1
        rad_deg = 180 / np.pi  # conversion factor
        theta = np.arctan2(e1, e2) * rad_deg
        theta = -theta  # change rotation angle to be CCW from North

    return theta


def calculate_depths(dataset):
    """
    Calculate ADCP bin depths in the water column
    Inputs:
        - dataset: dataset-type object created by reading in a netCDF ADCP file with the xarray package
    Outputs:
        - numpy array of ADCP bin depths
    """
    # depths = np.mean(ncdata.PRESPR01[0,:]) - ncdata.distance  #What Di used
    if dataset.orientation == 'up':
        return float(dataset.instrument_depth) - dataset.distance.data
    else:
        return float(dataset.instrument_depth) + dataset.distance.data


def vb_flag(dataset):
    """
    Create flag for missing vertical beam data in files from Sentinel V ADCPs
    flag = 0 if Sentinel V file has vertical beam data, or file not from Sentinel V
    flag = 1 if Sentinel V file does not have vertical beam data
    Inputs:
        - dataset: dataset-type object created by reading in a netCDF ADCP file with the xarray package
    Outputs:
        - value of flag
    """
    try:
        x = dataset.TNIHCE05.data
        return 0
    except AttributeError:
        if dataset.instrumentSubtype == 'Sentinel V':
            return 1
        else:
            return 0


def get_plot_dir(filename, dest_dir):
    if 'L0' in filename.data.tolist():
        plot_dir = './{}/L0_Python_plots/'.format(dest_dir)
    elif 'L1' in filename.data.tolist():
        plot_dir = './{}/L1_Python_plots/'.format(dest_dir)
    elif 'L2' in filename.data.tolist():
        plot_dir = './{}/L2_Python_plots/'.format(dest_dir)
    else:
        ValueError('Input netCDF file must be a L0, L1 or L2-processed file.')

    return plot_dir


def plots_diagnostic(nc, dest_dir, level0=False):
    """
    Preliminary plots:
    (1) Backscatter against depth, (2) mean velocity, and (3) principle component direction
    Credits: David Spear, Roy Hourston
    Inputs:
        - d: dataset-type object created by reading in a netCDF ADCP file with the xarray package
    Outputs:
        None
    """
    # Subplot 1/3: Plot avg backscatter against depth

    # First calculate depths
    depths = calculate_depths(nc)

    # Check if vertical beam data present in Sentinel V file
    flag_vb = vb_flag(nc)

    # Calculate average backscatter (amplitude intensity)
    amp_mean_b1 = np.nanmean(nc.TNIHCE01.data, axis=1)
    amp_mean_b2 = np.nanmean(nc.TNIHCE02.data, axis=1)
    amp_mean_b3 = np.nanmean(nc.TNIHCE03.data, axis=1)
    amp_mean_b4 = np.nanmean(nc.TNIHCE04.data, axis=1)

    if nc.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        amp_mean_b5 = np.nanmean(nc.TNIHCE05.data, axis=1)
        amp = [amp_mean_b1, amp_mean_b2, amp_mean_b3, amp_mean_b4, amp_mean_b5]
        colours = ['b', 'g', 'c', 'm', 'y'] #list of plotting colours
    else:
        # Done calculating time-averaged amplitude intensity
        amp = [amp_mean_b1, amp_mean_b2, amp_mean_b3, amp_mean_b4]
        colours = ['b', 'g', 'c', 'm']

    # Start plot

    # Make plot and first subplot
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(1, 3, 1)

    beam_no = 1
    for dat, col in zip(amp, colours):
        f1 = ax.plot(dat, depths, label='Beam {}'.format(beam_no), linewidth=1, marker='o', markersize=2, color=col)
        beam_no += 1
    ax.set_ylim(depths[-1], depths[0])
    ax.legend(loc='lower left')
    ax.set_xlabel('Counts')
    ax.set_ylabel('Depth (m)')  # Set y axis label for this subplot only out of the 3
    ax.grid()
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.set_title('Mean Backscatter', fontweight='semibold')  # subplot title
    # Flip over x-axis if instrument oriented 'up'
    if nc.orientation == 'up':
        plt.gca().invert_yaxis()

    # Subplot 2/3: Mean velocity
    ax = fig.add_subplot(1, 3, 2)

    # Calculate average velocities
    u_mean = np.zeros(int(nc.numberOfCells), dtype='float32')
    v_mean = np.zeros(int(nc.numberOfCells), dtype='float32')
    w_mean = np.zeros(int(nc.numberOfCells), dtype='float32')

    if level0:
        for i in range(len(u_mean)):
            u_mean[i] = np.nanmean(nc.VEL_MAGNETIC_EAST.data[i, :])
            v_mean[i] = np.nanmean(nc.VEL_MAGNETIC_NORTH.data[i, :])
            w_mean[i] = np.nanmean(nc.LRZAAP01.data[i, :])
    else:
        for i in range(len(u_mean)):
            u_mean[i] = np.nanmean(nc.LCEWAP01.data[i, :])
            v_mean[i] = np.nanmean(nc.LCNSAP01.data[i, :])
            w_mean[i] = np.nanmean(nc.LRZAAP01.data[i, :])

    if nc.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        w5_mean = np.zeros(int(nc.numberOfCells), dtype='float32')
        for i in range(len(u_mean)):
            w5_mean[i] = np.nanmean(nc.LRZUVP01.data[i, :])

        if level0:
            names = ['VEL_MAGNETIC_EAST', 'VEL_MAGNETIC_NORTH', 'LRZAAP01', 'LRZUVP01']
        else:
            names = ['LCEWAP01', 'LCNSAP01', 'LRZAAP01', 'LRZUVP01']
        vels = [u_mean, v_mean, w_mean, w5_mean]
    else:
        if level0:
            names = ['VEL_MAGNETIC_EAST', 'VEL_MAGNETIC_NORTH', 'LRZAAP01']
        else:
            names = ['LCEWAP01', 'LCNSAP01', 'LRZAAP01']
        vels = [u_mean, v_mean, w_mean]

    # Plot
    for i in range(len(names)):
        f2 = ax.plot(vels[i], depths, label=names[i], linewidth=1, marker='o', markersize=2, color=colours[i])
    ax.set_ylim(depths[-1], depths[0])  # set vertical limits
    ax.legend(loc='lower left')
    ax.set_xlabel('Velocity (m/s)')
    ax.grid()
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.set_title('Mean Velocity', fontweight='semibold')  # subplot title
    # Flip over x-axis if instrument oriented 'up'
    if nc.orientation == 'up':
        plt.gca().invert_yaxis()

    # Subplot 3/3: Principal axis

    orientation = np.zeros(int(nc.numberOfCells), dtype='float32')
    for ibin in range(len(orientation)):
        if level0:
            xx = nc.VEL_MAGNETIC_EAST.data[ibin, :]
            yy = nc.VEL_MAGNETIC_NORTH.data[ibin, :]
        else:
            xx = nc.LCEWAP01.data[ibin, :]
            yy = nc.LCNSAP01.data[ibin, :]
        orientation[ibin] = fpcdir(xx, yy)  # convert to CW direction

    mean_orientation = np.round(np.nanmean(orientation), decimals=1)
    middle_orientation = np.mean([np.nanmin(orientation), np.nanmax(orientation)])  #text plotting coordinate
    mean_depth = np.nanmean(depths)  #text plotting coordinate

    # Make the subplot
    ax = fig.add_subplot(1, 3, 3)
    f3 = ax.plot(orientation, depths, linewidth=1, marker='o', markersize=2)
    ax.set_ylim(depths[-1], depths[0])  # set vertical limits
    ax.set_xlabel('Orientation')
    ax.text(x=middle_orientation, y=mean_depth, s='Mean orientation = ' + str(mean_orientation) + '$^\circ$',
            horizontalalignment='center', verticalalignment='center', fontsize=10)
    ax.grid()  # set grid
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.set_title('Principal Axis', fontweight='semibold')  # subplot title
    # Flip over x-axis if instrument oriented 'up'
    if nc.orientation == 'up':
        plt.gca().invert_yaxis()

    # Create centred figure title
    fig.suptitle('{}-{} {} at {} m depth'.format(nc.station, nc.deployment_number, nc.serial_number,
                                                 nc.instrument_depth), fontweight='semibold')

    # Create plots subfolder
    plot_dir = get_plot_dir(nc.filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig_name = plot_dir + '{}-{}_{}_nc_diagnostic.png'.format(
        nc.station, str(nc.deployment_number), nc.serial_number)
    fig.savefig(fig_name)
    plt.close()

    return os.path.abspath(fig_name)


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


def get_vminvmax(v1_data, v2_data):
    """
    Get range of pcolor color bar
    Inputs:
        - v1_data, v2_data: 2 raw or filtered velocity data components (East and North or Along- and Cross-shore)
    Outputs:
        - 2-element list containing the color bar min and max
    """
    v1_std = np.nanstd(v1_data)
    v1_mean = np.nanmean(v1_data)
    v1_lim = np.max([np.abs(-(v1_mean + 2 * v1_std)), np.abs(v1_mean + 2 * v1_std)])
    v2_std = np.nanstd(v2_data)
    v2_mean = np.nanmean(v2_data)
    v2_lim = np.max([np.abs(-(v2_mean + 2 * v2_std)), np.abs(v2_mean + 2 * v2_std)])

    # determine which limit to use
    vel_lim = np.max([v1_lim, v2_lim])
    print(vel_lim)
    vminvmax = [-vel_lim, vel_lim]
    print(vminvmax)
    return vminvmax


def make_pcolor_ne(nc, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim, level0=False, filter_type='raw'):
    # filter_type options: 'raw' (default), '30h' (or, 35h, etc, average), 'Godin' (Godin Filtered)

    magnetic = '' #specify "Magnetic (North)" for L0 files, since magnetic declination wasn't applied to them
    if level0:
        magnetic = 'Magnetic '

    # vminvmax = [-0.5, 0.5] #vertical min and max of the colour bar in the plots
    vminvmax = get_vminvmax(ns_lim, ew_lim)
    fig = plt.figure(figsize=(13.75, 10))
    ax = fig.add_subplot(2, 1, 1)

    f1 = ax.pcolor(time_lim, bin_depths_lim, ns_lim[:, :], cmap='RdBu_r', vmin=vminvmax[0], vmax=vminvmax[1])
    cbar = fig.colorbar(f1, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)
    ax.set_ylabel('Depth [m]', fontsize=14)

    if filter_type == '30h':
        ax.set_title(
            'ADCP ({}North, 30h average) {}-{} {}m'.format(magnetic, nc.attrs['station'],
                                                           nc.attrs['deployment_number'],
                                                           str(int(nc.instrument_depth))), fontsize=14)
    elif filter_type == 'Godin':
        ax.set_title(
            'ADCP ({}North, Godin Filtered) {}-{} {}m'.format(magnetic, nc.attrs['station'],
                                                              nc.attrs['deployment_number'],
                                                              str(int(nc.instrument_depth))), fontsize=14)
    elif filter_type == 'raw':
        ax.set_title(
            'ADCP ({}North, raw) {}-{} {}m'.format(magnetic, nc.attrs['station'], nc.attrs['deployment_number'],
                                                   str(int(nc.instrument_depth))), fontsize=14)
    else:
        ValueError('Not a recognized data type; choose one of \'raw\', \'30h\' or \'Godin\'')

    if nc.orientation == 'up':
        plt.gca().invert_yaxis()

    ax2 = fig.add_subplot(2, 1, 2)

    f2 = ax2.pcolor(time_lim, bin_depths_lim, ew_lim[:, :], cmap='RdBu_r', vmin=vminvmax[0], vmax=vminvmax[1])
    cbar = fig.colorbar(f2, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)

    ax2.set_ylabel('Depth [m]', fontsize=14)
    if 'h' in filter_type:  # xxh-average; e.g. '30h', '35h'
        ax2.set_title('ADCP ({}East, {} average) {}-{} {}m'.format(magnetic, filter_type, nc.attrs['station'],
                                                                   nc.attrs['deployment_number'],
                                                                   str(int(nc.instrument_depth))), fontsize=14)
    elif filter_type == 'Godin':
        ax2.set_title('ADCP ({}East, Godin Filtered) {}-{} {}m'.format(magnetic, nc.attrs['station'],
                                                                       nc.attrs['deployment_number'],
                                                                       str(int(nc.instrument_depth))), fontsize=14)
    elif filter_type == 'raw':
        ax2.set_title(
            'ADCP ({}East, raw) {}-{} {}m'.format(magnetic, nc.attrs['station'], nc.attrs['deployment_number'],
                                                  str(int(nc.instrument_depth))), fontsize=14)

    if nc.orientation == 'up':
        plt.gca().invert_yaxis()

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    plot_dir = get_plot_dir(nc.filename, dest_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if level0:
        plot_name = plot_dir + nc.attrs['station'] + '-' + nc.attrs['deployment_number'] + '_{0}m'.format(
            str(int(nc.instrument_depth))) + '-magn_NE_{}.png'.format(filter_type)
    else:
        plot_name = plot_dir + nc.attrs['station'] + '-' + nc.attrs['deployment_number'] + '_{0}m'.format(
            str(int(nc.instrument_depth))) + '-NE_{}.png'.format(filter_type)
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


def make_pcolor_ac(data, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim, filter_type='raw', along_angle=None):
    # filter_type options: 'raw' (default), '30h' (or, 35h, etc, average), 'Godin' (Godin Filtered)
    # cross_angle in degrees; defaults to 25
    if along_angle is None:
        along_angle, cross_angle = determine_dom_angle(ew_lim, ns_lim)
    else:
        cross_angle = along_angle - 90  # deg
    print(along_angle, cross_angle)

    u_along, u_cross = resolve_to_alongcross(ew_lim, ns_lim, along_angle)
    AS = u_along
    CS = u_cross

    # vminvmax = [-0.5, 0.5] #vertical min and max of the colour bar in the plots
    vminvmax = get_vminvmax(AS, CS)
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

    # Create plots subfolder if not made already
    plot_dir = get_plot_dir(data.filename, dest_dir)
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
    plot_dir = get_plot_dir(nc.filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_name = plot_dir + nc.attrs['station'] + '-' + nc.attrs['deployment_number'] + '_{0}m'.format(
        str(math.ceil(nc.instrument_depth))) + '-{}_bin{}_compare_{}.png'.format(vel_code, bin_index + 1,
                                                                                 filter_type)
    fig.savefig(plot_name)
    plt.close()
    return os.path.abspath(plot_name)


def create_westcoast_plots(ncfile, dest_dir, filter_type="Godin", along_angle=None):
    """
    Inputs:
        - ncfile: file name of netCDF ADCP file
        - dest_dir: destination directory for output files
        - filter_type: "Godin", "30h", or "35h"
        - along_angle: Along-shore angle measured in degrees relative to geographic East, counter-clockwise
    Outputs:
        - list of absolute file names of output files
    """
    ncdata = xr.open_dataset(ncfile)

    if "L0" in ncfile:
        fname_diagnostic = plots_diagnostic(ncdata, dest_dir, True)

        time_lim, bin_depths_lim, ns_lim, ew_lim = limit_data(ncdata, ncdata.VEL_MAGNETIC_EAST.data, ncdata.VEL_MAGNETIC_NORTH.data)
        # North/East velocity plots
        fname_ne = make_pcolor_ne(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim, True)

        # Along/Cross-shelf velocity plots
        fname_ac = make_pcolor_ac(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim, 'raw', along_angle)

    else:
        fname_diagnostic = plots_diagnostic(ncdata, dest_dir)

        time_lim, bin_depths_lim, ns_lim, ew_lim = limit_data(ncdata, ncdata.LCEWAP01.data, ncdata.LCNSAP01.data)
        # North/East velocity plots
        fname_ne = make_pcolor_ne(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim)

        # Along/Cross-shelf velocity plots
        fname_ac = make_pcolor_ac(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim, 'raw', along_angle)


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
    fname_ac_filt = make_pcolor_ac(ncdata, dest_dir, time_lim, bin_depths_lim, ns_filt_lim, ew_filt_lim, filter_type, along_angle)

    # Compare velocity in bin 1
    if 'L0' in ncfile:
        fname_binplot = binplot_compare_filt(ncdata, dest_dir, time_lim, ew_lim, ew_filt_lim, filter_type,
                                             direction='magnetic_east')
    else:
        fname_binplot = binplot_compare_filt(ncdata, dest_dir, time_lim, ew_lim, ew_filt_lim, filter_type,
                                             direction='east')

    return [fname_diagnostic, fname_ne, fname_ac, fname_ne_filt, fname_ac_filt, fname_binplot]
