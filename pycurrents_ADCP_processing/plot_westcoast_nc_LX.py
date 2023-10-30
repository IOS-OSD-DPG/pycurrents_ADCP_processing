# -*- coding: utf-8 -*-

__author__ = 'diwan'

"""
This script contains functions that make several different kinds of plots of ADCP
netCDF-format data. The types of plots are:
    1. North and East current velocities (one plot containing a subplot for each)
    2. Along and cross-shelf current velocities (one plot containing a subplot for each)
    3. North and East Godin or 30h-averaged filtered current velocities
    4. Along and cross-shelf Godin or 30h-averaged filtered current velocities
    5. Bin plot for one month's worth of velocity data comparing raw and filtered
       (Godin or 30h-averaged) data
    6. Diagnostic plot containing subplots of:
        1. Time-averaged backscatter over depth
        2. Time-averaged velocity over depth
        3. Mean orientation
    7. Pressure (PRESPR01) vs time to help determine if/when boat strikes occur
    
Plots can be made from L0-, L1- or L2-processed data.

Credits:
Diagnostic plot code adapted from David Spear (originally in MatLab)
fmamidir() and fpcdir() function code from Roy Hourston (originally in MatLab)

From Roy Hourston's fpcdir.m script:
    "THETA = fpcdir(X,Y) is the principal component direction of X and Y.
    In other words, it is the angle of rotation counter-clockwise from
    north of the first principal component of X and Y. Applies to
    bivariate data set only. See Emery and Thomson, Data Analysis Methods
    in Oceanography, 1997, p.327, and Preisendorfer, Principal Component
    Analysis in Meteorology and Oceanography, 1988, p.15. Second principal
    angle is given by THETA + 90 degrees."
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd


def resolve_to_alongcross(u_true, v_true, along_angle):
    """
    Rotate North and East velocities to along- and cross-shore velocities given
    an along-shore angle
    :param u_true: East velocity data; array format
    :param v_true: North velocity data; array format
    :param along_angle: along-shore angle; measured in degrees counter-clockwise
           from geographic East
    :return: along-shore and cross-shore velocities in array format
    """

    along_angle = np.deg2rad(along_angle)

    u_along = u_true * np.cos(along_angle) + v_true * np.sin(along_angle)
    u_cross = u_true * np.sin(along_angle) - v_true * np.cos(along_angle)

    return u_along, u_cross


def get_L1_start_end(ncdata: xr.Dataset):
    """
    Obtain the number of leading and trailing ensembles that were set to nans in L1 processing
    from the processing_history global attribute.
    Then get the index of the last ensemble cut from the beginning of the time series
    and the index of the first ensemble cut from the end of the time series.
    ncdata: dataset-type object created by reading in a netCDF ADCP file with the xarray package
    Returns: a tuple of the form (a, b), where a is the index of the first good ensemble and b
             is the index of the last good ensemble
    """
    digits_in_process_hist = [int(s) for s in ncdata.attrs['processing_history'].split() if s.isdigit()]
    # The indices used are conditional upon the contents of the processing history remaining unchanged
    # in L1 and before L2. Appending to the end of the processing history is ok
    start_end_indices = (digits_in_process_hist[1], len(ncdata.time.data) - digits_in_process_hist[2] - 1)

    return start_end_indices


def fmamidir(u, v):
    """
    Computes principal component direction of u and v
    :param u: Eastward velocity in array format
    :param v: Northward velocity in array format
    :returns : major and minor axes
    """
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
    """
    Obtain the principal component angle of East and North velocity data
    :param x: Eastward velocity data in array format
    :param y: Northward velocity data in array format
    :return: principal orientation angle
    """
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


def calculate_depths(dataset: xr.Dataset):
    """
    Calculate ADCP bin depths in the water column
    Inputs:
        - dataset: dataset-type object created by reading in a netCDF ADCP
                   file with the xarray package
    Outputs:
        - numpy array of ADCP bin depths
    """
    # depths = np.mean(ncdata.PRESPR01[0,:]) - ncdata.distance  #What Di used
    if dataset.orientation == 'up':
        return float(dataset.instrument_depth) - dataset.distance.data
    else:
        return float(dataset.instrument_depth) + dataset.distance.data


def vb_flag(dataset: xr.Dataset):
    """
    Create flag for missing vertical beam data in files from Sentinel V ADCPs
    flag = 0 if Sentinel V file has vertical beam data, or file not from Sentinel V
    flag = 1 if Sentinel V file does not have vertical beam data
    Inputs:
        - dataset: dataset-type object created by reading in a netCDF ADCP file
                   with the xarray package
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
    """
    Function for creating the name of the output plot subdirectory, based on the
    processing level of the input netCDF file
    :param filename: name of input netCDF file containing ADCP data
    :param dest_dir: name of folder for all output files
    :return: name of subfolder for output files based on processing level of the
             input netCDF file
    """
    if 'L0' in filename.data.tolist():
        plot_dir = './{}/L0_Python_plots/'.format(dest_dir)
    elif 'L1' in filename.data.tolist():
        plot_dir = './{}/L1_Python_plots/'.format(dest_dir)
    elif 'L2' in filename.data.tolist():
        plot_dir = './{}/L2_Python_plots/'.format(dest_dir)
    else:
        ValueError('Input netCDF file must be a L0, L1 or L2-processed file.')
        return None

    return plot_dir


def plot_adcp_pressure(nc: xr.Dataset, dest_dir: str, resampled=None):
    """Plot pressure, PRESPR01, vs time
    :param nc: xarray dataset object from xarray.open_dataset(ncpath)
    :param dest_dir: name of subfolder to which plot will be saved
    :param resampled: "30min" if resampled to 30 minutes; None if not resampled
    """

    fig = plt.figure(dpi=400)

    plt.plot(nc.time.data, nc.PRESPR01.data)

    plt.gca().invert_yaxis()

    plt.ylabel("Pressure (dbar)")

    # Create plots subfolder
    plot_dir = get_plot_dir(nc.filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if resampled is None:
        plt.title("{}-{} {} PRESPR01".format(nc.station, nc.deployment_number,
                                             nc.instrument_serial_number.data))

        png_name = plot_dir + "{}-{}_{}_{}m_PRESPR01.png".format(
            nc.station, nc.deployment_number, nc.instrument_serial_number.data,
            int(np.round(nc.instrument_depth, 0)))
    else:
        plt.title("{}-{} {} PRESPR01 {} subsampled".format(
            nc.station, nc.deployment_number, nc.instrument_serial_number.data,
            resampled))

        png_name = plot_dir + "{}-{}_{}_{}m_PRESPR01_{}_subsamp.png".format(
            nc.station, nc.deployment_number, nc.instrument_serial_number.data,
            int(np.round(nc.instrument_depth, 0)), resampled)

    fig.savefig(png_name)
    plt.close(fig)

    return png_name


def plots_diagnostic(nc: xr.Dataset, dest_dir, level0=False, time_range=None, bin_range=None, resampled=None):
    """
    Preliminary plots:
    (1) Backscatter against depth, (2) mean velocity, and (3) principle component
    direction
    Credits: David Spear, Roy Hourston
    Inputs:
        - nc: dataset-type object created by reading in a netCDF ADCP file with the
              xarray package
        - dest_dir: name of folder to contain output files
        - level0 (optional): boolean value; True if nc is a level 0-processed dataset,
                             default False
        - time_range (optional): tuple of form (a, b), where a is the index of the
                                 first time stamp to include and b is the index of the
                                 last time stamp to include; default None
        - bin_range (optional): tuple of form (a, b), where a is the index of the
                                minimum bin to include and b is the index of the
                                maximum bin to include; default None
        - resampled (optional): "30min" if resampled to 30 minutes; None if not resampled
    Returns:
        Absolute path of the figure made by this function
    """
    if time_range is None:
        time_range = (None, None)
    if bin_range is None:
        bin_range = (None, None)

    # Subplot 1/3: Plot avg backscatter against depth

    # First calculate depths
    depths = calculate_depths(nc)[bin_range[0]:bin_range[1]]

    # Check if vertical beam data present in Sentinel V file
    flag_vb = vb_flag(nc)

    # Calculate average backscatter (amplitude intensity)
    amp_mean_b1 = np.nanmean(nc.TNIHCE01.data[bin_range[0]:bin_range[1],
                             time_range[0]:time_range[1]], axis=1)
    amp_mean_b2 = np.nanmean(nc.TNIHCE02.data[bin_range[0]:bin_range[1],
                             time_range[0]:time_range[1]], axis=1)
    amp_mean_b3 = np.nanmean(nc.TNIHCE03.data[bin_range[0]:bin_range[1],
                             time_range[0]:time_range[1]], axis=1)
    amp_mean_b4 = np.nanmean(nc.TNIHCE04.data[bin_range[0]:bin_range[1],
                             time_range[0]:time_range[1]], axis=1)

    if nc.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        amp_mean_b5 = np.nanmean(nc.TNIHCE05.data[bin_range[0]:bin_range[1],
                                 time_range[0]:time_range[1]], axis=1)
        amp = [amp_mean_b1, amp_mean_b2, amp_mean_b3, amp_mean_b4, amp_mean_b5]
        colours = ['b', 'g', 'c', 'm', 'y']  # list of plotting colours
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
        f1 = ax.plot(dat, depths, label='Beam {}'.format(beam_no), linewidth=1, marker='o',
                     markersize=2, color=col)
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
        ax.invert_yaxis()

    # Subplot 2/3: Mean velocity
    ax = fig.add_subplot(1, 3, 2)

    # Calculate average velocities (mean over time for each bin)
    if level0:
        u_mean = np.nanmean(nc.VEL_MAGNETIC_EAST.data[bin_range[0]:bin_range[1],
                            time_range[0]:time_range[1]], axis=1)
        v_mean = np.nanmean(nc.VEL_MAGNETIC_NORTH.data[bin_range[0]:bin_range[1],
                            time_range[0]:time_range[1]], axis=1)
        w_mean = np.nanmean(nc.LRZAAP01.data[bin_range[0]:bin_range[1],
                            time_range[0]:time_range[1]], axis=1)
    else:
        u_mean = np.nanmean(nc.LCEWAP01.data[bin_range[0]:bin_range[1],
                            time_range[0]:time_range[1]], axis=1)
        v_mean = np.nanmean(nc.LCNSAP01.data[bin_range[0]:bin_range[1],
                            time_range[0]:time_range[1]], axis=1)
        w_mean = np.nanmean(nc.LRZAAP01.data[bin_range[0]:bin_range[1],
                            time_range[0]:time_range[1]], axis=1)

    if nc.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        w5_mean = np.nanmean(nc.LRZUVP01.data[bin_range[0]:bin_range[1],
                             time_range[0]:time_range[1]], axis=1)

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

    # Plot averaged velocities over depth
    for i in range(len(names)):
        f2 = ax.plot(vels[i], depths, label=names[i], linewidth=1, marker='o', markersize=2,
                     color=colours[i])
    ax.set_ylim(depths[-1], depths[0])  # set vertical limits
    ax.legend(loc='lower left')
    ax.set_xlabel('Velocity (m/s)')
    ax.grid()
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.set_title('Mean Velocity', fontweight='semibold')  # subplot title
    # Flip over x-axis if instrument oriented 'up'
    if nc.orientation == 'up':
        ax.invert_yaxis()

    # Subplot 3/3: Principal axis

    orientation = np.zeros(len(depths), dtype='float32')
    if level0:
        xx = nc.VEL_MAGNETIC_EAST.data[bin_range[0]:bin_range[1], time_range[0]:time_range[1]]
        yy = nc.VEL_MAGNETIC_NORTH.data[bin_range[0]:bin_range[1], time_range[0]:time_range[1]]
    else:
        xx = nc.LCEWAP01.data[bin_range[0]:bin_range[1], time_range[0]:time_range[1]]
        yy = nc.LCNSAP01.data[bin_range[0]:bin_range[1], time_range[0]:time_range[1]]

    for ibin in range(len(xx)):
        orientation[ibin] = fpcdir(xx[ibin], yy[ibin])  # convert to CW direction

    mean_orientation = np.round(np.nanmean(orientation), decimals=1)
    # coordinate for plotting text
    middle_orientation = np.mean([np.nanmin(orientation), np.nanmax(orientation)])
    mean_depth = np.nanmean(depths)  # text plotting coordinate

    # Make the subplot
    ax = fig.add_subplot(1, 3, 3)
    f3 = ax.plot(orientation, depths, linewidth=1, marker='o', markersize=2)
    ax.set_ylim(depths[-1], depths[0])  # set vertical limits
    ax.set_xlabel('Orientation')
    ax.text(x=middle_orientation, y=mean_depth,
            s='Mean orientation = {}$^\circ$'.format(str(mean_orientation)),
            horizontalalignment='center', verticalalignment='center',
            fontsize=10)
    ax.grid()  # set grid
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.set_title('Principal Axis', fontweight='semibold')  # subplot title
    # Flip over x-axis if instrument oriented 'up'
    if nc.orientation == 'up':
        ax.invert_yaxis()

    # Create plots subfolder
    plot_dir = get_plot_dir(nc.filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Create centred figure title
    if resampled is None:
        fig.suptitle('{}-{} {} at {} m depth'.format(nc.station, nc.deployment_number, nc.serial_number,
                                                     np.round(nc.instrument_depth, 1)), fontweight='semibold')
        fig_name = plot_dir + '{}-{}_{}_{}m_diagnostic.png'.format(
            nc.station, str(nc.deployment_number), nc.serial_number,
            int(np.round(nc.instrument_depth, 0)))
    else:
        fig.suptitle('{}-{} {} at {} m depth, {} subsampled'.format(
            nc.station, nc.deployment_number, nc.serial_number,
            np.round(nc.instrument_depth, 1), resampled),
            fontweight='semibold')
        fig_name = plot_dir + '{}-{}_{}_{}m_diagnostic_{}_subsamp.png'.format(
            nc.station, str(nc.deployment_number), nc.serial_number,
            int(np.round(nc.instrument_depth, 0)), resampled)

    fig.savefig(fig_name)
    plt.close()

    return os.path.abspath(fig_name)


def limit_data(ncdata: xr.Dataset, ew_data, ns_data, time_range=None, bin_range=None):
    """
    Limits data to be plotted to only "good" data, either automatically or with user-input
    time and bin ranges
    :param ncdata: xarray dataset-type object containing ADCP data from a netCDF file
    :param ew_data: east-west velocity data
    :param ns_data: north-south velocity data
    :param time_range: optional; a tuple of the form (a, b) where a is the index of the first
                       good ensemble and b is the index of the last good ensemble in the dataset
    :param bin_range: optional; a tuple of the form (a, b) where a is the index of the minimum
                      good bin and b is the index of the maximum good bin in the dataset
    :return: time_lim, cleaned time data; bin_depths_lim; cleaned bin depth data; ns_lim,
             cleaned north-south velocity data; and ew_lim; cleaned east-west velocity data
    """
    if ncdata.orientation == 'up':
        bin_depths = ncdata.instrument_depth - ncdata.distance.data
    else:
        bin_depths = ncdata.instrument_depth + ncdata.distance.data
    # print(bin_depths)

    # data.time should be limited to the data.time with no NA values; bins must be limited
    if time_range is None:
        if 'L1' in ncdata.filename.data.tolist() or 'L2' in ncdata.filename.data.tolist():
            new_first_last = get_L1_start_end(ncdata=ncdata)
        else:
            new_first_last = (0, len(ew_data[0]))
    else:
        new_first_last = time_range

    # Remove bins where surface backscatter occurs
    time_lim = ncdata.time.data[new_first_last[0]:new_first_last[1]]

    if bin_range is None:
        bin_depths_lim = bin_depths[bin_depths >= 0]

        ew_lim = ew_data[bin_depths >= 0, new_first_last[0]:new_first_last[1]]  # Limit velocity data
        ns_lim = ns_data[bin_depths >= 0, new_first_last[0]:new_first_last[1]]
    else:
        bin_depths_lim = bin_depths[bin_range[0]: bin_range[1]]

        ew_lim = ew_data[bin_range[0]: bin_range[1], new_first_last[0]:new_first_last[1]]
        ns_lim = ns_data[bin_range[0]: bin_range[1], new_first_last[0]:new_first_last[1]]

    return time_lim, bin_depths_lim, ns_lim, ew_lim


def get_vminvmax(v1_data, v2_data=None):
    """
    Get range of pcolor color bar
    Inputs:
        - v1_data, v2_data: 2 raw or filtered velocity data components
          (East and North or Along- and Cross-shore)
    Outputs:
        - 2-element list containing the color bar min and max
    """
    v1_std = np.nanstd(v1_data)
    v1_mean = np.nanmean(v1_data)
    v1_lim = np.max([np.abs(-(v1_mean + 2 * v1_std)), np.abs(v1_mean + 2 * v1_std)])
    if v2_data is not None:
        v2_std = np.nanstd(v2_data)
        v2_mean = np.nanmean(v2_data)
        v2_lim = np.max([np.abs(-(v2_mean + 2 * v2_std)), np.abs(v2_mean + 2 * v2_std)])

        # determine which limit to use
        vel_lim = np.max([v1_lim, v2_lim])
    else:
        vel_lim = v1_lim
    # print(vel_lim)
    vminvmax = [-vel_lim, vel_lim]
    # print(vminvmax)
    return vminvmax


def make_pcolor_ne(nc: xr.Dataset, dest_dir, time_lim, bin_depths_lim,
                   ns_lim, ew_lim, level0=False,
                   filter_type='raw', colourmap_lim=None, resampled=None):
    """
    Function for plotting north and east velocities from ADCP data.
    :param nc: ADCP dataset from a netCDF file read in using the xarray package
    :param dest_dir: name of directory for containing output files
    :param time_lim: cleaned time data; array type
    :param bin_depths_lim: cleaned bin depth data; array type
    :param ns_lim: cleaned north-south velocity data; array type
    :param ew_lim: cleaned east-west velocity data; array type
    :param level0: boolean indicating whether the input dataset (nc) underwent L0
                   processing or not; default False
    :param filter_type: options are 'raw' (default), '30h' (or, 35h, etc, average),
                        or 'Godin' (Godin Filtered)
    :param colourmap_lim: user-input tuple of the form (a, b), where a is the minimum
                          colour map limit for the plot and b is the maximum colour map
                          limit for the plot (both floats); default is None in which
                          case the function chooses the colour map limits for the plot
    :param resampled: "30min" if resampled to 30 minutes; None if not resampled
    :return: Absolute file path of the figure this function creates
    """

    # specify "Magnetic (North)" for L0 files, since magnetic declination wasn't applied to them
    # Default North is geographic
    magnetic = ''
    if level0:
        magnetic = 'Magnetic '

    # For making figure naming easier
    if resampled is None:
        resampled_str = ''
        resampled_4fname = ''
    else:
        resampled_str = ', {} subsampled'.format(resampled)
        resampled_4fname = '_{}_subsamp'.format(resampled)

    if colourmap_lim is None:
        vminvmax = get_vminvmax(ns_lim, ew_lim)
    else:
        vminvmax = colourmap_lim

    fig = plt.figure(figsize=(13.75, 10))
    ax = fig.add_subplot(2, 1, 1)

    f1 = ax.pcolormesh(time_lim, bin_depths_lim, ns_lim[:, :], cmap='RdBu_r', vmin=vminvmax[0],
                       vmax=vminvmax[1], shading='auto')
    cbar = fig.colorbar(f1, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)
    ax.set_ylabel('Depth [m]', fontsize=14)

    if filter_type == '30h':
        ax.set_title(
            'ADCP ({}North, 30h average) {}-{} {}m{}'.format(
                magnetic, nc.attrs['station'], nc.attrs['deployment_number'],
                nc.instrument_depth, resampled_str), fontsize=14)
    elif filter_type == 'Godin':
        ax.set_title(
            'ADCP ({}North, Godin Filtered) {}-{} {}m{}'.format(
                magnetic, nc.attrs['station'], nc.attrs['deployment_number'],
                nc.instrument_depth, resampled_str), fontsize=14)
    elif filter_type == 'raw':
        ax.set_title(
            'ADCP ({}North, raw) {}-{} {}m{}'.format(
                magnetic, nc.attrs['station'], nc.attrs['deployment_number'],
                nc.instrument_depth, resampled_str), fontsize=14)
    else:
        ValueError('Not a recognized data type; choose one of \'raw\', \'30h\' or \'Godin\'')

    ax.invert_yaxis()

    ax2 = fig.add_subplot(2, 1, 2)

    f2 = ax2.pcolormesh(time_lim, bin_depths_lim, ew_lim[:, :], cmap='RdBu_r', vmin=vminvmax[0],
                        vmax=vminvmax[1], shading='auto')
    cbar = fig.colorbar(f2, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)

    ax2.set_ylabel('Depth [m]', fontsize=14)
    if 'h' in filter_type:  # xxh-average; e.g. '30h', '35h'
        ax2.set_title('ADCP ({}East, {} average) {}-{} {}m{}'.format(
            magnetic, filter_type, nc.attrs['station'], nc.attrs['deployment_number'],
            nc.instrument_depth, resampled_str), fontsize=14)
    elif filter_type == 'Godin':
        ax2.set_title('ADCP ({}East, Godin Filtered) {}-{} {}m{}'.format(
            magnetic, nc.attrs['station'], nc.attrs['deployment_number'],
            nc.instrument_depth, resampled_str), fontsize=14)
    elif filter_type == 'raw':
        ax2.set_title(
            'ADCP ({}East, raw) {}-{} {}m{}'.format(
                magnetic, nc.attrs['station'], nc.attrs['deployment_number'],
                nc.instrument_depth, resampled_str), fontsize=14)

    ax2.invert_yaxis()

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    plot_dir = get_plot_dir(nc.filename, dest_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if level0:
        plot_name = plot_dir + nc.attrs['station'] + '-' + nc.attrs['deployment_number'] + '_{0}m'.format(
            int(np.round(nc.instrument_depth))) + '-magn_NE_{}{}.png'.format(filter_type, resampled_4fname)
    else:
        plot_name = plot_dir + nc.attrs['station'] + '-' + nc.attrs['deployment_number'] + '_{0}m'.format(
            int(np.round(nc.instrument_depth))) + '-NE_{}{}.png'.format(filter_type, resampled_4fname)
    fig.savefig(plot_name)
    plt.close()

    return os.path.abspath(plot_name)


def make_pcolor_speed(dest_dir: str, station: str, deployment_number: str, instrument_depth: int,
                      time_lim: np.ndarray, bin_depths_lim: np.ndarray,
                      ns_lim: np.ndarray, ew_lim: np.ndarray, resampled=None, colourmap_lim: tuple = None):
    """
    Make subplots of speed and direction for unfiltered ADCP data
    :param dest_dir: name of directory for containing output files
    :param station: station name
    :param deployment_number: deployment number for mooring
    :param instrument_depth:
    :param time_lim: cleaned time data; array type
    :param bin_depths_lim: cleaned bin depth data; array type
    :param ns_lim: cleaned north-south velocity data; array type
    :param ew_lim: cleaned east-west velocity data; array type
    :param resampled: "30min" if resampled to 30 minutes; None if not resampled
    :param colourmap_lim: user-input tuple of the form (a, b), where a is the minimum
                          colour map limit for the plot and b is the maximum colour map
                          limit for the plot (both floats); default is None in which
                          case the function chooses the colour map limits for the plot
    :return: Absolute file path of the figure this function creates
    """

    speed = np.repeat(np.nan, len(ns_lim.flatten())).reshape(ns_lim.shape)
    direction = np.repeat(np.nan, len(ns_lim.flatten())).reshape(ns_lim.shape)

    for i in range(len(ns_lim[:, 0])):
        for j in range(len(ns_lim[0, :])):
            speed[i, j] = (ns_lim[i, j] ** 2 + ew_lim[i, j] ** 2) ** .5
            direction[i, j] = np.rad2deg(np.arctan(ns_lim[i, j] / ew_lim[i, j]))  # degrees CCW from East

    # Use the upper bound but set the bottom bound to zero since speed can't be negative
    if colourmap_lim is None:
        vminvmax = get_vminvmax(ns_lim, ew_lim)
    else:
        vminvmax = colourmap_lim

    # Make a plot with subplots for speed and direction
    # Use a circular color map for direction
    fig = plt.figure(figsize=(13.75, 10))

    ax = fig.add_subplot(2, 1, 1)
    f1 = ax.pcolormesh(time_lim, bin_depths_lim, speed, cmap='Greens', shading='auto',
                       vmin=0, vmax=vminvmax[1])
    cbar = fig.colorbar(f1, shrink=0.8)
    cbar.set_label('Speed [m s$^{-1}$]', fontsize=14)
    ax.set_ylabel('Depth [m]', fontsize=14)
    ax.invert_yaxis()
    ax.set_title(f'ADCP current speed {station}-{deployment_number} {instrument_depth}m')

    ax2 = fig.add_subplot(2, 1, 2)
    f2 = ax2.pcolormesh(time_lim, bin_depths_lim, direction, cmap='hsv', vmin=0,
                        vmax=180, shading='auto')
    cbar = fig.colorbar(f2, shrink=0.8)
    cbar.set_label('Direction [degrees]', fontsize=14)
    ax2.set_ylabel('Depth [m]', fontsize=14)
    ax2_title = f'ADCP current direction (CCW from East) {station}-{deployment_number} {instrument_depth}m'
    ax2.set_title(ax2_title)

    plot_name = f'{station}-{deployment_number}_{instrument_depth}m_spd_dir.png'
    if resampled:
        plot_name.replace('.png', f'_{resampled}_resampled.png')

    plot_name = os.path.join(dest_dir, plot_name)
    fig.savefig(plot_name)
    plt.close(fig)

    return plot_name


def determine_dom_angle(u_true, v_true):
    """
    Determine the dominant angle in degrees. The along_angle measured in degrees relative to
    geographic East, counter-clockwise.
    :param u_true: Eastward velocity data relative to geographic North
    :param v_true: Northward velocity data relative to geographic North
    """
    angles = np.arange(0, 180)
    max_rms = 0.
    max_angle = 0.
    for angle in angles:
        along_angle = np.deg2rad(angle)

        u_along = u_true * np.cos(along_angle) + v_true * np.sin(along_angle)
        rms = np.sqrt(np.nanmean(u_along * u_along))
        if rms > max_rms:
            max_rms = rms
            max_angle = angle  # in degrees
    along_angle = max_angle
    cross_angle = max_angle - 90.
    return along_angle, cross_angle


def make_pcolor_ac(data: xr.Dataset, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim,
                   filter_type='raw', along_angle=None, colourmap_lim=None, resampled=None):
    """
    Function for plotting north and east velocities from ADCP data.
    :param data: ADCP dataset from a netCDF file read in using the xarray package
    :param dest_dir: name of directory for containing output files
    :param time_lim: cleaned time data; array type
    :param bin_depths_lim: cleaned bin depth data; array type
    :param ns_lim: cleaned north-south velocity data; array type
    :param ew_lim: cleaned east-west velocity data; array type
    :param filter_type: options are 'raw' (default), '30h' (or, 35h, etc, average), or 'Godin'
                        (Godin Filtered)
    :param along_angle: along-shore angle in degrees
    :param colourmap_lim: user-input tuple of the form (a, b), where a is the minimum colour map
                          limit for the plot and b is the maximum colour map limit for the plot
                          (both floats); default is None in which case the function chooses the
                          colour map limits for the plot
    :param resampled: "30min" if resampled to 30 minutes; None if not resampled
    :return: Absolute file path of the figure this function creates
    """

    # For making figure naming easier
    if resampled is None:
        resampled_str = ''
        resampled_4fname = ''
    else:
        resampled_str = ', {} subsampled'.format(resampled)
        resampled_4fname = '_{}_subsamp'.format(resampled)

    # filter_type options: 'raw' (default), '30h' (or, 35h, etc, average), 'Godin' (Godin Filtered)
    # cross_angle in degrees; defaults to 25
    if along_angle is None:
        along_angle, cross_angle = determine_dom_angle(ew_lim, ns_lim)
    else:
        cross_angle = along_angle - 90  # deg
    # print(along_angle, cross_angle)

    u_along, u_cross = resolve_to_alongcross(ew_lim, ns_lim, along_angle)
    AS = u_along
    CS = u_cross

    if colourmap_lim is None:
        vminvmax = get_vminvmax(AS, CS)
    else:
        vminvmax = colourmap_lim

    fig = plt.figure(figsize=(13.75, 10))
    ax1 = fig.add_subplot(2, 1, 1)

    f1 = ax1.pcolormesh(time_lim, bin_depths_lim, AS[:, :], cmap='RdBu_r', vmin=vminvmax[0],
                        vmax=vminvmax[1], shading='auto')
    cbar = fig.colorbar(f1, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)

    ax1.set_ylabel('Depth [m]', fontsize=14)
    if 'h' in filter_type:
        # XXh-type filter (e.g., 30h rolling mean, etc)
        ax1.set_title(
            'ADCP (along, {} average) {}$^\circ$ (CCW from E) {}-{} {}m{}'.format(
                filter_type, along_angle, data.attrs['station'], data.attrs['deployment_number'],
                data.instrument_depth, resampled_str),
            fontsize=14)
    elif filter_type == 'Godin':
        ax1.set_title(
            'ADCP (along, Godin Filtered) {}$^\circ$ (CCW from E) {}-{} {}m{}'.format(
                along_angle, data.attrs['station'], data.attrs['deployment_number'],
                data.instrument_depth, resampled_str),
            fontsize=14)
    elif filter_type == 'raw':
        ax1.set_title('ADCP (along, raw) {}$^\circ$ (CCW from E) {}-{} {}m{}'.format(
            along_angle, data.attrs['station'], data.attrs['deployment_number'],
            data.instrument_depth, resampled_str),
            fontsize=14)
    else:
        ValueError('Not a recognized data type; choose one of \'raw\', \'30h\' or \'Godin\'')

    ax1.invert_yaxis()

    ax2 = fig.add_subplot(2, 1, 2)

    f2 = ax2.pcolormesh(time_lim, bin_depths_lim, CS[:, :], cmap='RdBu_r', vmin=vminvmax[0],
                        vmax=vminvmax[1], shading='auto')
    cbar = fig.colorbar(f2, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)

    ax2.set_ylabel('Depth [m]', fontsize=14)
    if 'h' in filter_type:  # xxh-average; e.g. '30h', '35h'
        ax2.set_title(
            'ADCP (cross, {} average) {}$^\circ$ (CCW from E) {}-{} {}m{}'.format(
                filter_type, cross_angle, data.attrs['station'], data.attrs['deployment_number'],
                data.instrument_depth, resampled_str),
            fontsize=14)
    elif filter_type == 'Godin':
        ax2.set_title(
            'ADCP (cross, Godin Filtered) {}$^\circ$ (CCW from E) {}-{} {}m{}'.format(
                str(cross_angle), data.attrs['station'], data.attrs['deployment_number'],
                data.instrument_depth, resampled_str),
            fontsize=14)
    elif filter_type == 'raw':
        ax2.set_title(
            'ADCP (cross, raw) {}$^\circ$ (CCW from E) {}-{} {}m{}'.format(
                str(cross_angle), data.attrs['station'], data.attrs['deployment_number'],
                data.instrument_depth, resampled_str),
            fontsize=14)
    else:
        ValueError('Not a recognized data type; choose one of \'raw\', \'30h\' or \'Godin\'')

    ax2.invert_yaxis()

    # Create plots subfolder if not made already
    plot_dir = get_plot_dir(data.filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_name = data.attrs['station'] + '-' + data.attrs['deployment_number'] + '_{}m'.format(
        int(np.round(data.instrument_depth))) + '-AC_{}{}.png'.format(filter_type, resampled_4fname)
    fig.savefig(plot_dir + plot_name)
    plt.close()

    return os.path.abspath(plot_dir + plot_name)


def num_ens_per_hr(nc: xr.Dataset):
    """
    Calculate the number of ensembles recorded per hour
    :param nc: dataset object obtained from reading in an ADCP netCDF file with the
               xarray package
    """
    time_incr = float(nc.time.data[1] - nc.time.data[0])
    hr2min = 60
    min2sec = 60
    sec2nsec = 1e9
    hr2nsec = hr2min * min2sec * sec2nsec
    return int(np.round(hr2nsec / time_incr, decimals=0))


def filter_godin(nc: xr.Dataset):
    """
    Make North and East lowpassed plots using the simple 3-day Godin filter
    Running average of 24 hours first, then another time with a 24 hour filter,
    then again with a 25 hour filter. Repeat with along- and cross-shore velocities
    :param nc: xarray Dataset-type object from reading in a netCDF file; contains 1D
               numpy array of values
    :returns : filtered East and North velocity data called ew_filter_final and
               ns_filter_final; 1D numpy arrays of values of same size as the time
               series within nc, padded with nan's
    """
    # Determine the number of ensembles taken per hour
    ens_per_hr = num_ens_per_hr(nc)
    # print('Time stamps per hour:', ens_per_hr, sep=' ')
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
    ew_filt_temp1 = ew_df.rolling(window=num_hrs * window, min_periods=None, center=True,
                                  win_type=None).median()
    ns_filt_temp1 = ns_df.rolling(window=num_hrs * window, min_periods=None, center=True,
                                  win_type=None).median()
    # 24h rolling average
    ew_filt_temp2 = ew_filt_temp1.rolling(window=num_hrs * window, min_periods=None, center=True,
                                          win_type=None).median()
    ns_filt_temp2 = ns_filt_temp1.rolling(window=num_hrs * window, min_periods=None, center=True,
                                          win_type=None).median()
    # 25h rolling average
    ew_filt_final = ew_filt_temp2.rolling(window=(num_hrs + 1) * window, min_periods=None,
                                          center=True, win_type=None).median()
    ns_filt_final = ns_filt_temp2.rolling(window=(num_hrs + 1) * window, min_periods=None,
                                          center=True, win_type=None).median()

    # Convert to numpy arrays
    ew_filt_final = ew_filt_final.to_numpy().transpose()
    ns_filt_final = ns_filt_final.to_numpy().transpose()

    return ew_filt_final, ns_filt_final


def filter_XXh(nc: xr.Dataset, num_hrs=30):
    """
    Perform XXh averaging on velocity data (30-hour, 35-hour, ...)
    :param nc: xarray Dataset-type object from reading in a netCDF file; contains 1D numpy
    array of values
    :param num_hrs: Number of hours to use in the rolling average; default is 30 hours
    :returns : filtered East and North velocity data called ew_filter_final and ns_filter_final;
    1D numpy arrays of values of same size as the time series within nc, padded with nan's
    """

    # Determine the number of ensembles taken per hour
    ens_per_hr = num_ens_per_hr(nc)
    # print(ens_per_hr)
    window = int(ens_per_hr)

    # Rolling window calculations
    # Need to take transpose so that the rolling average is taken along the time (not bin) axis
    if 'L0' in nc.filename.data.tolist():
        ew_df = pd.DataFrame(data=nc.VEL_MAGNETIC_EAST.data.transpose())
        ns_df = pd.DataFrame(data=nc.VEL_MAGNETIC_NORTH.data.transpose())
    else:
        ew_df = pd.DataFrame(data=nc.LCEWAP01.data.transpose())
        ns_df = pd.DataFrame(data=nc.LCNSAP01.data.transpose())
    ew_filt_final = ew_df.rolling(window=num_hrs * window, min_periods=None, center=True,
                                  win_type=None).median()
    ns_filt_final = ns_df.rolling(window=num_hrs * window, min_periods=None, center=True,
                                  win_type=None).median()

    # Convert to numpy arrays
    ew_filt_final = ew_filt_final.to_numpy().transpose()
    ns_filt_final = ns_filt_final.to_numpy().transpose()

    return ew_filt_final, ns_filt_final


def binplot_compare_filt(nc: xr.Dataset, dest_dir, time, dat_raw, dat_filt, filter_type, direction,
                         resampled=None):
    """
    Function to take one bin from the unfiltered (raw) data and the corresponding bin in
    the filtered data, and plot the time series together on one plot. Restrict time
    series to 1 month.
    :param nc: xarray dataset-type object containing ADCP data from a netCDF file
    :param dest_dir: destination directory for output files
    :param time: time data
    :param dat_raw: raw velocity data
    :param dat_filt: velocity data filtered using the method defined in filter_type
    :param filter_type: options are 'Godin' or 'xxh' (e.g., '30h', '35h')
    :param direction: 'east' or 'north'
    :param resampled: "30min" if resampled to 30 minutes; None if not resampled
    """

    if direction == 'magnetic_east':
        vel_code = 'VEL_MAGNETIC_EAST'
    elif direction == 'magnetic_north':
        vel_code = 'VEL_MAGNETIC_NORTH'
    elif direction == 'east':
        vel_code = 'LCEWAP01'
    elif direction == 'north':
        vel_code = 'LCNSAP01'
    else:
        ValueError('direction', direction, 'not valid')

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
        f2 = ax1.plot(time[:ens_per_mth], dat_filt[bin_index, :ens_per_mth], color='b',
                      label='Godin Filtered')
    elif 'h' in filter_type:
        f2 = ax1.plot(time[:ens_per_mth], dat_filt[bin_index, :ens_per_mth], color='b',
                      label='{} average'.format(filter_type))

    ax1.set_ylabel('Velocity [m s$^{-1}$]', fontsize=14)
    ax1.legend(loc='lower left')

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    plot_dir = get_plot_dir(nc.filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if resampled is None:
        ax1.set_title(
            'ADCP {}-{} {} bin {} at {}m'.format(nc.attrs['station'], nc.attrs['deployment_number'],
                                                 vel_code, bin_index + 1, bin_depth), fontsize=14)
        plot_name = nc.attrs['station'] + '-' + nc.attrs['deployment_number'] + '_{0}m'.format(
            str(math.ceil(nc.instrument_depth))) + '-{}_bin{}_compare_{}.png'.format(
            vel_code, bin_index + 1, filter_type)
    else:
        ax1.set_title('ADCP {}-{} {} bin {} at {}m, {} subsampled'.format(
            nc.attrs['station'], nc.attrs['deployment_number'], vel_code, bin_index + 1, bin_depth,
            resampled), fontsize=14)
        plot_name = nc.attrs['station'] + '-' + nc.attrs['deployment_number'] + '_{0}m'.format(
            str(math.ceil(nc.instrument_depth))) + '-{}_bin{}_compare_{}_{}_subsamp.png'.format(
            vel_code, bin_index + 1, filter_type, resampled)

    fig.savefig(plot_dir + plot_name)
    plt.close()
    return os.path.abspath(plot_dir + plot_name)


def resample_adcp_interp(ncname, ncdata: xr.Dataset):
    # Code from Di Wan
    # Not implemented because this method "creates" data
    u1 = ncdata.resample(time='5min').interpolate("linear")
    # print('u1', u1)
    # print(u1.time)
    # u2 = u1.resample(time='5min').mean(dim='time', keep_attrs=True)
    u2 = u1.rolling(time=6, center=True).mean(keep_attrs=True)
    # print('u2', u2)
    u3 = u2.resample(time='1H').interpolate("linear").transpose("distance", "time")
    # print('u3', u3)
    ncout_name = ncname.replace('.adcp', '_resampled.adcp')

    # Add global attributes from ncdata to u3
    for key, value in ncdata.attrs.items():
        u3.attrs[key] = value
    # Add variable attributes from ncdata to u3
    for varname in ncdata.data_vars:
        if varname in u3.data_vars:
            for key, value in ncdata[varname].attrs.items():
                u3[varname].attrs[key] = value
        else:
            print('Warning: {} not in resampled adcp file'.format(varname))

    u3.to_netcdf(ncout_name)
    return ncout_name


def resample_adcp_manual(ncname, ncdata: xr.Dataset, dest_dir):
    """Resample netCDF ADCP file for plotting by extracting one ensemble (measurement)
    per 30 min
    :param ncname: Full name including path of the netCDF file
    :param ncdata: netCDF dataset object that was created by xarray.open_dataset()
    :param dest_dir: destination directory; the name of the subfolder in which files
                     will be output
    """

    # Convert time variable to pandas datetime format in order to access minute and time
    time_pd = pd.to_datetime(ncdata.time.data)

    # Take first difference of time to get intervals lengths between measurements
    time_diff = np.diff(time_pd)  # nanoseconds
    median_interval = np.nanmedian(time_diff).astype('float')  # ns

    # Minute to nanosecond converter
    min2ns = 60 * 1e9
    desired_interval_min = 30  # 30 minutes

    # Number of ensembles per half hour
    # Corrected 2023-01-24
    # num_per_30min = int(np.floor(median_interval / min2ns * desired_interval_min))
    num_per_30min = int(np.floor(1 / median_interval * min2ns * desired_interval_min))

    # Subset time first so that we can check that diffs are constant
    time_subset = time_pd[0::num_per_30min]

    # Take first difference
    time_diff = np.diff(time_subset).astype('int64')

    # Find where time differences are less than 29 min or greater than 31 min
    check_diff = np.where(
        (time_diff < (desired_interval_min - 1) * min2ns) |
        (time_diff > (desired_interval_min + 1) * min2ns))[0]

    # print(len(check_diff))
    # print(check_diff)

    if len(check_diff) > 0:
        print('Warning: data subsampling intervals are not even')

    # # Get minute and second of the first time stamp
    # min1 = pd.to_datetime(time_pd[0]).minute
    # sec1 = pd.to_datetime(time_pd[0]).second
    #
    # if min1 < 30:
    #     offset_30min = 30
    # else:
    #     offset_30min = -30
    #
    # # Get 30min measurements
    # # Measurements not always exactly on the minute
    # # e.g., 29min59sec away instead of 30min00sec
    # subsampler = np.where(((time_pd.minute == min1) |
    #                        (time_pd.minute == min1 + offset_30min)) &
    #                       (time_pd.second == sec1))[0]
    #
    # print(len(subsampler))
    # print(subsampler)

    ncout = xr.Dataset(
        coords={'time': ncdata.time.data[0::num_per_30min], 'distance': ncdata.distance.data})

    # Iterate through coordinates
    for coordname in ncout.coords:
        # print(coordname)
        # Iterate through the variable's encoding
        for key, value in ncdata[coordname].encoding.items():
            ncout[coordname].encoding[key] = value

        # Iterate through the variable's attrs
        for key, value in ncdata[coordname].attrs.items():
            ncout[coordname].attrs[key] = value

    # Iterate through the variables in the input netCDF dataset
    for varname in ncdata.data_vars:
        if len(ncdata[varname].data.shape) == 2:
            ncout = ncout.assign({varname: (['distance', 'time'],
                                            ncdata[varname].data[:, 0::num_per_30min])})
        elif len(ncdata[varname].data.shape) == 1 and ncdata[varname].dims[0] == 'time':
            ncout = ncout.assign({varname: (['time'], ncdata[varname].data[0::num_per_30min])})
        elif len(ncdata[varname].data.shape) == 1 and ncdata[varname].dims[0] == 'distance':
            ncout = ncout.assign({varname: (['distance'], ncdata[varname].data)})
        elif len(ncdata[varname].data.shape) == 0:
            ncout = ncout.assign({varname: ([], ncdata[varname].data)})

        # Iterate through the variable's encoding
        for key, value in ncdata[varname].encoding.items():
            ncout[varname].encoding[key] = value

        # Iterate through the variable's attrs
        for key, value in ncdata[varname].attrs.items():
            ncout[varname].attrs[key] = value

    # Add global attrs
    for key, value in ncdata.attrs.items():
        ncout.attrs[key] = value

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    plot_dir = get_plot_dir(ncdata.filename, dest_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    ncout_name = os.path.basename(ncname).replace('.adcp', '_30min_subsamp.adcp')
    # outdir = '/home/hourstonh/Downloads/'
    ncout_path = plot_dir + ncout_name
    ncout.to_netcdf(ncout_path)
    ncout.close()
    ncdata.close()

    return ncout_path


def quiver_plot(dest_dir: str, station: str, deployment_number: str, instrument_depth: int,
                time_lim: np.ndarray, bin_depths_lim: np.ndarray,
                ns_lim: np.ndarray, ew_lim: np.ndarray, resampled=None):
    """
    Make subplots of speed and direction for unfiltered ADCP data
    :param dest_dir: name of directory for containing output files
    :param station: station name
    :param deployment_number: deployment number for mooring
    :param instrument_depth: instrument depth
    :param time_lim: cleaned time data; array type
    :param bin_depths_lim: cleaned bin depth data; array type
    :param ns_lim: cleaned north-south velocity data; array type
    :param ew_lim: cleaned east-west velocity data; array type
    :param resampled: "30min" if resampled to 30 minutes; None if not resampled
    :return: Absolute file path of the figure(s) this function creates
    """

    speed = np.repeat(np.nan, len(ns_lim.flatten())).reshape(ns_lim.shape)
    # direction = np.repeat(np.nan, len(ns_lim.flatten())).reshape(ns_lim.shape)

    for i in range(len(ns_lim[:, 0])):
        for j in range(len(ns_lim[0, :])):
            speed[i, j] = (ns_lim[i, j] ** 2 + ew_lim[i, j] ** 2) ** .5
            # direction[i, j] = np.rad2deg(np.arctan(ns_lim[i, j] / ew_lim[i, j]))  # degrees CCW from East

    # Plots will have 6 rows and 2 columns of subplots for a total of 12 subplots
    num_bins = len(ns_lim[:, 0])
    num_figures = int(np.ceil(num_bins / 12))
    zeros = np.zeros(time_lim.size)

    # Invert the depth bins to go shallowest to deepest
    bin_depths_lim = bin_depths_lim[::-1]
    ew_lim = ew_lim[::-1, :]
    ns_lim = ns_lim[::-1, :]
    speed = speed[::-1, :]
    # direction = direction[::-1, :]

    plot_list = []

    for k in range(num_figures):
        fig = plt.figure(figsize=(15, 30))
        fig.subplots_adjust(hspace=0.75, wspace=0.35)

        first_bin = 12 * k + 1
        last_bin = 12 * k + 12

        for i in range(1, 13):
            idx = 12 * k + i - 1
            if idx < len(bin_depths_lim):
                ax = fig.add_subplot(6, 2, i)
                ax.fill_between(time_lim, speed[idx, :], zeros, color='grey', alpha=0.1)
                ax.quiver(time_lim, zeros, ew_lim[idx, :], ns_lim[idx, :], color='blue',
                          width=0.004, units='y', scale_units='y', scale=1)
                ax.set_ylim(-0.60, 0.60)
                ax.set_title('Bin depth: ' + str(np.around(bin_depths_lim[idx], 1)) + 'm', fontsize=8,
                             fontweight='semibold')
                ax.set_ylabel('Speed (m/s)', size=8)
                ax.yaxis.labelpad = 0
                ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6], minor=False)
                ax.tick_params(axis='both', direction='in', top=True, right=True, labelsize=7)
                # p = ax.add_patch(plt.Rectangle((1, 1), 1, 1, fc='k', alpha=0.1))
                # leg = ax.legend([p], ["Current magnitude (m/s)"], loc='lower right', fontsize=8)
                # leg._drawFrame = False
            else:
                last_bin = idx - 1

        plt.suptitle(f'{station}-{deployment_number} Current Speed (m/s)')
        plot_name = f'{station}-{deployment_number}_{instrument_depth}m_quiver_bins{first_bin}-{last_bin}.png'
        if resampled:
            plot_name.replace('.png', f'_{resampled}_resampled.png')
        plot_name = os.path.join(dest_dir, plot_name)
        plt.savefig(plot_name)
        plt.close()
        plot_list.append(plot_name)

    return plot_list


def create_westcoast_plots(ncfile, dest_dir, filter_type="Godin", along_angle=None,
                           time_range=None, bin_range=None, colourmap_lim=None,
                           override_resample=False):
    """
    Inputs:
        - ncfile: file name of netCDF ADCP file
        - dest_dir: destination directory for output files
        - filter_type: "Godin", "30h", or "35h"
        - along_angle: Along-shore angle measured in degrees relative to geographic East,
                       counter-clockwise; can be user-input but defaults to None, in which
                       case the function calculates the along-shore angle
        - time_range (optional): tuple of form (a, b), where a is the index of the first
                                 time stamp to include and b is the index of the last time
                                 stamp to include; default None
        - bin_range (optional): tuple of form (a, b), where a is the index of the minimum
                                bin to include and b is the index of the maximum bin to
                                include; default None
        - colourmap_lim (optional): user-input tuple of the form (a, b), where a is the
                                    minimum colour map limit for the plot and b is the
                                    maximum colour map limit for the plot (both floats);
                                    default is None in which case the function chooses the
                                    colour map limits for the plot
        - override_resample (optional): ADCP netCDF files exceeding 100 MB in size will be
                                        resampled unless *True*, so as to not crash the web
                                        app. Default False
    Outputs:
        - list of absolute file names of output files
    """
    # Initialize list to hold the full names of all files produced by this function
    fout_name_list = []

    ncdata = xr.open_dataset(ncfile)

    # Check that time range and bin range limits are not out of range
    if time_range is not None:
        if time_range[1] > len(ncdata.time.data):
            UserWarning('User-input time range is out of range for input dataset; setting '
                        'value to None for automatic recalculation')
            print('Upper time index limit', time_range[1], '>', len(ncdata.time.data))
            time_range = None

    if bin_range is not None:
        if bin_range[1] > len(ncdata.distance.data):
            UserWarning('User-input bin range is out of range for input dataset; setting '
                        'value to None for automatic recalculation')
            print('Upper bin index limit', bin_range[1], '>', len(ncdata.distance.data))
            bin_range = None

    if "L0" in ncfile:
        level0 = True
        direction = 'magnetic_east'
    else:
        level0 = False
        direction = 'east'

    # Check size of file
    ncsize = os.path.getsize(ncfile)  # bytes
    byte2MB = 1. / 1e6
    threshold = 100  # Megabytes
    # If size exceeds threshold, do subsampling
    if ncsize * byte2MB > threshold and not override_resample:
        ncname_resampled = resample_adcp_manual(ncfile, ncdata, dest_dir)
        # Add path to file list
        fout_name_list.append(ncname_resampled)
        # Re-open dataset
        ncdata = xr.open_dataset(ncname_resampled)
        resampled = '30min'
    else:
        resampled = None

    # Make diagnostic plots
    fname_diagnostic = plots_diagnostic(ncdata, dest_dir, level0, time_range, bin_range,
                                        resampled)
    # Limit data if limits are not input by user
    time_lim, bin_depths_lim, ns_lim, ew_lim = limit_data(ncdata, ncdata.LCEWAP01.data,
                                                          ncdata.LCNSAP01.data, time_range,
                                                          bin_range)

    # Plot pressure PRESPR01 vs time
    fname_pres = plot_adcp_pressure(ncdata, dest_dir, resampled)

    # North/East velocity plots
    fname_ne = make_pcolor_ne(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim,
                              level0, 'raw', colourmap_lim, resampled)

    # Along/Cross-shelf velocity plots
    fname_ac = make_pcolor_ac(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim,
                              'raw', along_angle, colourmap_lim, resampled)

    # Redo whole process with tidal-filtered data

    if filter_type == "Godin":
        ew_filt, ns_filt = filter_godin(ncdata)
    elif filter_type.endswith("h"):
        ew_filt, ns_filt = filter_XXh(ncdata, num_hrs=int(filter_type[:-1]))
    else:
        ValueError("filter_type value not understood !")

    # Limit data
    time_lim, bin_depths_lim, ns_filt_lim, ew_filt_lim = limit_data(ncdata, ew_filt, ns_filt,
                                                                    time_range, bin_range)

    fname_ne_filt = make_pcolor_ne(ncdata, dest_dir, time_lim, bin_depths_lim, ns_filt_lim,
                                   ew_filt_lim, level0, filter_type, colourmap_lim, resampled)

    # Along-shore/cross-shore
    fname_ac_filt = make_pcolor_ac(ncdata, dest_dir, time_lim, bin_depths_lim, ns_filt_lim,
                                   ew_filt_lim, filter_type, along_angle, colourmap_lim,
                                   resampled)

    fname_binplot = binplot_compare_filt(ncdata, dest_dir, time_lim, ew_lim, ew_filt_lim,
                                         filter_type, direction, resampled)

    # Close netCDF file
    ncdata.close()

    # Assemble all file names of the plots produced
    fout_name_list += [fname_diagnostic, fname_pres, fname_ne, fname_ac, fname_ne_filt,
                       fname_ac_filt, fname_binplot]

    return fout_name_list


def test():
    f = ('C:\\Users\\HourstonH\\Documents\\adcp_processing\\moored\\2022-069_recoveries\\'
         'ncdata\\newnc\\e01_20210602_20220715_0097m.adcp.L1.nc')
    # f = ('C:\\Users\\HourstonH\\Documents\\adcp_processing\\moored\\2022-069_recoveries\\'
    #      'ncdata\\newnc\\scott3_20210603_20220718_0230m.adcp.L1.nc')
    # f = ('C:\\Users\\HourstonH\\Documents\\adcp_processing\\moored\\2022-069_recoveries\\'
    #      'ncdata\\newnc\\hak1_20210703_20220430_0042m.adcp.L1.nc')

    ds = xr.open_dataset(f)

    dest_dir = 'C:\\Users\\HourstonH\\Documents\\adcp_processing\\plan_for_update\\'

    if ds.orientation == 'up':
        bin_depths = ds.instrument_depth - ds.distance.data
    else:
        bin_depths = ds.instrument_depth + ds.distance.data

    make_pcolor_speed(dest_dir, station=ds.station, deployment_number=ds.deployment_number,
                      instrument_depth=int(np.round(ds.instrument_depth)),
                      time_lim=ds.time.data, bin_depths_lim=bin_depths[:-2], ns_lim=ds.LCNSAP01.data[:-2, :],
                      ew_lim=ds.LCEWAP01.data[:-2, :])

    quiver_plot(dest_dir, station=ds.station, deployment_number=ds.deployment_number,
                instrument_depth=int(np.round(ds.instrument_depth)),
                time_lim=ds.time.data, bin_depths_lim=bin_depths[:-2], ns_lim=ds.LCNSAP01.data[:-2, :],
                ew_lim=ds.LCEWAP01.data[:-2, :])
    return
