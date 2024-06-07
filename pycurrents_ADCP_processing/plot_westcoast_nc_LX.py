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
from scipy import signal as ssignal
from scipy import stats as sstats
import bisect
from scipy.optimize import minimize_scalar
import matplotlib.ticker as plticker
import ttide
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
import matplotlib
# from pycurrents_ADCP_processing.utils import parse_processing_history
import warnings
from pycurrents_ADCP_processing.utils import round_to_int, vb_flag, calculate_depths


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
        plot_dir = './{}/L0_plots/'.format(dest_dir)
    elif 'L1' in filename.data.tolist():
        plot_dir = './{}/L1_plots/'.format(dest_dir)
    elif 'L2' in filename.data.tolist():
        plot_dir = './{}/L2_plots/'.format(dest_dir)
    else:
        ValueError('Input netCDF file must be a L0, L1 or L2-processed file.')
        return None

    return plot_dir


def review_plot_naming(plot_title, png_name, serial_number, is_pre_split=False, resampled=None):
    """Final plot title and png name amendments, which are done for most plots"""

    if serial_number == 'Unknown':
        if plot_title is not None:
            plot_title = plot_title.replace('Unknown ', '')
        png_name = png_name.replace('Unknown_', '')

    if is_pre_split:
        if plot_title is not None:
            plot_title += ', pre-split'
        png_name = png_name.replace('.png', '_pre-split.png')

    if resampled is not None:
        if plot_title is not None:
            plot_title += f', {resampled} subsampled'
        png_name = png_name.replace('.png', f'_{resampled}_subsamp.png')

    return plot_title, png_name


def plot_adcp_pressure(nc: xr.Dataset, dest_dir: str, resampled=None, is_pre_split=False):
    """Plot pressure, PRESPR01, vs time
    :param nc: xarray dataset object from xarray.open_dataset(ncpath)
    :param dest_dir: name of subfolder to which plot will be saved
    :param resampled: "30min" if resampled to 30 minutes; None if not resampled
    :param is_pre_split: True if plot is being made before splitting the dataset into multiple
        segments in the case of a mooring strike in L1, False otherwise
    """

    fig = plt.figure(dpi=400)

    plt.plot(nc.time.data, nc.PRESPR01.data, linewidth=.8)

    plt.gca().invert_yaxis()

    plt.ylabel("Pressure (dbar)")

    # Create plots subfolder
    plot_dir = get_plot_dir(nc.filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_title = "{}-{} {} PRESPR01".format(nc.station, nc.deployment_number,
                                            nc.instrument_serial_number.data)

    png_name = plot_dir + "{}-{}_{}_{}m_PRESPR01.png".format(
        nc.station, nc.deployment_number, nc.instrument_serial_number.data,
        round_to_int(nc.instrument_depth.data))

    plot_title, png_name = review_plot_naming(
        plot_title, png_name, nc.instrument_serial_number.data, is_pre_split, resampled
    )

    plt.title(plot_title)

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

    if nc.instrument_subtype == 'Sentinel V' and flag_vb == 0:
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
    # Modify legend location in backscatter and velocity subplots to not overlap with places of greater variation
    legend_loc = 'lower left' if nc.orientation == 'up' else 'upper left'

    beam_no = 1
    for dat, col in zip(amp, colours):
        f1 = ax.plot(dat, depths, label='Beam {}'.format(beam_no), linewidth=1, marker='o',
                     markersize=2, color=col)
        beam_no += 1
    ax.set_ylim(depths[-1], depths[0])
    ax.legend(loc=legend_loc)
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

    if nc.instrument_subtype == 'Sentinel V' and flag_vb == 0:
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
    ax.legend(loc=legend_loc)
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
    suptitle = '{}-{} {} at {} m depth'.format(
        nc.station, nc.deployment_number, nc.instrument_serial_number.data, str(np.round(nc.instrument_depth.data, 1))
    )
    fig_name = plot_dir + '{}-{}_{}_{}m_diagnostic.png'.format(
        nc.station, str(nc.deployment_number), nc.instrument_serial_number.data,
        round_to_int(nc.instrument_depth.data)
    )

    suptitle, fig_name = review_plot_naming(
        suptitle, fig_name, nc.instrument_serial_number.data, is_pre_split=False, resampled=resampled
    )

    fig.suptitle(suptitle, fontweight='semibold')

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
        bin_depths = ncdata.instrument_depth.data - ncdata.distance.data
    else:
        bin_depths = ncdata.instrument_depth.data + ncdata.distance.data
    # print(bin_depths)

    # REVISION Jan 2024: bad leading and trailing ensembles are deleted from dataset, so don't need this step
    # data.time should be limited to the data.time with no NA values; bins must be limited
    if time_range is None:
        # if 'L1' in ncdata.filename.data.tolist() or 'L2' in ncdata.filename.data.tolist():
        #     leading_ens_cut, trailing_ens_cut = parse_processing_history(
        #         ncdata.attrs['processing_history']
        #     )
        #     time_first_last = (leading_ens_cut, len(ew_data[0]) - trailing_ens_cut)
        time_first_last = (0, len(ew_data[0]))
    else:
        time_first_last = time_range

    # Remove bins where surface backscatter occurs
    time_lim = ncdata.time.data[time_first_last[0]:time_first_last[1]]

    if bin_range is None:
        bin_first_last = (np.where(bin_depths >= 0)[0][0], np.where(bin_depths >= 0)[0][-1])
        bin_depths_lim = bin_depths[bin_depths >= 0]

        ew_lim = ew_data[bin_depths >= 0, time_first_last[0]:time_first_last[1]]  # Limit velocity data
        ns_lim = ns_data[bin_depths >= 0, time_first_last[0]:time_first_last[1]]
    else:
        bin_first_last = (bin_range[0], bin_range[1])
        bin_depths_lim = bin_depths[bin_range[0]: bin_range[1]]

        ew_lim = ew_data[bin_range[0]: bin_range[1], time_first_last[0]:time_first_last[1]]
        ns_lim = ns_data[bin_range[0]: bin_range[1], time_first_last[0]:time_first_last[1]]

    return time_lim, bin_depths_lim, ns_lim, ew_lim, time_first_last, bin_first_last


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

    instrument_depth = np.round(float(nc.instrument_depth.data), 1)

    fig = plt.figure(figsize=(13.75, 10))
    ax = fig.add_subplot(2, 1, 1)

    f1 = ax.pcolormesh(time_lim, bin_depths_lim, ns_lim[:, :], cmap='RdBu_r', vmin=vminvmax[0],
                       vmax=vminvmax[1], shading='auto')
    cbar = fig.colorbar(f1, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)
    ax.set_ylabel('Depth [m]', fontsize=14)

    if 'h' in filter_type:  # xxh-average; e.g. '30h', '35h'
        filter_type_title = '{} average'.format(filter_type)
    elif filter_type == 'Godin':
        filter_type_title = 'Godin filtered'
    elif filter_type == 'raw':
        filter_type_title = filter_type
    else:
        ValueError('Not a recognized data type; choose one of \'raw\', \'30h\' or \'Godin\'')

    ax.set_title(
        'ADCP ({}North, {}) {}-{} {}m{}'.format(
            magnetic, filter_type_title, nc.attrs['station'], nc.attrs['deployment_number'],
            instrument_depth, resampled_str
        ), fontsize=14
    )

    ax.invert_yaxis()

    ax2 = fig.add_subplot(2, 1, 2)

    f2 = ax2.pcolormesh(time_lim, bin_depths_lim, ew_lim[:, :], cmap='RdBu_r', vmin=vminvmax[0],
                        vmax=vminvmax[1], shading='auto')
    cbar = fig.colorbar(f2, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)

    ax2.set_ylabel('Depth [m]', fontsize=14)

    ax2.set_title(
        'ADCP ({}East, {}) {}-{} {}m{}'.format(
            magnetic, filter_type_title, nc.attrs['station'], nc.attrs['deployment_number'],
            instrument_depth, resampled_str
        ),
        fontsize=14
    )

    ax2.invert_yaxis()

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    plot_dir = get_plot_dir(nc.filename, dest_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if level0:
        vel_type = 'magn_NE'
    else:
        vel_type = 'NE'

    # Have to round instrument depth twice due to behaviour of the float
    plot_name = plot_dir + '{}-{}_{}_{}m_{}_{}{}.png'.format(
        nc.attrs['station'], nc.attrs['deployment_number'], nc.instrument_serial_number.data,
        round_to_int(instrument_depth), vel_type, filter_type, resampled_4fname
    )
    fig.savefig(plot_name)
    plt.close()

    return os.path.abspath(plot_name)


def make_pcolor_speed(dest_dir: str, station: str, deployment_number: str, instrument_depth: int,
                      time_lim: np.ndarray, bin_depths_lim: np.ndarray,
                      ns_lim: np.ndarray, ew_lim: np.ndarray, resampled=None, colourmap_lim: tuple = None):
    """ Skip this plot for now in favour of quiver plots
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

    plot_name = f'{station}-{deployment_number}_{round_to_int(instrument_depth)}m_spd_dir.png'
    if resampled:
        plot_name = plot_name.replace('.png', f'_{resampled}_resampled.png')

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


def make_pcolor_ac(nc: xr.Dataset, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim,
                   filter_type='raw', along_angle=None, colourmap_lim=None, resampled=None):
    """
    Function for plotting north and east velocities from ADCP data.
    :param nc: ADCP dataset from a netCDF file read in using the xarray package
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

    instrument_depth = np.round(float(nc.instrument_depth.data), 1)

    u_along, u_cross = resolve_to_alongcross(ew_lim, ns_lim, along_angle)
    AS = u_along
    CS = u_cross

    if colourmap_lim is None:
        vminvmax = get_vminvmax(AS, CS)
    else:
        vminvmax = colourmap_lim

    fig = plt.figure(figsize=(13.75, 10))
    ax1 = fig.add_subplot(2, 1, 1)  # along-shore subplot

    f1 = ax1.pcolormesh(time_lim, bin_depths_lim, AS[:, :], cmap='RdBu_r', vmin=vminvmax[0],
                        vmax=vminvmax[1], shading='auto')
    cbar = fig.colorbar(f1, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)

    ax1.set_ylabel('Depth [m]', fontsize=14)

    if 'h' in filter_type:  # xxh-average; e.g. '30h', '35h'
        filter_type_title = '{} average'.format(filter_type)
    elif filter_type == 'Godin':
        filter_type_title = 'Godin filtered'
    elif filter_type == 'raw':
        filter_type_title = filter_type
    else:
        ValueError('Not a recognized data type; choose one of \'raw\', \'30h\' or \'Godin\'')

    ax1.set_title(
        'ADCP (along, {}) {}$^\circ$ (CCW from E) {}-{} {}m{}'.format(
            filter_type_title, along_angle, nc.attrs['station'], nc.attrs['deployment_number'],
            instrument_depth, resampled_str
        ),
        fontsize=14
    )

    ax1.invert_yaxis()

    ax2 = fig.add_subplot(2, 1, 2)  # cross-shore subplot

    f2 = ax2.pcolormesh(time_lim, bin_depths_lim, CS[:, :], cmap='RdBu_r', vmin=vminvmax[0],
                        vmax=vminvmax[1], shading='auto')
    cbar = fig.colorbar(f2, shrink=0.8)
    cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)

    ax2.set_ylabel('Depth [m]', fontsize=14)

    ax2.set_title(
        'ADCP (cross, {}) {}$^\circ$ (CCW from E) {}-{} {}m{}'.format(
            filter_type_title, str(cross_angle), nc.attrs['station'], nc.attrs['deployment_number'],
            instrument_depth, resampled_str
        ),
        fontsize=14
    )

    ax2.invert_yaxis()

    # Create plots subfolder if not made already
    plot_dir = get_plot_dir(nc.filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # have to round instrument depth twice due to behaviour of float
    plot_name = '{}-{}_{}_{}m_AC_{}{}.png'.format(
        nc.attrs['station'], nc.attrs['deployment_number'], nc.instrument_serial_number.data,
        round_to_int(instrument_depth), filter_type, resampled_4fname
    )
    fig.savefig(plot_dir + plot_name)
    plt.close()

    return os.path.abspath(plot_dir + plot_name)


def num_ens_per_hr(time_data: np.ndarray):
    """
    Calculate the number of ensembles recorded per hour
    :param time_data: ncdata.time.data
    """
    time_incr = float(time_data[1] - time_data[0])
    hr2min = 60
    min2sec = 60
    sec2nsec = 1e9
    hr2nsec = hr2min * min2sec * sec2nsec
    return round_to_int(hr2nsec / time_incr)


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
    ens_per_hr = num_ens_per_hr(nc.time.data)
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

    # Replace any nans with the mean value to avoid nans propagating
    ew_df.fillna(value=np.nanmean(ew_df), inplace=True)

    ns_df.fillna(value=np.nanmean(ns_df), inplace=True)

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
    ens_per_hr = num_ens_per_hr(nc.time.data)
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

    # Replace any nans with the mean value to avoid nans propagating
    ew_df.fillna(value=np.nanmean(ew_df), inplace=True)

    ns_df.fillna(value=np.nanmean(ns_df), inplace=True)

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
    ens_per_hr = num_ens_per_hr(nc.time.data)
    hr2day = 24
    day2mth = 30
    ens_per_mth = ens_per_hr * hr2day * day2mth

    bin_index = 0  # which bin to plot
    if nc.orientation == 'up':
        bin_depth = nc.instrument_depth.data - (bin_index + 1) * nc.cell_size
    else:
        bin_depth = nc.instrument_depth.data + (bin_index + 1) * nc.cell_size

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
        plot_name = nc.attrs['station'] + '-' + nc.attrs['deployment_number'] + '_{}m'.format(
            round_to_int(nc.instrument_depth.data)) + '-{}_bin{}_compare_{}.png'.format(
            vel_code, bin_index + 1, filter_type)
    else:
        ax1.set_title('ADCP {}-{} {} bin {} at {}m, {} subsampled'.format(
            nc.attrs['station'], nc.attrs['deployment_number'], vel_code, bin_index + 1, bin_depth,
            resampled), fontsize=14)
        plot_name = nc.attrs['station'] + '-' + nc.attrs['deployment_number'] + '_{}m'.format(
            round_to_int(nc.instrument_depth.data)) + '-{}_bin{}_compare_{}_{}_subsamp.png'.format(
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


def quiver_plot(dest_dir: str, data_filename,
                station: str, deployment_number: str, instrument_depth: float, serial_number: str,
                time_lim: np.ndarray, bin_depths_lim: np.ndarray,
                ns_lim: np.ndarray, ew_lim: np.ndarray, single_bin_inds: list, resampled=None):
    """
    Make subplots of speed and direction for unfiltered ADCP data
    :param dest_dir: name of directory for containing output files
    :param data_filename: object returned from xr.Dataset.filename
    :param station: station name
    :param deployment_number: deployment number for mooring
    :param instrument_depth: instrument depth
    :param serial_number: instrument serial number
    :param time_lim: cleaned time data; array type
    :param bin_depths_lim: cleaned bin depth data; array type
    :param ns_lim: cleaned north-south velocity data; array type
    :param ew_lim: cleaned east-west velocity data; array type
    :param single_bin_inds: indices of bins to plot [shallow, middle, deep]
    :param resampled: "30min" if resampled to 30 minutes; None if not resampled
    :return: Absolute file path of the figure(s) this function creates
    """

    instrument_depth = round_to_int(instrument_depth)

    # Apply the bin indices
    ew_lim = ew_lim[single_bin_inds, :]
    ns_lim = ns_lim[single_bin_inds, :]
    bin_depths_lim = bin_depths_lim[single_bin_inds]

    speed = np.repeat(np.nan, len(ns_lim.flatten())).reshape(ns_lim.shape)
    # direction = np.repeat(np.nan, len(ns_lim.flatten())).reshape(ns_lim.shape)

    for i in range(len(ns_lim[:, 0])):
        for j in range(len(ns_lim[0, :])):
            speed[i, j] = (ns_lim[i, j] ** 2 + ew_lim[i, j] ** 2) ** .5
            # direction[i, j] = np.rad2deg(np.arctan(ns_lim[i, j] / ew_lim[i, j]))  # degrees CCW from East

    max_abs_speed = np.nanmax(abs(speed))  # Use for setting y-axis limits

    # Axis for placing arrows
    zeros = np.zeros(time_lim.size)

    # # Invert the depth bins to go shallowest to deepest
    # bin_depths_lim = bin_depths_lim[::-1]
    # ew_lim = ew_lim[::-1, :]
    # ns_lim = ns_lim[::-1, :]
    # speed = speed[::-1, :]
    # # direction = direction[::-1, :]

    # Plots will have 6 rows and 2 columns of subplots for a total of 12 subplots
    num_bins = len(bin_depths_lim)
    max_bins_per_col = 6
    max_width = 13
    max_height = 10.35  # 30
    if num_bins > max_bins_per_col:
        max_bins_per_plot = 12
        bins_per_col = max_bins_per_col
        num_figures = int(np.ceil(num_bins / max_bins_per_plot))
        num_cols = 2  # Number of columns in the figure
        figsize = (max_width, max_height)
    else:
        # One column instead of two, and a shorter column if num_buns < 6
        max_bins_per_plot = num_bins
        bins_per_col = num_bins
        num_figures = 1
        num_cols = 1
        # Adjust the height by a factor of how many bins
        # figsize = (max_width/1.75, max_height * (bins_per_col / max_bins_per_col))
        figsize = (max_width / 1.75, np.linspace(2.2, max_height, max_bins_per_col, endpoint=True)[num_bins - 1])

    # print(figsize)
    plot_list = []

    for k in range(num_figures):
        fig = plt.figure(figsize=figsize)  # Setting figsize here doesn't work
        # print(fig.get_size_inches())
        fig.subplots_adjust(hspace=0.75, wspace=0.35)

        for i in range(1, max_bins_per_plot + 1):
            idx = max_bins_per_plot * k + i - 1
            if idx < len(bin_depths_lim):
                ax = fig.add_subplot(bins_per_col, num_cols, i)  # bins_per_col
                ax.fill_between(time_lim, speed[idx, :], zeros, color='grey', alpha=0.1)
                ax.quiver(time_lim, zeros, ew_lim[idx, :], ns_lim[idx, :], color='blue',
                          width=0.004, units='y', scale_units='y', scale=1)
                # Scale because the arrows aren't usually perpendicular
                ax.set_ylim(-max_abs_speed/1.75, max_abs_speed/1.75)
                ax.set_title('Bin depth: ' + str(np.around(bin_depths_lim[idx], 1)) + 'm', fontsize=8,
                             fontweight='semibold')
                ax.set_ylabel('Speed (m/s)', size=8)
                ax.yaxis.labelpad = 0
                # ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6], minor=False)
                ax.tick_params(axis='both', direction='in', top=True, right=True, labelsize=7)
                # p = ax.add_patch(plt.Rectangle((1, 1), 1, 1, fc='k', alpha=0.1))
                # leg = ax.legend([p], ["Current magnitude (m/s)"], loc='lower right', fontsize=8)
                # leg._drawFrame = False

        # print(fig.get_size_inches())  # What is the real figsize??
        fig.set_size_inches(*figsize)

        plt.suptitle(f'{station}-{deployment_number} Current Speed (m/s)')

        plt.tight_layout()  # Make sure labels don't overlap

        plot_name = f'{station}-{deployment_number}_{serial_number}_{instrument_depth}m_quiver_plotCOUNTER.png'

        _, plot_name = review_plot_naming(
            plot_title=None, png_name=plot_name, serial_number=serial_number, is_pre_split=False, resampled=resampled
        )

        # Create L1_Python_plots or L2_Python_plots subfolder if not made already
        plot_dir = get_plot_dir(data_filename, dest_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot_name = os.path.join(plot_dir, plot_name)

        counter = 1
        while os.path.exists(plot_name.replace('COUNTER', str(counter))):
            counter += 1
            if counter > 100:  # Safeguard against runaway iterations
                break

        plot_name = plot_name.replace('COUNTER', str(counter))

        plt.savefig(plot_name)
        plt.close()
        plot_list.append(plot_name)

    return plot_list


def sampling_freq(time) -> float:
    """
    Return fs in units of CPD (cycles per day)
    """
    s_per_day = 3600 * 24
    dt_s = (time[1] - time[0]).astype('timedelta64[s]')  # seconds
    return np.round(s_per_day / dt_s.astype(np.float32))


# noinspection GrazieInspection
def rot(u, v=None, fs=1.0, nperseg=None, noverlap=None, detrend='constant', axis=-1, conf=0.95):
    """ Rotary and cross-spectra with coherence estimate.
    Function from https://gitlab.com/krassovski/pyap/-/raw/master/analysispkg/pkg_spectrum.py
    Parameters
    ----------
    u : array_like
        Sampled series, e.g. u-component of velocity.
    v : array_like, optional
        Second series or v-component of velocity.
    fs : float, optional
        Sampling frequency of the series. Defaults to 1.0.
    nperseg : int, optional
        Length of spectral window. Defaults to the power of 2 nearest to the
        quarter of the series length.
    noverlap : int, optional
        Number of points to overlap between windows. If `None`,
        ``noverlap = nperseg / 2``. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    axis : int, optional
        Axis along which the periodogram is computed; the default is
        over the last axis (i.e. ``axis=-1``).
    conf : float, optional
        Confidence limits for power spectra and confidence level for coherence to calculate,
        e.g. conf=0.95 gives 95% confidence limits. Defaults to 0.95.

    Returns
    -------
    Dictionary with fields:
        dof     number of degrees of freedom
        f       frequency vector
        period  corresponding period
        pxx     x-component spectrum (u)
        pyy     y-component spectrum (v)
        cxy     co-spectrum
        qxy     quad-spectrum
        hxy     admittance (transfer,response) function
        hhxy    ratio of sx and sy
        r2xy    coherence squared
        phase   phase
        pneg    clockwise component of rotary spectra
        ppos    counter-clockwise component of rotary spectra
        ptot    total spectra
        orient  orientation of major axis of the ellipse
        rotcoeff    rotary coefficient
        stabil  stability of the ellipse
        conf    upper and lower limits of the confidence interval relative to unity
        cohconf confidence level for coherence squared

    Adapted from a Fortran routine ROTCOL written by A.Rabinovich.
    Reference: Gonella, 1972, DSR, Vol.19, 833-846
    """
    n = u.shape[axis]  # series length

    if nperseg is None:
        nperseg = int(2 ** np.round(np.log2(n / 4)))
    if noverlap is None:
        noverlap = np.floor(nperseg / 2)

    k = np.fix(n / nperseg) + np.fix((n - nperseg / 2) / nperseg)

    # power per unit frequency
    f, sx = ssignal.welch(u, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend, axis=axis)

    sx = np.real(sx)  # needed in case there are NaNs in u (sx is NaN+i*NaN)
    if v is not None:
        _, sy = ssignal.welch(v, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend, axis=axis)
        sy = np.real(sy)  # needed in case there are NaNs in v  # TODO Check

        # cross-spectra
        # power per unit frequency
        _, pxy = ssignal.csd(u, v, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend, axis=axis)

        cxy = np.real(pxy)  # co-spectrum
        qxy = -np.imag(pxy)  # quadrature spectrum
        hxy = abs(pxy) / sx  # admittance function
        hhxy = sy / sx  # component spectra ratio

        r2xy = np.abs(pxy) ** 2 / (sx * sy)  # coherence squared
        phase = np.degrees(-np.angle(pxy))  # phase

        # rotary spectra
        sm = (sx + sy - 2 * qxy) / 2
        sp = (sx + sy + 2 * qxy) / 2
        st = sm + sp

        # major axis orientation
        orient = np.degrees(np.arctan2(cxy * 2, sx - sy) / 2)

        # rotary coefficient
        rotcoeff = (sm - sp) / st

        # stability of the ellipse
        stabil = np.sqrt((st ** 2 - (sx * sy - cxy ** 2) * 4) / (st ** 2 - 4 * qxy ** 2))

        # confidence level for coherence
        # dof = k * 2
        # confidence = np.array([.99, .95, .90, .80])
        # cohconf = 1 - (1 - confidence) ** (2 / dof)
        cohconf = 1 - (1 - conf) ** (1 / k)
    else:
        sy = cxy = qxy = hxy = hhxy = r2xy = phase = sm = sp = st = orient = rotcoeff = stabil = cohconf = None

    # confidence intervals
    vw = 1.5  # coefficient for hann window
    # k = np.floor((n - noverlap) / (nperseg - noverlap))  # number of windows
    dof = 2 * np.round(k * 9 / 11 * vw)  # degrees of freedom
    # c = dof / np.array(scipy.stats.chi2.interval(conf, df=dof))

    # TSA way (takes 50% overlap into account and applies window coefficient):
    # conf_lims = chi2conf(confidence, dof);    % limits relative to 1
    conf_lims = dof / np.array(sstats.chi2.interval(conf, df=dof))
    # conf_lims = np.array(scipy.stats.chi2.interval(conf, df=dof))
    # confu = conf_lims[0]
    # confl = conf_lims[1]
    # confLims = sm * confLims1  # limits for each spectral value

    with np.errstate(invalid='ignore', divide='ignore'):
        period = 1 / f
    # pack all results in an output dict
    r = dict(dof=dof, f=f, period=period, pxx=sx, pyy=sy, cxy=cxy, qxy=qxy, hxy=hxy, hhxy=hhxy, r2xy=r2xy, phase=phase,
             pneg=sm, ppos=sp, ptot=st, orient=orient, rotcoeff=rotcoeff, stabil=stabil,
             conf=conf_lims, cohconf=cohconf)
    return r


def plot_spectrum(f, p, c=None, clabel=None,
                  cx=None, cy=None,
                  color=None, ccolor='k',
                  units='m', funits='cpd',
                  ax=None, **options):
    """ Spectrum plot in log-log axes with confidence interval
    Function from https://gitlab.com/krassovski/pyap/-/raw/master/analysispkg/pkg_spectrum.py
    Parameters
    ----------
    f : array_like
        Array of sample frequencies.
    p : array_like
        Power spectral density or power spectrum of x.
    c : array of length 2, optional
        Upper and lower confidence limit relative to 1. Returned by spectrum()
        in `c`. No error bar by default.
    clabel : str, optional
        Label for the error bar, e.g. '95%'. No label by default.
    cx,cy : float, optional
        Coordinates (in data units) for the error bar. By default, cx is at
        0.8 of x-axis span, and cy is at 3 lower confidence intervals above max
        power value in the band spanning the error bar with label.
    color : matplotlib compatible color, optional
        Color cpecification for the plot line. Defaults to Matplotlib default
        color.
    ccolor : matplotlib compatible color or 'same', optional
        Color specification for the error bar. If 'same', uses same color as
        the plot line. Defaults to black, 'k'.
    units : str, optional
        Units for analysed series, e.g. 'm' for sea level, 'm/s' for velocity.
        Defaults to 'm'.
    funits : str, optional
        Frequency units. Defaults to 'cpd'.
    ax : Matplotlib axes handle, optional
        Defaults to the current axes.
    **options : kwargs
        Extra options for matplotlib's loglog()
    """

    # exclude zero-frequency value
    ival = f != 0
    f = f[ival]
    p = p[ival]

    if ax is None:
        ax = plt.gca()

    # spectrum
    hp, = ax.loglog(f, p, color=color, **options)
    ax.set_xlabel('Frequency (' + funits + ')')
    ax.set_ylabel(r'PSD (' + units + ')$^2$/' + funits)
    #    ax.grid(True,which='both')
    ax.grid(True, which='major', linewidth=1)
    ax.grid(True, which='minor', linewidth=.5)

    # error bar
    if c is not None:
        # attempt of automatic error bar placement
        if cx is None or cy is None:
            xlim = ax.get_xlim()
            xliml = np.log10(xlim)
            dxliml = np.abs(np.diff(xliml))
            if cx is None:
                cx = 10 ** (np.min(xliml) + dxliml * .8)  # at 0.8 of x-axis span
            if cy is None:
                # approx x-range for error bar
                xrng = 10 ** (np.log10(cx) + dxliml * [-0.05, 0.15])
                # find p within the error bar horizontal span, extrapolate if necessary
                inx = np.logical_and(f > xrng[0], f < xrng[1])
                if np.any(inx):
                    pxr = p[inx]
                else:
                    pxr = np.interp(xrng, f, p)
                # 3 lower intervals above max power value in the band
                cy = pxr.max() * (1 / c[1]) ** 3
        if ccolor == 'same':
            ccolor = hp[0].get_color()
        # Need an error bar that goes from cy*c[1] to cy*c[0];
        # plt.errorbar plots bar from cy-clower to cy+cupper;
        # convert c to yerr accordingly:
        yerr = cy * ((c[::-1] - 1) * [-1, 1])[:, None]
        ax.errorbar(cx, cy, yerr=yerr, color=ccolor, capsize=5, marker='o', markerfacecolor='none')
        # capsize is the width of horizontal ticks at upper and lower bounds in points
        if clabel is not None:
            ax.text(cx, cy, '   ' + clabel, verticalalignment='center', color=ccolor)
    return hp


def mark_frequency(fmark, fname='', ax=None, f=None, p=None, fig=None):
    """ Annotation arrow with text label above spectra plot at a specified frequency
    Author: Maxim Krassovski
    https://gitlab.com/krassovski/pyap/-/blob/master/analysispkg/pkg_spectrum.py?ref_type=heads

    Parameters
    ----------

    fmark : float
        Frequency to mark
    fname : str, optional
        Text for annotation. Defaults to empty string.
    ax : axes handle, optional
        Use all line objects in these axes to determine vertical position of
        the annotation. If supplied, f,p are ignored. Also denotes axes for
        plotting. Defaults to None.
    f,p : array_like, optional
        x- and y-coordinates for the curve to use for vertical positioning of
        the annotation. Defaults to None.
    fig : matplotlib figure or None
        Figure instance for which single bin rotary spectra have been plotted
    """
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    # determine y-position from all lines in the axes
    if f is not None and p is not None:  # determine y-position from supplied f,p
        pmark = np.interp(fmark, f, p)
    else:
        lines = ax.lines
        # find highest line at this freq
        pmark = np.nan
        for ln in lines:
            pmark = np.fmax(pmark,
                            np.interp(fmark, ln.get_xdata(), ln.get_ydata(),
                                      left=np.nan, right=np.nan))
        if np.isnan(pmark):  # no lines at this frequency
            # place it in the middle
            ylim = ax.get_ylim()
            if ax.get_yscale() == 'log':
                pmark = 10 ** (np.mean(np.log10(ylim)))
            else:
                pmark = np.mean(ylim)

        # raise RuntimeError('Either axes or f,p values are needed to determine y-position for frequency mark.')

    arrowprops = dict(arrowstyle="simple, head_width=0.25, tail_width=0.05",
                      color='k', lw=0.5, shrinkB=10)
    # shrinkB is the offset in pixels from (f,p) line

    ann = ax.annotate(fname, xy=(fmark, pmark), xytext=(0, 25),
                      textcoords='offset points', horizontalalignment='center',
                      arrowprops=arrowprops)

    # matplotlib.use('qt4agg')
    fig.canvas.draw()
    box = matplotlib.text.Text.get_window_extent(ann)
    ylim = ax.get_ylim()
    yann = ax.transData.inverted().transform(box)[:, 1]
    # from IPython import embed;    embed()  ######################
    if ylim[1] < yann[1]:
        ax.set_ylim([ylim[0], yann[1] * yann[1] / yann[0]])
        # ax.grid()
    return


def plot_rot(r: dict, clabel=None, cx=None, cy=None, color=None, ccolor='k', units='m/s', funits='cpd',
             fig=None, ax_neg=None, ax_pos=None, **options):
    """ Plot rotary components spectra
    Function from https://gitlab.com/krassovski/pyap/-/raw/master/analysispkg/pkg_spectrum.py
    fig : matplotlib figure or subfigure
    """
    if fig is None and (ax_pos is None and ax_neg is None):
        fig = plt.figure(figsize=(8.5, 4.5))  # (width, height) so the subplots are more square
    if ax_pos is None and ax_neg is None:
        (ax_neg, ax_pos) = fig.subplots(1, 2, sharey=True)  # , subplot_kw={'aspect': 'equal'})
        ax_neg.invert_xaxis()
    if clabel is None:
        c = None
    else:
        c = r['conf']

    hneg = plot_spectrum(r['f'], r['pneg'], color=color, ccolor=ccolor,
                         units=units, funits=funits, ax=ax_neg, **options)
    hpos = plot_spectrum(r['f'], r['ppos'], c=c, clabel=clabel, cx=cx, cy=cy, color=color, ccolor=ccolor,
                         units=units, funits=funits, ax=ax_pos, **options)
    ax_pos.set(ylabel=None)
    # fig.tight_layout()
    return fig, ax_neg, ax_pos, hneg, hpos


def make_plot_rotary_spectra(dest_dir: str, data_filename, station: str, deployment_number: str,
                             instrument_depth: float, serial_number: str,
                             bin_number: int, bin_depths_lim: np.ndarray, time_lim: np.ndarray,
                             ns_lim: np.ndarray, ew_lim: np.ndarray, latitude: float,
                             resampled=None, axis=-1, do_tidal_annotation=True):
    """
    Make single-bin plot of rotary spectra with separate subplots for CW and CCW components

    from https://gitlab.com/krassovski/pyap/-/blob/master/analysispkg/pkg_spectrum.py?ref_type=heads
    Follow functions rot() and plot_rot()

    fs: sampling frequency in units of CPD

    Notes:
    NaNs need to be treated before applying this function. Maxim recommends to fill with mean or linearly interpolate
    if the tide is strong a better way would be to remove tide, fill gaps as above and add the tide back in
    long gaps will distort the spectra, so avoid showing spectra for very gappy series if possible, or specify the
    amount of missing data as a disclaimer

    inputs
    - bin_number: bin index, starts at zero, applies to subsetted data not the complete dataset at this point...

    """

    instrument_depth = round_to_int(instrument_depth)

    # Cycles per hour to cycles per day
    cph_to_cpd = 24

    fs = sampling_freq(time_lim)

    # Remove/replace nans from velocity components otherwise they propagate in ssignal.welch()
    u = ew_lim[bin_number, :]
    u[np.isnan(u)] = np.nanmean(u)
    v = ns_lim[bin_number, :]
    v[np.isnan(v)] = np.nanmean(v)

    bin_depth = bin_depths_lim[bin_number]

    r = rot(u=u, v=v, axis=axis, fs=fs)

    fig, ax_neg, ax_pos, hneg, hpos = plot_rot(r)

    # Add annotation of major frequencies M2 and K1 to ax_neg and ax_pos
    fnames = ['M2', 'K1']

    if do_tidal_annotation:
        # keep or filling of nans does not affect result frequencies
        try:
            result = run_ttide(
                U=u,
                V=v,
                xin_units='m/s',
                time_lim=time_lim,
                latitude=latitude,
                constitnames=fnames
            )
        except ValueError as e:
            print('t_tide failed with error', e)
            return

        # nameu might be in different order than fnames !!
        for fname_S4, fmark_cph in zip(result['nameu'], result['fu']):
            # Convert component name to string and remove any trailing whitespace
            fname = fname_S4.astype('str').strip(' ')
            # Convert frequency fmark from per hour to per day
            fmark_cpd = fmark_cph * cph_to_cpd
            mark_frequency(fmark=fmark_cpd, fname=fname, ax=ax_neg, f=r['f'], p=r['pneg'], fig=fig)
            mark_frequency(fmark=fmark_cpd, fname=fname, ax=ax_pos, f=r['f'], p=r['ppos'], fig=fig)

    # Add titles to the subplots
    ax_neg.set_title('CW')
    ax_pos.set_title('CCW')

    plt.suptitle(f'{station}-{deployment_number} Rotary Spectra - {np.round(bin_depth, 2)}m bin')

    # Save the figure
    plot_name = (f'{station}-{deployment_number}_{serial_number}_{instrument_depth}m'
                 f'_rotary_spectra_bin_{round_to_int(bin_depth)}m.png')

    if do_tidal_annotation:
        plot_name = plot_name.replace('.png', '_ttide.png')

    _, plot_name = review_plot_naming(
        plot_title=None, png_name=plot_name, serial_number=serial_number, is_pre_split=False, resampled=resampled
    )

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    plot_dir = get_plot_dir(data_filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_name = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_name)
    plt.close()

    return plot_name


def rot_freqinterp(rot_dict: dict) -> tuple:
    # Collect all frequency values from all spectra and make this the target frequency vector
    ftarget = np.array([])
    for depth in rot_dict.keys():
        fnew = rot_dict[depth]['f']
        ftarget = np.unique(np.concatenate((ftarget, fnew)))

    # Interpolate spectra values to "standard" frequencies

    # Initialize 2d arrays with shape (depth, num_frequencies)
    pneg_interp = np.zeros(len(rot_dict) * len(ftarget)).reshape((len(rot_dict), len(ftarget)))
    ppos_interp = np.zeros(len(rot_dict) * len(ftarget)).reshape((len(rot_dict), len(ftarget)))

    # Do 1d interpolation
    for i, depth in enumerate(rot_dict.keys()):
        # Default interpolation type is linear
        func_pneg = interp1d(x=rot_dict[depth]['f'], y=rot_dict[depth]['pneg'])
        func_ppos = interp1d(x=rot_dict[depth]['f'], y=rot_dict[depth]['ppos'])
        # Apply the returned functions
        pneg_interp[i, :] = func_pneg(ftarget)
        ppos_interp[i, :] = func_ppos(ftarget)

    return ftarget, pneg_interp, ppos_interp


def pcolor_rot_component(dest_dir: str, data_filename, station: str, deployment_number: str,
                         instrument_depth, serial_number: str,
                         x: np.ndarray, y, c: np.ndarray, clim: tuple, neg_or_pos: str,
                         funits='cpd', resampled=False):
    """
    x: depth
    y: standard frequencies; the product of interpolation
    c: real component of pneg_interp or ppos_interp
    clim: c limits in format (min, max)
    neg_or_pos: "neg" for negative (CW) component, "pos" for positive (CCW) component
    """
    # % CW: R, z, componentfield, separateaxes, units
    # [~,climcw,hcw] = pcolor_component(R,z,'sm',separateaxes,units); % ,'neg',[prefix '_cw']
    # % CCW
    # [~,climccw,hccw] = pcolor_component(R,z,'sp',separateaxes,units); % ,'pos',[prefix '_ccw']
    fig, ax = plt.subplots()
    f1 = ax.pcolormesh(x, y, c, cmap='jet', shading='auto',
                       norm=LogNorm(vmin=clim[0], vmax=clim[1]))  # Maxim's code uses the jet colormap

    # Make x axis log scale
    ax.set_xscale('log')

    # Invert y axis
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymax, ymin))

    # Add legend
    cbar = fig.colorbar(f1, ax=ax)
    cbar.set_label(r'PSD (m/s)$^2$/' + funits)

    ax.set_xlabel(f'Frequency ({funits})')
    ax.set_ylabel('Depth (m)')

    title = f'{station}-{deployment_number} Rotary Spectra (XX)'
    if neg_or_pos == 'neg':
        title = title.replace('XX', 'neg./CW')
    else:
        title = title.replace('XX', 'pos./CCW')
    ax.set_title(title)

    plt.tight_layout()

    # Save the figure
    plot_name = (f'{station}-{deployment_number}_{serial_number}_{round_to_int(instrument_depth)}m'
                 f'_depth_prof_rot_spec.png')
    if neg_or_pos == 'neg':
        plot_name = plot_name.replace('.png', '_neg_cw.png')
    else:
        plot_name = plot_name.replace('.png', '_pos_ccw.png')

    _, plot_name = review_plot_naming(
        plot_title=None, png_name=plot_name, serial_number=serial_number, is_pre_split=False, resampled=resampled
    )

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    plot_dir = get_plot_dir(data_filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_name = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_name)
    plt.close()
    return plot_name


def make_depth_prof_rot_spec(dest_dir: str, data_filename, station: str, deployment_number: str,
                             instrument_depth,
                             serial_number: str, bin_depths_lim: np.ndarray, ns_lim: np.ndarray,
                             ew_lim: np.ndarray, time_lim: np.ndarray, resampled=None):
    """
    Plot depth profiles of rotary spectra in a pseudo-color (pcolor) plot

    Adapted from Maxim Krassovski's MatLab code
    Calculate rotary spectra: https://gitlab.com/krassovski/Tools/-/blob/master/Signal/series/adcp_rot.m?ref_type=heads
    Plot them: https://gitlab.com/krassovski/Tools/-/blob/master/Signal/series/adcp_report.m?ref_type=heads
    -> rot_pcolor_plot()
    """
    instrument_depth = round_to_int(instrument_depth)

    fs = sampling_freq(time_lim)

    # Remove nans
    u = ew_lim
    u[np.isnan(u)] = np.nanmean(u)
    v = ns_lim
    v[np.isnan(v)] = np.nanmean(v)

    # Initialize dict to hold results for each bin
    rot_dict = {}

    # Iterate through all the bins
    for bin_idx in range(len(bin_depths_lim)):
        rot_dict[bin_depths_lim[bin_idx]] = rot(u=u[bin_idx, :], v=v[bin_idx, :], axis=0, fs=fs)

    # Standard frequencies
    ftarget, pneg_interp, ppos_interp = rot_freqinterp(rot_dict)

    # Exclude zero frequency, assuming zero frequency is first element in ftarget
    ftarget = ftarget[1:]
    pneg_interp = pneg_interp[:, 1:]
    ppos_interp = ppos_interp[:, 1:]

    # Get color range to unify both negative and positive plots
    cneg = np.real(pneg_interp)
    cpos = np.real(ppos_interp)
    cmin = np.min([np.min(cneg), np.min(cpos)])
    cmax = np.max([np.max(cneg), np.max(cpos)])

    if cmin >= cmax:
        clim = None
    else:
        clim = (cmin, cmax)

    # pcolor plot, skipping the 0 frequency
    pneg_plot_name = pcolor_rot_component(
        dest_dir,
        data_filename,
        station,
        deployment_number,
        instrument_depth,
        serial_number,
        x=ftarget,
        y=rot_dict.keys(),
        c=cneg,
        clim=clim,
        neg_or_pos='neg',
        resampled=resampled
    )

    ppos_plot_name = pcolor_rot_component(
        dest_dir,
        data_filename,
        station,
        deployment_number,
        instrument_depth,
        serial_number,
        x=ftarget,
        y=rot_dict.keys(),
        c=cpos,
        clim=clim,
        neg_or_pos='pos',
        resampled=resampled
    )

    # Spectra for selected bins/depths?

    return [pneg_plot_name, ppos_plot_name]


def ttide_constituents(time: np.ndarray, si: float, ray: float):
    """
    From https://gitlab.com/krassovski/pyap/-/blob/master/analysispkg/pkg_tides.py?ref_type=heads#L244
    Get a set of constituents which t_tide would use for a given series and Rayleigh number.

    constituents = ['MSM', 'MM', 'MSF', 'MF', 'ALP1', '2Q1', 'SIG1', 'Q1', 'RHO1', 'O1',
                    'TAU1', 'BET1', 'NO1', 'CHI1', 'PI1', 'P1', 'S1', 'K1', 'PSI1', 'PHI1',
                    'THE1', 'J1', 'SO1', 'OO1', 'UPS1', 'OQ2', 'EPS2', '2N2', 'MU2', 'N2',
                    'NU2', 'GAM2', 'H1', 'M2', 'H2', 'MKS2', 'LDA2', 'L2', 'T2', 'S2', 'R2',
                    'K2', 'MSN2', 'ETA2', 'MO3', 'M3', 'SO3', 'MK3', 'SK3', 'MN4', 'M4',
                    'SN4', 'MS4', 'MK4', 'S4', 'SK4', '2MK5', '2SK5', '2MN6', 'M6', '2MS6',
                    '2MK6', '2SM6', 'MSK6', '3MK7', 'M8']

    """
    from ttide import t_utils as tu
    from ttide import time as tm
    nobs = len(time)
    nobsu = nobs - np.remainder(nobs - 1, 2)
    stime = tm.date2num(time[0])
    centraltime = stime + np.floor(nobsu / 2) * si / 24.0
    nameu = tu.constituents(ray / (si * nobsu), np.array([]), np.array([]), np.array([]), np.array([]), centraltime)[0]
    return nameu


def plot_ellipse(ellipse, gap, clr):
    """Takes in a calculated ellipses and plots it, with a dot indicating a particle at the end of
    tracing out the ellipse.
    From
    https://gitlab.com/krassovski/pyap/-/blob/master/analysispkg/pkg_plot_currents_comparison.py?ref_type=heads
    """
    # Plot real and imaginary components of the ellipse
    plt.plot(np.real(ellipse[10 * gap:3600]), np.imag(ellipse[10 * gap:3600]), color=clr, linewidth=1)
    plt.scatter(np.real(ellipse[-1]), np.imag(ellipse[-1]), s=3, c=clr, marker='.')
    return


# this is a duplicate of the subroutine in pkg_plot_currents.py
def calculate_ellipse(major: float, minor: float, inc: float, phase: float):
    """
    Reads in the tidal parameters, returns a series of dots for plotting.
    (Easier than trying to describe the ellipse analytically)
    May need to be reworked for plotting ellipses on a map.
    From
    https://gitlab.com/krassovski/pyap/-/blob/master/analysispkg/pkg_plot_currents_comparison.py?ref_type=heads

    major, minor, inc, phase: parameters returned from ttide.t_tide() in the 'tidecon' key value
    """
    # major, minor, inc, phase = tidal_params
    # todo: add error checking here

    # Convert inclination and phase to radians
    inc = inc * np.pi / 180.0
    phase = phase * np.pi / 180.0

    # Eccentricity
    okmajor = np.nonzero(major)
    if len(okmajor[0]) > 0:
        ecc = minor / major
    else:
        ecc = np.nan
        major = np.nan

    # Ellipse
    nsegs = 3600
    t = np.linspace(0, 2.0 * np.pi, nsegs)
    ellipse = major * (np.cos(t[:] - phase) + 1j * ecc * np.sin(t[:] - phase)) * np.exp(1j * inc)

    return ellipse


def best_tick(span, mostticks):
    """
    Author : Maxim Krassovski
    """
    # https://stackoverflow.com/questions/361681/algorithm-for-nice-grid-line-intervals-on-a-graph
    minimum = span / mostticks
    magnitude = 10 ** math.floor(math.log(minimum, 10))
    residual = minimum / magnitude
    # this table must begin with 1 and end with 10
    table = [1, 2, 5, 10]  # options for gradations in each decimal interval
    tick = table[bisect.bisect_right(table, residual)] if residual < 10 else 10
    return tick * magnitude


def run_ttide(U: np.ndarray, V: np.ndarray, xin_units: str, time_lim: np.ndarray, latitude: float,
              constitnames: list):
    """
    Wrapper for ttide.t_tide()

    inputs:
    - U: east-west velocity component for a single bin, 1-D, in m/s
    - V: north-south velocity component for a single bin, 1-D, in m/s
    - xin_units: units to apply to the xin parameter for ttide.t_tide()
    - time_lim: time array
    - latitude: latitude of ADCP
    - constitnames: list of tidal constituent names for which to compute the frequency, amplitude, and phase
    """
    xin = U + 1j * V
    if xin_units == 'cm/s':
        xin *= 100

    # Round to 3 decimal places, units are hours
    sampling_interval = np.round(pd.Timedelta(time_lim[1] - time_lim[0]).seconds / 3600, 3)
    ray = 1
    synth = 0
    stime = pd.to_datetime(time_lim[0])

    result = ttide.t_tide(
        xin=xin,
        dt=sampling_interval,  # Sampling interval in hours, default = 1
        stime=stime,  # The start time of the series
        lat=latitude,
        constitnames=constitnames,  # ttide_constituents(datetime_lim, sampling_interval, ray),
        shallownames=[],
        synth=synth,  # The signal-to-noise ratio of constituents to use for the "predicted" tide
        ray=ray,  # Rayleigh criteria, default 1
        lsq='direct',  # avoid running as a long series
        out_style=None  # No printed output
    )
    return result


def find_constituent(con_names, constituent):
    """ Find the index of the requested constituent.
    Author: Maxim Krassovski"""

    # Check that the list of names is in unicode and not byte.
    tmp = con_names.astype('U4')
    ind = np.where(tmp == constituent)[0]
    if len(ind) == 0:
        return None
    else:
        return ind[0]


def parse_tidal_constituents(constituents, tide_result):
    """ Given the full tidal output from ttide, return the specified constituents.
    Author: Maxim Krassovski

    Parameters
    ----------
    constituents : list
        Constituents to parse
    tide_result : dict
        ttide output

    Returns
    -------
    dictionary of tuples,
    with the keys being the constituent names and the tuples being either (amp,phase) or (maj,min,inc,phase) if complex.
    """

    tidal_par, error_par = {}, {}

    for con in constituents:
        if 'nameu' in tide_result.keys():
            ind = find_constituent(tide_result['nameu'], con.ljust(4))
        else:
            ind = None  # case where tidal analysis was not performed
        # todo: option for obs but not mod and vice versa?
        if ind is not None:
            # Quantity, error
            a, ea = tide_result['tidecon'][ind, 0], tide_result['tidecon'][ind, 1]
            b, eb = tide_result['tidecon'][ind, 2], tide_result['tidecon'][ind, 3]
            if tide_result['tidecon'].shape[1] == 8:
                # Has complex components
                c, ec = tide_result['tidecon'][ind, 4], tide_result['tidecon'][ind, 5]
                d, ed = tide_result['tidecon'][ind, 6], tide_result['tidecon'][ind, 7]
            else:
                c, ec = np.nan, np.nan
                d, ed = np.nan, np.nan

        else:
            a, ea = np.nan, np.nan
            b, eb = np.nan, np.nan
            c, ec = np.nan, np.nan
            d, ed = np.nan, np.nan

        tidal_par[con] = (float(a), float(b), float(c), float(d))
        error_par[con] = (float(ea), float(eb), float(ec), float(ed))
    return tidal_par, error_par


def make_plot_tidal_ellipses(dest_dir: str, data_filename, station: str, deployment_number: str,
                             serial_number: str, instrument_depth, latitude: float,
                             time_lim: np.ndarray, bin_depth_lim: np.ndarray, ns_lim: np.ndarray,
                             ew_lim: np.ndarray, resampled=None):
    """
    Follow usage in
    https://gitlab.com/krassovski/pyap/-/blob/master/analysispkg/pkg_tides.py?ref_type=heads#L148

    Plot vertical distribution of tidal ellipses

    From ttide documentation:
    Although missing data can be handled with NaN, it is wise not to
    have too many of them. If your time series has a lot of missing
    data at the beginning and/or end, then truncate the input time
    series.
    """

    def plot_tidal_ellipse(ellipse, el_color, xoff=0., yoff=0., scale=1):
        # plot conjugated ellipse for inverted y-axis
        gap = 0
        plot_ellipse(np.conjugate(ellipse) * scale + (xoff + 1j * yoff), gap, el_color)

    def ellipses_lims(scale):
        """ Data limits for scaled and shifted ellipses """
        x_span, y_span, margin, y_data_lim = ellipses_span(scale)
        x_data_lim = -(x_span / 2 + margin), x_span / 2 + margin
        y_data_lim = y_data_lim[0] - np.sign(y_span) * margin, y_data_lim[1] + np.sign(y_span) * margin
        return x_data_lim, y_data_lim

    def aspect_func(scale):
        """ Difference between aspect of the area occupied by ellipses and axes aspect.
        Used in ellipse scale optimization.
        """
        x_span, y_span, margin, _ = ellipses_span(scale)
        return abs(ax_aspect - (abs(y_span) + margin) / (x_span + margin))

    def ellipses_span(scale):
        """ Span of scaled and shifted ellipses along x- and y-axis and including y_min, y_max """
        els = np.array([])
        # for el in ell_dict.keys():
        for dep in bin_dict.keys():
            for tc in bin_dict[dep]['ellipses'].keys():
                try:
                    els = np.append(els, bin_dict[dep]['ellipses'][tc] * scale + (0 + 1j * dep))
                    # els = np.append(els, el[1] * scale + (0 + 1j * el[0]))  # el[0] is depth, el[1] is ellipse points
                except:
                    continue
                    # els = np.append(els, 0+0j )
        x_span = 2 * np.nanmax(np.abs(np.real(els)))
        y_data_lim = min([y_min, np.nanmin(np.imag(els))]), max([y_max, np.nanmax(np.imag(els))])
        y_span = y_data_lim[1] - y_data_lim[0]
        margin = 0.1 * min(x_span, abs(y_span))
        return x_span, y_span, margin, y_data_lim

    def set_lims(a):
        """ Ensure axes limits include x_lim, y_lim. Works only for non-inverted axes. """
        if np.min(np.abs(a.get_xlim())) < x_lim[1]:
            a.set_xlim(x_lim)
        a_ylim = a.get_ylim()
        if a_ylim[0] > y_lim[0] or a_ylim[1] < y_lim[1]:
            a.set_ylim(min(a_ylim[0], y_lim[0]), max(a_ylim[1], y_lim[1]))
        return

    instrument_depth = round_to_int(instrument_depth)

    # Major tidal constituents
    major_constit = ['M2', 'K1', 'N2', 'S2', 'O1', 'P1', 'Q1', 'K2']
    bin_dict = {}

    # Iterate through the bins
    # two-level dict with bin depth first level then tidal const. second level
    for i in range(len(ew_lim)):
        result = run_ttide(
            U=ew_lim[i, :],
            V=ns_lim[i, :],
            xin_units='cm/s',
            time_lim=time_lim,
            latitude=latitude,
            constitnames=major_constit
        )

        # Find the (maj,min,inc,phase) for each tidal constituent and use them to calculate the ellipse
        tidal_par, error_par = parse_tidal_constituents(constituents=major_constit, tide_result=result)

        ell_dict = {}
        for k, name in enumerate(major_constit):
            # result['tidecon'] has columns for
            # major, major err, minor, minor err, inc (inclination), inc err, phase, phase err.
            # Missing leading column for freq stored in result['fu']
            # Missing trailing column for SNR stored in result['snr']
            ell_dict[name] = calculate_ellipse(
                major=tidal_par[name][0],  # result['tidecon'][k, 0],
                minor=tidal_par[name][1],  # result['tidecon'][k, 2],
                inc=tidal_par[name][2],  # result['tidecon'][k, 4],
                phase=tidal_par[name][3],  # result['tidecon'][k, 6]
            )

        # Add to dictionary containing all the results
        bin_dict[bin_depth_lim[i]] = {'result': result, 'ellipses': ell_dict}

    # # ellipse points for plotting
    # ell_list = [[(k, calculate_ellipse(el_par_z)) for k, el_par_z in zip(ozk, el_par_case)]
    #             for el_par_case in ell_params]
    # # make a flat list for all cases for unified scaling calculation
    # combined_ell_list = []
    # for one in ell_list:
    #     combined_ell_list.extend(one)

    # include these limits in the plot even if ellipses don't span to them
    buffer = 10  # meters
    y_min = 0
    y_max = np.max(bin_depth_lim) + buffer  # ozk.max()

    # Make the plot
    n_cases = len(major_constit)  # Number of tidal constituents??
    fig, axes = plt.subplots(1, n_cases, sharey='row', figsize=(1.5 * n_cases, 8))
    plt.subplots_adjust(wspace=.0)
    title = f'{station}-{deployment_number} Tidal Ellipses'
    fig.suptitle(title)

    for ax in axes:
        ax.set_aspect('equal', adjustable='datalim')  # this is for correct ellipse aspect
        ax.set_xlabel('Ellipse Size\n(cm/s)')  # do before plt.tight_layout() otherwise gets cut off

    axes[0].set_ylabel('Depth (m)')  # do before plt.tight_layout() otherwise gets cut off

    # do tight_layout before axes[0].get_ylim() for correct numbers (it renders fig?)
    plt.tight_layout()
    # axes aspect to fit the ellipses in
    ax_aspect = abs(np.diff(axes[0].get_ylim()) / np.diff(axes[0].get_xlim()))
    # best ellipse scale to fit the axes
    with np.errstate(divide='ignore', invalid='ignore', all='ignore'):
        optim = minimize_scalar(aspect_func)
    ell_scale = optim.x[0]
    x_lim, y_lim = ellipses_lims(ell_scale)
    # plot ellipses
    for ax, ell_case in zip(axes, major_constit):
        plt.sca(ax)
        for depth in bin_dict.keys():
            # el[0] is depth, el[1] is ellipse points
            plot_tidal_ellipse(bin_dict[depth]['ellipses'][ell_case], 'k', yoff=depth, scale=ell_scale)
        set_lims(ax)  # make sure ellipses are not clipped

    # same xtick for all axes
    tick_base = best_tick(np.ptp(axes[0].get_xlim()) / ell_scale, 6)
    loc = plticker.MultipleLocator(base=tick_base * ell_scale)

    # scaled x-ticks, labels and annotations
    for ax, case in zip(axes, major_constit):
        ax.xaxis.set_major_locator(loc)
        x_ticks = plticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / ell_scale))
        ax.xaxis.set_major_formatter(x_ticks)
        # ax.set_xlabel('Ellipse Size (m/s)')  # moved to earlier in the function
        ax.grid(which='both')
        ax.text(0.05, 0.98, case, ha='left', va='top', transform=ax.transAxes)

    axes[0].invert_yaxis()  # do it only once; shared y-axes applies it to the rest
    # axes[0].set_ylabel('Depth (m)')

    # Save the figure
    plot_name = f'{station}-{deployment_number}_{serial_number}_{instrument_depth}m_tidal_ellipses.png'

    _, plot_name = review_plot_naming(
        plot_title=None, png_name=plot_name, serial_number=serial_number, is_pre_split=False, resampled=resampled
    )

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    plot_dir = get_plot_dir(data_filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_name = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_name)
    plt.close()
    return plot_name


def get_closest_bin_idx(bin_depths: np.ndarray, requested_depth: float):
    """
    Get the closest bin to the requested depth

    Credit: https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    """
    # Evaluate whether the requested depth is too far out of range
    threshold = bin_depths[2] - bin_depths[0]  # meters
    if requested_depth > max(bin_depths) + threshold:
        return None
    else:
        closest_bin = min(bin_depths, key=lambda x: abs(x - requested_depth))
        bin_idx = np.where(bin_depths == closest_bin)[0][0]
        return bin_idx


def default_single_bins(ncdata: xr.Dataset, time_range_idx: tuple, bin_range_idx: tuple):
    """
    Get default bottom, middle, and top bin indices for quiver and rotary spectra plots

    Use percent good data or beam averaged backscatter data to find an uncontaminated shallow bin
    """

    if hasattr(ncdata, 'PCGDAP00'):
        goodness_thresh = 75  # Percentage

        # Take average over time
        if hasattr(ncdata, 'PCGDAP05'):  # Sentinel V fifth beam
            dim1 = 5
        else:
            dim1 = 4
        # Apply prior bin and time limiting
        time_avg_pg = np.zeros(shape=(dim1, bin_range_idx[1] - bin_range_idx[0]))
        time_avg_pg[0, :] = np.nanmean(ncdata.PCGDAP00.data[bin_range_idx[0]:bin_range_idx[1],
                                       time_range_idx[0]:time_range_idx[1]], axis=1)
        time_avg_pg[1, :] = np.nanmean(ncdata.PCGDAP02.data[bin_range_idx[0]:bin_range_idx[1],
                                       time_range_idx[0]:time_range_idx[1]], axis=1)
        time_avg_pg[2, :] = np.nanmean(ncdata.PCGDAP03.data[bin_range_idx[0]:bin_range_idx[1],
                                       time_range_idx[0]:time_range_idx[1]], axis=1)
        time_avg_pg[3, :] = np.nanmean(ncdata.PCGDAP04.data[bin_range_idx[0]:bin_range_idx[1],
                                       time_range_idx[0]:time_range_idx[1]], axis=1)
        if hasattr(ncdata, 'PCGDAP05'):
            time_avg_pg[4, :] = np.nanmean(ncdata.PCGDAP05.data[bin_range_idx[0]:bin_range_idx[1],
                                           time_range_idx[0]:time_range_idx[1]], axis=1)

        # Avg over all 4 or 5 beams to get a single number for each bin
        max_time_avg_pg = np.nanmax(time_avg_pg, axis=0)

        if ncdata.orientation == 'up':
            try:
                shallow = np.where(max_time_avg_pg >= goodness_thresh)[0][-1]  # size of number of bins
            except IndexError:
                warnings.warn(f'Threshold of {goodness_thresh}% not exceeded by time-mean max percent-good data; '
                              f'shallow bin choice defaulting to third-shallowest bin for single bin plots')
                shallow = len(max_time_avg_pg) - 4
            deep = 0
        else:
            try:
                shallow = np.where(max_time_avg_pg >= goodness_thresh)[0][0]  # First good bin
                deep = np.where(max_time_avg_pg >= goodness_thresh)[0][-1]  # Last good bin
            except IndexError:
                warnings.warn(f'Threshold of {goodness_thresh}% not exceeded by time-mean max percent-good data; '
                              f'bin choices defaulting to shallowest bin and deepest bin above water depth for '
                              f'single bin plots')
                shallow = 0
                if ncdata.DISTTRAN.data[-1] > ncdata.water_depth.data:
                    # Bin depths exceed water depth
                    deep = np.where(ncdata.DISTTRAN.data > ncdata.water_depth.data)[0][-1]
                else:
                    # Bin depths do not exceed water depth
                    deep = len(max_time_avg_pg) - 1

    elif ncdata.orientation == 'up':
        # Does not have percent good data
        # Use beam- and time-averaged backscatter if percent good data are not available
        if hasattr(ncdata, 'TNIHCE05'):
            amp_beam_avg = np.nanmean([ncdata.TNIHCE01.data[:, :],
                                       ncdata.TNIHCE02.data[:, :],
                                       ncdata.TNIHCE03.data[:, :],
                                       ncdata.TNIHCE04.data[:, :],
                                       ncdata.TNIHCE05.data[:, :]], axis=(0, 2))
        else:
            amp_beam_avg = np.nanmean([ncdata.TNIHCE01.data[:, :],
                                       ncdata.TNIHCE02.data[:, :],
                                       ncdata.TNIHCE03.data[:, :],
                                       ncdata.TNIHCE04.data[:, :]], axis=(0, 2))

        # Take 1st order differences; do not prepend so that we don't index bad data
        diffs = np.diff(amp_beam_avg)  # , prepend=amp_beam_avg[0]
        # Locate where largest negative differences take place
        # backscatter decreases from bottom to surface, then increases rapidly towards surface
        deep = 0
        diff_increase_threshold = 10
        shallow = np.where(diffs > diff_increase_threshold)[0][0]

    else:
        # Does not have percent good data
        # Orientation == 'down'
        bin_depths = ncdata.instrument_depth.data + ncdata.distance.data
        # Find deepest bin above the sea floor
        deep = np.where(bin_depths < ncdata.water_depth)[0][-1]
        shallow = 0

    middle = int(abs(deep - shallow) / 2)  # Midpoint between shallow and deep bins

    return [shallow, middle, deep]


def get_single_bin_inds(single_bin_inds, single_bin_depths, ncdata: xr.Dataset, time_range_idx,
                        bin_range_idx, bin_depths_lim) -> list:
    """
    Get indices of bins to make rotary spectra and quiver/feather plots of
    """
    if single_bin_inds is None and single_bin_depths is None:
        # single_bin_inds = [5, int(len(bin_depths_lim)/2), len(bin_depths_lim) - 5]
        single_bin_inds = default_single_bins(ncdata, time_range_idx, bin_range_idx)  # (shallow, middle, deep)
    elif single_bin_inds == 'all' or single_bin_depths == 'all':
        single_bin_inds = np.arange(len(bin_depths_lim)).tolist()
    elif single_bin_depths is not None:
        single_bin_inds = [get_closest_bin_idx(bin_depths_lim, d) for d in single_bin_depths]

    # Sort the indices such that the shallowest bin is first and the deepest last
    if ncdata.orientation == 'up':
        single_bin_inds.sort(reverse=True)
    else:
        single_bin_inds.sort()

    return single_bin_inds


def plot_single_bin_velocity(
        time: np.ndarray, U: np.ndarray, V: np.ndarray, depth: np.ndarray, dest_dir, data_filename, station,
        deployment_number, serial_number, instrument_depth, bin_index=None, resampled=None, level0=False,
        filter_type='raw'
):
    """
    Plot with 2 horizontal subplots containing n and e velocities
    """
    if bin_index is None:  # default plot bin nearest the ADCP
        bin_index = 0
    elif bin_index > len(U[0, :]):
        warnings.warn(
            f'Bin index {bin_index} out of range of velocity with dims {U.shape}'
        )
        bin_index = len(U[0, :]) - 1

    bin_depth = np.round(depth[bin_index], 2)

    # vlim = np.nanmax([np.nanmax(abs(V[bin_index, :])), np.nanmax(abs(U[bin_index, :]))])
    # vlim = vlim + (0.1 * vlim)  # add buffer

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(time, V[bin_index, :], linewidth=.8)
    ax[1].plot(time, U[bin_index, :], linewidth=.8)
    plt.suptitle(f'{station}-{deployment_number} {serial_number} Bin {bin_index + 1} ({bin_depth} m) Velocities')
    velocity_names = ['North', 'East']
    if level0:
        velocity_names = [f'Magnetic {x}' for x in velocity_names]

    # Make y axis limits the same
    vlim = np.nanmax(abs(np.array([*ax[0].get_ylim(), *ax[1].get_ylim()])))

    for i, vel_name in zip([0, 1], velocity_names):
        ax[i].set_ylabel('Velocity [m s$^{-1}$]')
        ax[i].set_title(vel_name)
        ax[i].set_ylim((-vlim, vlim))
        ax[i].tick_params(axis='both', direction='in', top=True, right=True)

    plot_name = (f'{station}-{deployment_number}_{serial_number}_{round_to_int(instrument_depth)}m_'
                 f'NE_bin_{round_to_int(bin_depth)}m_{filter_type}.png')

    _, plot_name = review_plot_naming(
        plot_title=None, png_name=plot_name, serial_number=serial_number, is_pre_split=False, resampled=resampled
    )

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    plot_dir = get_plot_dir(data_filename, dest_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_name = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_name)
    plt.close(fig)

    return plot_name


def create_westcoast_plots(
        ncfile, dest_dir, filter_type="Godin", along_angle=None, time_range=None, bin_range=None,
        single_bin_inds=None, single_bin_depths=None, colourmap_lim=None, override_resample=False,
        do_all_plots=False, do_diagnostic=False, do_pressure=False, do_single_bin_ne=False,
        do_ne=False, do_ac=False, do_quiver=False,
        do_single_rotary_spectra=False, do_tidal=False, do_profile_rotary_spectra=False, do_filter_ne=False,
        do_filter_ac=False,
):
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
        - single_bin_inds (optional): list-like object containing the indices of the bins to
                                    make single-bin plots of rotary spectra and feather plots of,
                                    with 0 being the bin closest to the ADCP. Or instead of a
                                    list, the string "all", which makes a plot for each bin.
                                    Defaults to plot bottom, middle, and top bins only if None provided.
        - single_bin_depths (optional): Alternative to single_bin_inds, where the requested depth(s) are
                                        provided in list format and the routine finds the closest bin(s)
                                        to the input depths.
                                        Defaults to plot bottom, middle, and top bins only if None provided.
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
    output_file_list = []

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
        # direction = 'magnetic_east'
    else:
        level0 = False
        # direction = 'east'

    # Check size of file
    ncsize = os.path.getsize(ncfile)  # bytes
    byte2MB = 1. / 1e6
    threshold = 100  # Megabytes
    # If size exceeds threshold, do subsampling
    if ncsize * byte2MB > threshold and not override_resample:
        ncname_resampled = resample_adcp_manual(ncfile, ncdata, dest_dir)
        # Add path to file list
        output_file_list.append(ncname_resampled)
        # Re-open dataset
        ncdata = xr.open_dataset(ncname_resampled)
        resampled = '30min'
    else:
        resampled = None

    # Make diagnostic plots
    if do_diagnostic or do_all_plots:
        fname_diagnostic = plots_diagnostic(ncdata, dest_dir, level0, time_range, bin_range,
                                            resampled)
        output_file_list.append(fname_diagnostic)

    # Limit data if limits are not input by user
    if level0:
        east_vel = 'VEL_MAGNETIC_EAST'
        north_vel = 'VEL_MAGNETIC_NORTH'
    else:
        east_vel = 'LCEWAP01'
        north_vel = 'LCNSAP01'

    time_lim, bin_depths_lim, ns_lim, ew_lim, time_range_idx, bin_range_idx = limit_data(
        ncdata, ncdata[east_vel].data, ncdata[north_vel].data, time_range, bin_range)

    # Plot pressure PRESPR01 vs time
    if do_pressure or do_all_plots:
        if hasattr(ncdata, 'PRESPR01'):
            fname_pres = plot_adcp_pressure(ncdata, dest_dir, resampled)
            output_file_list.append(fname_pres)

    # North/East velocity plots
    if do_ne or do_all_plots:
        fname_ne = make_pcolor_ne(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim,
                                  level0, 'raw', colourmap_lim, resampled)
        output_file_list.append(fname_ne)

    # Along/Cross-shelf velocity plots
    if do_ac or do_all_plots:
        fname_ac = make_pcolor_ac(ncdata, dest_dir, time_lim, bin_depths_lim, ns_lim, ew_lim,
                                  'raw', along_angle, colourmap_lim, resampled)
        output_file_list.append(fname_ac)

    # Single-bin plots
    single_bin_inds = get_single_bin_inds(single_bin_inds, single_bin_depths, ncdata,
                                          time_range_idx, bin_range_idx, bin_depths_lim)

    if do_single_bin_ne or do_all_plots:
        # Default to choosing the bin closest to the instrument
        # single_bin_inds = [shallow, middle, deep] for both down and up facing
        vel_bin_idx = single_bin_inds[0] if ncdata.orientation == 'down' else single_bin_inds[-1]

        fname_single_ne = plot_single_bin_velocity(
            time_lim, U=ew_lim, V=ns_lim, depth=bin_depths_lim, dest_dir=dest_dir, data_filename=ncdata.filename,
            station=ncdata.station, deployment_number=ncdata.deployment_number,
            serial_number=ncdata.instrument_serial_number.data, instrument_depth=ncdata.instrument_depth.data,
            bin_index=vel_bin_idx, resampled=resampled, level0=level0, filter_type='raw'
        )
        output_file_list.append(fname_single_ne)

    # Feather/quiver plots

    # Returns a list of names not a single name; apply to non-filtered data
    if do_quiver or do_all_plots:
        fnames_quiver = quiver_plot(dest_dir, ncdata.filename, ncdata.station, ncdata.deployment_number,
                                    ncdata.instrument_depth.data, ncdata.instrument_serial_number.data,
                                    time_lim, bin_depths_lim,
                                    ns_lim, ew_lim, single_bin_inds, resampled)
        output_file_list += fnames_quiver

    # Rotary spectra
    if do_single_rotary_spectra or do_all_plots:
        # Initialize a list to hold the names of all output files
        fnames_rot_spec = []
        # Iterate through the bins
        for bin_idx in single_bin_inds:
            # Only proceed if the bin index is in range
            if bin_idx < len(bin_depths_lim):
                # Add extra step in case the tidal analysis fails
                try:
                    fname_rot_spec = make_plot_rotary_spectra(
                        dest_dir, ncdata.filename, ncdata.station, ncdata.deployment_number,
                        ncdata.instrument_depth.data, ncdata.instrument_serial_number.data,
                        bin_number=bin_idx, bin_depths_lim=bin_depths_lim, time_lim=time_lim,
                        ns_lim=ns_lim, ew_lim=ew_lim, latitude=ncdata.latitude.data,
                        resampled=resampled, axis=-1
                    )
                    fnames_rot_spec.append(fname_rot_spec)
                except ValueError as e:
                    print(f'Single-bin rotary spectra plot failed with error: {e}')
            else:
                print(f'Warning: Bin index {bin_idx} for rotary spectra out of range of limited bins with '
                      f'length {len(bin_depths_lim)}')

        output_file_list += fnames_rot_spec

    # Profile plots of tidal ellipses
    if do_tidal or do_all_plots:
        try:
            fname_tidal_ellipse = make_plot_tidal_ellipses(
                dest_dir, ncdata.filename, ncdata.station, ncdata.deployment_number,
                ncdata.instrument_serial_number.data, ncdata.instrument_depth.data, ncdata.latitude.data,
                time_lim, bin_depths_lim, ns_lim, ew_lim, resampled
            )
            output_file_list.append(fname_tidal_ellipse)
        except (ValueError, IndexError) as e:
            warnings.warn(f'Tidal analysis failed with error: {e}')

    # pcolor (pseudocolour) depth profile plot of rotary spectra
    if do_profile_rotary_spectra or do_all_plots:
        try:
            fnames_depth_prof = make_depth_prof_rot_spec(
                dest_dir, ncdata.filename, station=ncdata.station, deployment_number=ncdata.deployment_number,
                serial_number=ncdata.instrument_serial_number.data, instrument_depth=ncdata.instrument_depth.data,
                bin_depths_lim=bin_depths_lim, ns_lim=ns_lim, ew_lim=ew_lim, time_lim=time_lim
            )
            output_file_list += fnames_depth_prof
        except ValueError as e:
            print(f'Depth profile rotary spectra plot failed with error: {e}')

    # Redo part of process with tidal-filtered data
    if do_filter_ne or do_filter_ac or do_all_plots:
        if filter_type == "Godin":
            ew_filt, ns_filt = filter_godin(ncdata)
        elif filter_type.endswith("h"):
            ew_filt, ns_filt = filter_XXh(ncdata, num_hrs=int(filter_type[:-1]))
        else:
            ValueError("filter_type value not understood !")

        # Limit data
        time_lim, bin_depths_lim, ns_filt_lim, ew_filt_lim, time_range_idx, bin_range_idx = limit_data(
            ncdata, ew_filt, ns_filt, time_range, bin_range)

        # Northward and eastward velocity colormesh plots
        if do_filter_ne or do_all_plots:
            fname_ne_filt = make_pcolor_ne(ncdata, dest_dir, time_lim, bin_depths_lim, ns_filt_lim,
                                           ew_filt_lim, level0, filter_type, colourmap_lim, resampled)
            output_file_list.append(fname_ne_filt)

        # Along-shore/cross-shore
        if do_filter_ac or do_all_plots:
            fname_ac_filt = make_pcolor_ac(ncdata, dest_dir, time_lim, bin_depths_lim, ns_filt_lim,
                                           ew_filt_lim, filter_type, along_angle, colourmap_lim,
                                           resampled)
            output_file_list.append(fname_ac_filt)

        # # Compare filtered data with raw data
        # fname_binplot = binplot_compare_filt(ncdata, dest_dir, time_lim, ew_lim, ew_filt_lim,
        #                                      filter_type, direction, resampled)

    # Close netCDF file
    ncdata.close()

    return output_file_list
