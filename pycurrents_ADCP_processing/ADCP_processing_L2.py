"""
author: Hana Hourston

Credits:
Pre-process plotting code adapted from David Spear (originally in MatLab)
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

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pycurrents_ADCP_processing import plot_westcoast_nc_LX


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
        print(dataset.TNIHCE05.data)
        return 0
    except AttributeError:
        if dataset.instrumentSubtype == 'Sentinel V':
            return 1
        else:
            return 0


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


def date2ns(date):
    # Convert datetime64 object to nanoseconds since 1970-01-01
    return pd.to_datetime(date).value


def add_attrs_prexmcat(dataset, name_ctd):
    """
    Adds attributes to CTD pressure variable. dataset should already include the PREXMCAT variable
    Inputs:
        - dataset: dataset-type object created by reading in a netCDF ADCP file with the xarray package
        - name_ctd: string containing the name of the CTD file whose pressure data was used
    Outputs:
        None
    """

    var = dataset.PREXMCAT
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = dataset.PRESPR01.units
    var.attrs['_FillValue'] = dataset.PRESPR01.encoding['_FillValue']
    var.attrs['long_name'] = 'pressure'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = dataset.PRESPR01.sensor_depth
    var.attrs['serial_number'] = dataset.serial_number
    # var.attrs['sdn_parameter_name'] = 'Pressure (spatial co-ordinate) exerted by the water body by profiling ' \
    #                                  'pressure sensor and corrected to read zero at sea level'
    var.attrs['ctd_file_used_for_pressure'] = os.path.basename(name_ctd)
    var.attrs['sdn_parameter_name'] = 'Pressure (measured variable) exerted by the water body by semi-fixed ' \
                                      'moored SBE MicroCAT'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UPDB'
    var.attrs['sdn_uom_name'] = 'Decibars'
    var.attrs['standard_name'] = 'sea_water_pressure'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    return


def add_pressure_ctd(nc_adcp, nc_ctd, name_ctd):
    """
    Function for calculating ADCP pressure from SBE (Sea-Bird CTD) pressure on the same mooring line
    Inputs:
        - nc_adcp: dataset-type object created by reading in a netCDF ADCP file with the xarray package
        - nc_ctd: dataset-type object created by reading in a netCDF CTD file with the xarray package
        - name_ctd: the name of the input ctd file (string type)
    Outputs:
        - out_adcp_dataset: a new dataset-type object, the same as nc_adcp but with the new pressure
                            variable PREXMCAT derived from CTD pressure.
    """

    # Initialize new pressure array with nans
    pres_adcp_from_ctd = np.empty(shape=nc_adcp.PRESPR01.data.shape, dtype='float32')
    pres_adcp_from_ctd[:] = np.nan
    print(pres_adcp_from_ctd.shape)

    min2sec = 60
    sec2nsec = 1e9
    min2nano = min2sec * sec2nsec

    # Calculate ratio of sampling time increments
    time_increment_adcp = date2ns(nc_adcp.time.data[1]) - date2ns(nc_adcp.time.data[0])
    time_increment_ctd = date2ns(nc_ctd.time.data[1]) - date2ns(nc_ctd.time.data[0])
    time_increment_ratio = time_increment_adcp / time_increment_ctd
    print(time_increment_ratio)

    if np.round(time_increment_ratio, decimals=1) < 1:
        if np.round(time_increment_ratio, decimals=1) == 1/1.5:
            index_increment_adcp = 3
            index_increment_ctd = 2
        else:
            # use_ratio_as_ctd_step = False  # use on adcp instead
            # time_increment_ratio = int(np.round(1 / time_increment_ratio, decimals=0))
            index_increment_adcp = 1
            index_increment_ctd = int(np.round(1 / time_increment_ratio, decimals=0))
    else:
        if np.round(time_increment_ratio, decimals=1) == 1.5:
            # To account for time ratios of 1.5 (e.g. adcp every 15 min, ctd every 10 min)
            # time_increment_ratio = int(np.round(2 * time_increment_ratio, decimals=0))
            index_increment_adcp = 2
            index_increment_ctd = 3
        else:
            # time_increment_ratio = int(np.round(time_increment_ratio, decimals=0))
            index_increment_adcp = int(np.round(time_increment_ratio, decimals=0))
            index_increment_ctd = 1

    print('Index increment for adcp =', index_increment_adcp)
    print('Index increment for ctd =', index_increment_ctd)
    max_increment = index_increment_adcp if index_increment_adcp > index_increment_ctd else index_increment_ctd

    # Try finding index of start time of one time in the other time range from [0] up to [time_increment_ratio]
    # and index of end time of one time in the other time range from [-1] up to [-time_increment_ratio]

    # Initialize indices
    start_index_in_CTD = -9
    end_index_in_CTD = -9
    start_index_in_ADCP = -9
    end_index_in_ADCP = -9

    # Due to differences in sampling time intervals between instruments, may need to take later start time
    # and earlier end time, in order to match up times as closely as possible
    earliest_start_index = -9
    latest_end_index = -9

    # Step 1

    # Find index of first time, either in the ctd time range or in the adcp time range
    # Test if CTD time starts before the ADCP time
    if date2ns(nc_ctd.time.data[0]) < date2ns(nc_adcp.time.data[0]):
        # adcp start time is in ctd time range
        print("CTD start-time earlier than ADCP start-time !")

        for start_index in range(max_increment):
            print(start_index)
            for t_i in range(len(nc_ctd.time.data)):
                # Find the index of the first adcp time measurement in the ctd time range within 4 min
                if np.abs(date2ns(nc_ctd.time.data[t_i]) - date2ns(
                        nc_adcp.time.data[start_index])) < 4 * min2nano:
                    start_index_in_CTD = t_i
                    break
            # Break out of loop if start_index_in_CTD is found
            # Otherwise try the next time in the adcp time range
            if start_index_in_CTD != -9:
                earliest_start_index = start_index  # in CTD time
                break
    else:
        # adcp start time before first ctd time
        print("ADCP start-time earlier than CTD start-time !")

        for start_index in range(max_increment):
            print(start_index)
            for t_j in range(len(nc_adcp.time.data)):
                # Find the index of the first ctd time measurement in the adcp time range within 4 min
                if np.abs(date2ns(nc_adcp.time.data[t_j]) - date2ns(
                        nc_ctd.time.data[start_index])) < 4 * min2nano:
                    start_index_in_ADCP = t_j
                    break
            # Break out of loop if start_index_in_CTD is found
            # Otherwise try the next time in the adcp time range
            if start_index_in_ADCP != -9:
                earliest_start_index = start_index  # in ADCP time
                break

    # Find index of last time, either in the ctd time range or in the adcp time range
    # end_index is the index where the earlier end time matches a time in the other time range
    if date2ns(nc_ctd.time.data[-1]) > date2ns(nc_adcp.time.data[-1]):
        print("CTD end-time later than ADCP end-time !")
        for end_index in range(1, max_increment + 1):
            for t_i in range(len(nc_ctd.time.data)):
                # Find the index of the last adcp time measurement in the ctd time range within 4 min
                if np.abs(date2ns(nc_ctd.time.data[t_i]) - date2ns(
                        nc_adcp.time.data[-end_index])) < 4 * min2nano:
                    # Add +1 so that the time range is inclusive?
                    end_index_in_CTD = t_i  # + 1
                    break
            # Break out of loop if start_index_in_CTD is found
            # Otherwise try the next time in the adcp time range
            if end_index_in_CTD != -9:
                latest_end_index = -end_index  # in CTD time
                break
    else:
        # adcp end time is after last ctd time
        print("SBE end-time earlier than ADCP end-time !")

        for end_index in range(1, max_increment + 1):
            print(end_index)
            for t_j in range(len(nc_adcp.time.data)):
                if np.abs(date2ns(nc_adcp.time.data[t_j]) - date2ns(
                        nc_ctd.time.data[-end_index])) < 4 * min2nano:
                    # Add +1 so that the time range is inclusive?
                    end_index_in_ADCP = t_j  # + 1
                    break
            # Break out of loop if start_index_in_CTD is found
            # Otherwise try the next time in the adcp time range
            if end_index_in_ADCP != -9:
                latest_end_index = -end_index  # in ADCP time
                break

    print('start_index_in_CTD:', start_index_in_CTD, sep=' ')
    print('end_index_in_CTD:', end_index_in_CTD, sep=' ')
    print('start_index_in_ADCP:', start_index_in_ADCP, sep=' ')
    print('end_index_in_ADCP:', end_index_in_ADCP, sep=' ')

    print('earliest_start_index:', earliest_start_index, sep=' ')
    print('latest_end_index:', latest_end_index, sep=' ')

    if earliest_start_index == -9 or latest_end_index == -9:
        IndexError('Invalid earliest start index or latest end index')

    # Step 2

    # Fill in pressure array that was initialized with nans
    if start_index_in_CTD != -9:
        print('Start adcp time is in ctd time range')
        if end_index_in_CTD != -9:
            print('End adcp time is in ctd time range')
            pres_adcp_from_ctd[earliest_start_index:latest_end_index
                               :index_increment_ctd] = nc_ctd.PRESPR01.data[
                                                       start_index_in_ADCP:end_index_in_ADCP:index_increment_adcp]
        else:
            print('End adcp time is after ctd time range')
            # len_pres_ctd_interval = length of the ctd pres interval that we want to insert into the new adcp pres
            len_pres_ctd_interval = len(nc_ctd.PRESPR01.data[start_index_in_ADCP::time_increment_ratio])
            pres_adcp_from_ctd[earliest_start_index:len_pres_ctd_interval
                               :index_increment_ctd] = nc_ctd.PRESPR01.data[
                                                       start_index_in_ADCP:latest_end_index:index_increment_adcp]
    else:
        print('adcp time starts before ctd time range')
        if end_index_in_CTD != -9:
            print('adcp end time is within ctd time range')
            len_pres_ctd_interval = len(nc_ctd.PRESPR01.data[
                                        earliest_start_index:end_index_in_CTD:time_increment_ratio])
            pres_adcp_from_ctd[start_index_in_ADCP:len_pres_ctd_interval
                               :index_increment_ctd] = nc_ctd.PRESPR01.data[
                                                       earliest_start_index:end_index_in_CTD:index_increment_adcp]
        else:
            # End ctd time is in adcp time range
            print('adcp end time is after ctd time range')
            pres_adcp_from_ctd[start_index_in_ADCP:end_index_in_ADCP
                               :index_increment_ctd] = nc_ctd.PRESPR01.data[
                                                       earliest_start_index:latest_end_index:index_increment_adcp]

    # Step 3

    # Apply depth corrections to new pressure
    # Calculate depth difference between ADCP and SBE
    # Positive if ADCP deeper than SBE
    depth_diff = nc_adcp.instrument_depth - nc_ctd.instrument_depth.data  # meters
    print(depth_diff)
    pres_adcp_from_ctd = pres_adcp_from_ctd + np.round(depth_diff, decimals=1)

    out_adcp_dataset = nc_adcp.assign(PREXMCAT=(('time'), pres_adcp_from_ctd))
    print('Created new xarray dataset object containing CTD-derived pressure')

    add_attrs_prexmcat(out_adcp_dataset, name_ctd=name_ctd)
    print('Added attributes to PREXMCAT')

    return out_adcp_dataset


def plots_preprocess_L2(d):
    """
    Preliminary plots:
    (1) Backscatter against depth, (2) mean velocity, and (3) principle component direction
    Inputs:
        - d: dataset-type object created by reading in a netCDF ADCP file with the xarray package
    Outputs:
        None
    """
    # Subplot 1/3: Plot avg backscatter against depth

    # First calculate depths
    depths = calculate_depths(d)

    # Check if vertical beam data present in Sentinel V file
    flag_vb = vb_flag(d)

    # Calculate average backscatter (amplitude intensity)
    amp_mean_b1 = np.nanmean(d.TNIHCE01.data, axis=1)
    amp_mean_b2 = np.nanmean(d.TNIHCE02.data, axis=1)
    amp_mean_b3 = np.nanmean(d.TNIHCE03.data, axis=1)
    amp_mean_b4 = np.nanmean(d.TNIHCE04.data, axis=1)

    if d.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        amp_mean_b5 = np.nanmean(d.TNIHCE05.data, axis=1)
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
    if d.orientation == 'up':
        plt.gca().invert_yaxis()

    # Subplot 2/3: Mean velocity
    ax = fig.add_subplot(1, 3, 2)

    # Calculate average velocities
    u_mean = np.zeros(int(d.numberOfCells), dtype='float32')
    v_mean = np.zeros(int(d.numberOfCells), dtype='float32')
    w_mean = np.zeros(int(d.numberOfCells), dtype='float32')

    for i in range(len(u_mean)):
        u_mean[i] = np.nanmean(d.LCEWAP01.data[i, :])
        v_mean[i] = np.nanmean(d.LCNSAP01.data[i, :])
        w_mean[i] = np.nanmean(d.LRZAAP01.data[i, :])

    if d.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        w5_mean = np.zeros(int(d.numberOfCells), dtype='float32')
        for i in range(len(u_mean)):
            w5_mean[i] = np.nanmean(d.LRZUVP01.data[i, :])

        names = ['LCEWAP01', 'LCNSAP01', 'LRZAAP01', 'LRZUVP01']
        vels = [u_mean, v_mean, w_mean, w5_mean]
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
    if d.orientation == 'up':
        plt.gca().invert_yaxis()

    # Subplot 3/3: Principal axis

    orientation = np.zeros(int(d.numberOfCells), dtype='float32')
    for ibin in range(len(orientation)):
        xx = d.LCEWAP01[ibin, :]
        yy = d.LCNSAP01[ibin, :]
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
    if d.orientation == 'up':
        plt.gca().invert_yaxis()

    # Create centred figure title
    fig.suptitle('{}-{} {} at {} m depth'.format(d.station, d.deployment_number, d.serial_number,
                                                 d.instrument_depth), fontweight='semibold')

    # Create L2_Python_plots subfolder
    plot_dir = './L2_Python_plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig_name = plot_dir + '{}-{}_{}_nc_preprocess.png'.format(
        d.station.lower(), str(d.deployment_number), d.serial_number)
    fig.savefig(fig_name)
    plt.close()

    return


def flag_by_pres(d, use_prexmcat=False):
    """
    For flagging velocity data by bins where pressure is negative at each time step. This function
    is for upward-facing instruments only.
    Inputs:
        - d: dataset-type object created by reading in a netCDF ADCP file with the xarray package
        - use_prexmcat: Defaults to False, which uses PRESPR01. True means PREXMCAT will be used.
    Outputs:
        - bad_bin_list: an array containing the index of the first bad bin at each time step;
                        "bad bins" are designated as having negative pressure or are below ocean
                        floor depth
    """

    if use_prexmcat:
        print('Using PREXMCAT as pressure data')
        pressure = d.PREXMCAT
    else:
        print('Using PRESPR01 as pressure data')
        pressure = d.PRESPR01

    # Obtain the number of leading and trailing ensembles that were set to nans during L1 processing
    start_end_L1 = plot_westcoast_nc_LX.get_L1_start_end(ncdata=d)
    print('Start and end indices between nans from L1:', start_end_L1[0], start_end_L1[1], sep=' ')

    # List of indices where bins start to have negative pressure
    bad_bin_list = np.zeros(shape=d.time.data.shape, dtype='int32') #initialize with zeros
    bad_bin_list[start_end_L1[0]:start_end_L1[1]] = -1 #assume no bad bins in this range; index=-1

    # Flag by negative pressure timestep by timestep
    for t_i in range(start_end_L1[0], start_end_L1[1]):
        flag = 0
        # Iterate through bins
        for b_i in range(len(d.distance.data)):  # bin with index 0 == 1st bin
            # Calculate pressure at each bin depth by subtracting each bin distance from pressure at time step
            bin_pres = pressure.data[t_i] - d.distance.data[b_i]
            if bin_pres < 0:
                # Identify first bin index with negative pressure
                bad_bin_1 = b_i
                bad_bin_list[t_i] = bad_bin_1
                flag += 1
            if flag != 0:
                break

    # Flag velocity data
    flag_vb = vb_flag(d)
    print(flag_vb)
    if d.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        vels_QC = [d.LCEWAP01_QC, d.LCNSAP01_QC, d.LCNSAP01_QC, d.LRZUVP01_QC]
    else:
        vels_QC = [d.LCEWAP01_QC, d.LCNSAP01_QC, d.LCNSAP01_QC]

    for v_QC in vels_QC:
        for t_i in range(start_end_L1[0], start_end_L1[1]):
            # Skip time steps with no bad bins
            if bad_bin_list[t_i] != -1:
                v_QC[bad_bin_list[t_i]:, t_i] = 4  # bad_value flag=4

    d.attrs['processing_history'] = d.processing_history + " Level 2 processing was performed on the data."
    d.attrs['processing_history'] = d.processing_history + " Bins with negative pressure (i.e. were located above" \
                                                           " sea level) were flagged as bad_value" \
                                                           " timestep by timestep."

    return bad_bin_list


def flag_below_seafloor(d):
    """
    Flag by depth when bin depth > ocean depth, timestep by timestep. This function is for
    downward-facing instruments only.
    Inputs:
        - d: dataset-type object created by reading in a netCDF ADCP file with the xarray package
    Outputs:
        - bad_bin_list: an array containing the index of the first bad bin at each time step;
                        "bad bins" are designated as having negative pressure or are below ocean
                        floor depth
    """

    # Obtain the number of leading and trailing ensembles that were set to nans during L1 processing
    start_end_L1 = plot_westcoast_nc_LX.get_L1_start_end(ncdata=d)
    print('Start and end indices between nans from L1:', start_end_L1[0], start_end_L1[1], sep=' ')

    bad_bin_list = np.zeros(shape=d.time.data.shape, dtype='int32') #initialize with zeros
    bad_bin_list[start_end_L1[0]:start_end_L1[1]] = -1 #assume no bad bins in range
    for t_i in range(start_end_L1[0], start_end_L1[1]):
        flag = 0
        # Find shallowest bin that is within the ocean floor and flag it and all deeper ones as 4 == bad_data
        for b_i in range(len(d.distance.data)):
            bin_depth = d.instrument_depth + d.distance.data[b_i]
            if bin_depth > d.water_depth:
                bad_bin_1 = b_i
                bad_bin_list[t_i] = bad_bin_1
                flag += 1
            if flag != 0:
                break

    # Flag velocity data
    flag_vb = vb_flag(d)
    if d.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        vels_QC = [d.LCEWAP01_QC, d.LCNSAP01_QC, d.LCNSAP01_QC, d.LRZUVP01_QC]
    else:
        vels_QC = [d.LCEWAP01_QC, d.LCNSAP01_QC, d.LCNSAP01_QC]

    for v_QC in vels_QC:
        for t_i in range(start_end_L1[0], start_end_L1[1]):
            # If time is after deployment time and before recovery time, from L1
            if bad_bin_list[t_i] != -1:
                v_QC[bad_bin_list[t_i]:, t_i] = 4  # bad_value flag=4
                v_QC[:bad_bin_list[t_i], t_i] = 2  # probably_good_value flag=2, or good_value flag=1

    d.attrs['processing_history'] = d.processing_history + " Level 2 processing was performed on the data."
    d.attrs['processing_history'] = d.processing_history + " Bins below the ocean floor depth were flagged as" \
                                                           " bad_value timestep by timestep."
    d.attrs['processing_history'] = d.processing_history + "The remaining bins were flagged as " \
                                                           "probably_good_value."

    return bad_bin_list


def flag_by_backsc(d, bad_bin_list):
    """
    Flag where beam-averaged backscatter increases at each time step for upwards-facing ADCPs
    Inputs:
        - d: dataset-type object created by reading in a netCDF ADCP file with the xarray package
        - bad_bin_list: numpy array containing the index of the first "bad" bin (bin where pressure is
                        negative) obtained from the function flag_by_pres()
    Outputs:
        - susp_bin_list: an array containing the index of the first suspicious bin at each time step;
                         counting away from the instrument. "Suspicious bins" define bins with
                         backscatter increases.
    """

    # Obtain the number of leading and trailing ensembles that were set to nans during L1 processing
    # From these get the start index after the nans and the end index before the nans
    start_end_L1 = plot_westcoast_nc_LX.get_L1_start_end(ncdata=d)
    print('Start and end indices from L1:', start_end_L1[0], start_end_L1[1], sep=' ')

    # Take beam-averaged backscatter, excluding data from before deployment and after recovery
    # Beam-averaged backscatter is shorter than time series
    flag_vb = vb_flag(d)
    if d.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        # Include amplitude intensity from vertical 5th beam
        amp_beam_avg = np.nanmean([d.TNIHCE01.data[:, start_end_L1[0]:start_end_L1[1]],
                                   d.TNIHCE02.data[:, start_end_L1[0]:start_end_L1[1]],
                                   d.TNIHCE03.data[:, start_end_L1[0]:start_end_L1[1]],
                                   d.TNIHCE04.data[:, start_end_L1[0]:start_end_L1[1]],
                                   d.TNIHCE05.data[:, start_end_L1[0]:start_end_L1[1]]], axis=0)
    else:
        amp_beam_avg = np.nanmean([d.TNIHCE01.data[:, start_end_L1[0]:start_end_L1[1]],
                                   d.TNIHCE02.data[:, start_end_L1[0]:start_end_L1[1]],
                                   d.TNIHCE03.data[:, start_end_L1[0]:start_end_L1[1]],
                                   d.TNIHCE04.data[:, start_end_L1[0]:start_end_L1[1]]], axis=0)

    # Then determine bin where beam-averaged backscatter increases for each timestep
    susp_bin_list = np.zeros(shape=d.time.data.shape, dtype='int32') #initialize with zeros
    susp_bin_list[start_end_L1[0]:start_end_L1[1]] = -1 #assume no suspicious bins in range
    print(susp_bin_list.shape)
    print(amp_beam_avg.shape)
    print(range(start_end_L1[0], start_end_L1[1]))

    # Iterate through time stamps between the deployment and recovery times
    for t_i in range(start_end_L1[0], start_end_L1[1]):
        flag = 0
        # Iterate through bin numbers. Start at index=2 because sometimes one-bin increases occur in the first
        # 2 bins closest to ADCP, not near the surface.
        for b_i in range(2, len(d.distance.data)):
            if amp_beam_avg[b_i, t_i - start_end_L1[0]] > amp_beam_avg[b_i - 1, t_i - start_end_L1[0]]:
                # Index of first suspicious bin counting away from the instrument
                # print(b_i)
                susp_bin_list[t_i] = b_i
                flag += 1
            if flag == 1:
                break

    print('Median index over time of first bin where beam-averaged backscatter increases:',
          np.median(susp_bin_list[start_end_L1[0]:start_end_L1[1]]), sep=' ')

    # Flag the qc variables
    if d.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        vels_QC = [d.LCEWAP01_QC.data, d.LCNSAP01_QC.data, d.LCNSAP01_QC.data, d.LRZUVP01_QC.data]
    else:
        vels_QC = [d.LCEWAP01_QC.data, d.LCNSAP01_QC.data, d.LCNSAP01_QC.data]

    print(d.LCEWAP01_QC.data.shape)

    for v_QC in vels_QC:
        # Iterate through time stamps
        for t_i in range(start_end_L1[0], start_end_L1[1]):
            # Iterate through bins
            for b_i in range(len(d.distance.data)):
                if susp_bin_list[t_i] != -1:
                    if bad_bin_list[t_i] != -1:
                        v_QC[susp_bin_list[t_i]:bad_bin_list[t_i], t_i] = 3  # probably_bad_value
                        v_QC[:susp_bin_list[t_i], t_i] = 2  # probably_good_value flag=2, or good_value flag=1
                    else:
                        #bad_bin_list[t_i] == -1 so no bad bins
                        v_QC[susp_bin_list[t_i]:, t_i] = 3  # probably_bad_value
                        v_QC[:susp_bin_list[t_i], t_i] = 2  # probably_good_value flag=2, or good_value flag=1
                else:
                    #susp_bin_list[t_i] and bad_bin_list[t_i] == -1 so no suspicious and no bad bins
                    v_QC[:, t_i] = 2  # probably_good_value flag=2, or good_value flag=1

    print('Velocity qc variables updated')

    d.attrs['processing_history'] = d.processing_history + " Bins where beam-averaged backscatter increases were" \
                                                           " flagged as probably_bad_value timestep by timestep."

    d.attrs['processing_history'] = d.processing_history + " The remaining bins were flagged as " \
                                                           "probably_good_value."

    return susp_bin_list


def plot_pres_compare(d):
    """
    For plotting ADCP pressure for ADCPs without pressure sensors
    Such ADCP files had pressure calculated from static instrument depth in L1 processing
    Requires PREXMCAT data: ADCP pressure calculated from CTD pressure
    Requires PRESPR01 data: ADCP pressure calculated from static instrument depth in L1
    Inputs:
        - d: netCDF ADCP data in xarray dataset format
    Outputs:
        None
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    f1 = ax.plot(d.time.data, d.PREXMCAT.data, linewidth=1, color='r',
                 label='PREXMCAT (CTD-derived)')
    f2 = ax.plot(d.time.data, d.PRESPR01.data, linewidth=1, color='k',
                 label='PRESPR01 (static)')
    ax.set_ylim(bottom=0)  # don't set top limit
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Pressure (dbar)')
    ax.legend(loc='lower left')
    ax.set_title('{}-{} {} at {} m depth: Static VS CTD-derived pressure'.format(
        d.attrs['station'], d.attrs['deployment_number'], d.attrs['serial_number'],
        d.attrs['instrument_depth']))

    # Create L2_Python_plots subfolder if not made already
    plot_dir = './L2_Python_plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig_name = plot_dir + '{}-{}_{}_CTD_pressure_compare.png'.format(
        d.station.lower(), str(d.deployment_number), d.serial_number)
    fig.savefig(fig_name)

    return


def plot_backscatter_qc(d):
    """
    Calculate depths from bin distances from instrument
    Inputs:
        - d: dataset-type object created by reading in a netCDF ADCP file with the xarray package
        - bad_bin_list: an array containing the index of the first bad bin at each time step;
                        "bad bins" are designated as having negative pressure or are below ocean floor depth
        - susp_bin_list: an array containing the index of the first suspicious bin at each time step;
                         "suspicious bins" are designated as observing backscatter increases.
                         This kwarg is optional as it only applies for upwards-facing ADCPs.
    Outputs:
        None
    """

    # Calculate bin depths
    depths = calculate_depths(d)

    # Calculate average backscatter (amplitude intensity) over time for each beam
    amp_mean_b1 = np.nanmean(d.TNIHCE01.data, axis=1)
    amp_mean_b2 = np.nanmean(d.TNIHCE02.data, axis=1)
    amp_mean_b3 = np.nanmean(d.TNIHCE03.data, axis=1)
    amp_mean_b4 = np.nanmean(d.TNIHCE04.data, axis=1)

    # Check if vertical beam data present in Sentinel V file
    flag_vb = vb_flag(d)
    if d.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        amp_mean_b5 = np.nanmean(d.TNIHCE05.data, axis=1)
        # Make lists of mean amplitude and corresponding plotting colours
        amp = [amp_mean_b1, amp_mean_b2, amp_mean_b3, amp_mean_b4, amp_mean_b5]
        colours = ['b', 'g', 'c', 'm', 'y']
    else:
        amp = [amp_mean_b1, amp_mean_b2, amp_mean_b3, amp_mean_b4]
        colours = ['b', 'g', 'c', 'm']

    amp_mean_all = np.nanmean(amp, axis=0)

    # Do backscatter increase and negative pressure flagging specially for each time time-avged beam
    mean_bad_bin = 0
    mean_susp_bin = 0
    if d.orientation == 'up':
        try:
            pres_mean = np.nanmean(d.PREXMCAT.data)
        except AttributeError:
            pres_mean = np.nanmean(d.PRESPR01.data)

        for i in range(len(d.distance.data)):
            bin_pres = pres_mean - d.distance.data[i]
            if bin_pres < 0:
                mean_bad_bin = i
                break

        for i in range(2, len(amp_mean_all)):
            if amp_mean_all[i] > amp_mean_all[i-1]:
                mean_susp_bin = i
                break
    else:
        for i in range(len(d.distance.data)):
            bin_depth = d.instrument_depth + d.distance.data[i]
            if bin_depth > d.water_depth:
                mean_bad_bin = i

    # Plot flagged average backscatter (QC):
    # Plot Good, Suspicious, and Bad data separately on the same plot
    fig = plt.figure(figsize=(4, 8.5))  # figsize=(11, 8.5), (8, 8.5) for two subplots
    ax = fig.add_subplot(1, 1, 1)  # (nrows, ncols, index)
    beam_no = 1
    for dat, col in zip(amp, colours):
        if mean_bad_bin != 0:
            print('Median bad bin is not zero') #plot bad_data
            f1 = ax.plot(dat[mean_bad_bin - 1:], depths[mean_bad_bin - 1:], linewidth=1, marker='o', markersize=2,
                         color='r')
            if mean_susp_bin != 0:
                print('Median number of bins flagged by backscatter increases is not zero')  # Plot suspicious data
                f2 = ax.plot(dat[mean_susp_bin - 1:mean_bad_bin], depths[mean_susp_bin - 1:mean_bad_bin],
                             linewidth=1, marker='o', markersize=2, color='tab:orange')
                # Plot good_data
                f3 = ax.plot(dat[:mean_susp_bin], depths[:mean_susp_bin], label='Beam {}'.format(beam_no),
                             linewidth=1, marker='o', markersize=2, color=col)
            else:
                # Zero suspicious bins (bins where backscatter increases); plot good_data limited by bad_data
                f2 = ax.plot(dat[:mean_bad_bin], depths[:mean_bad_bin], label='Beam {}'.format(beam_no), linewidth=1,
                             marker='o', markersize=2, color=col)
        else:
            print('Median zero bad bins (bins where pressure is negative)')
            if mean_susp_bin != 0:
                print('Median number of bins flagged by backscatter increases is not zero')
                f1 = ax.plot(dat[mean_susp_bin - 1:], depths[mean_susp_bin - 1:],
                             linewidth=1, marker='o', markersize=2, color='tab:orange')
                f2 = ax.plot(dat[:mean_susp_bin], depths[:mean_susp_bin], label='Beam {}'.format(beam_no),
                             linewidth=1, marker='o', markersize=2, color=col)
            else:
                print('Zero bins where backscatter increases')
                f1 = ax.plot(dat[:], depths[:], label='Beam {}'.format(beam_no), linewidth=1,
                             marker='o', markersize=2, color=col)
        beam_no += 1

    # Add flag plot text
    if mean_bad_bin != 0:
        print(depths[mean_bad_bin:])
        y_pos = np.nanmean(depths[mean_bad_bin:])
        ax.text(x=np.max(amp_mean_b1), y=y_pos, s='Flag=4',
                horizontalalignment='right', verticalalignment='center', fontsize=10)
    if mean_susp_bin != 0:
        x_pos = np.nanmean(amp_mean_b1)
        print(depths[mean_susp_bin - 1:mean_bad_bin])
        y_pos = np.nanmean(depths[mean_susp_bin - 1:mean_bad_bin])
        ax.text(x=x_pos, y=y_pos,
                s='Flag=3', horizontalalignment='right', verticalalignment='center', fontsize=10)

    ax.set_ylim(depths[-1], depths[0])
    ax.legend(loc='lower left')
    ax.set_xlabel('Counts')
    ax.set_ylabel('Depth (m)')  # Set y axis label for this subplot only out of the 3
    ax.grid()
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.set_title('Mean Backscatter', fontweight='semibold')  # subplot title
    if d.orientation == 'up':
        plt.gca().invert_yaxis()

    fig.suptitle('{}-{} {} at {} m depth: L2 QC'.format(d.attrs['station'],
                                                        d.attrs['deployment_number'],
                                                        d.attrs['serial_number'],
                                                        d.attrs['instrument_depth']), fontweight='semibold')

    # Create L2_Python_plots subfolder if not already existing
    plot_dir = './L2_Python_plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig_name = plot_dir + '{}-{}_{}_nc_L2.png'.format(
        d.station.lower(), str(d.deployment_number), d.serial_number)
    fig.savefig(fig_name)
    plt.close()

    return


def plot_vel_by_bin_L2(d, which_vel=1):
    """
    Make plots of velocity per each bin
    Inputs:
        - d: dataset-type object created by reading in a netCDF ADCP file with the xarray package
        - which_vel: optional kwarg indicating which velocity to plot
                     1=LCEWAP01 (Eastward), 2=LCNSAP01 (Northward), 3=LRZAAP01 (upward),
                     5=LRZUVP01 (upward from vertical beam) (error velocity not plotted)
    Outputs:
        None
    """

    if which_vel == 1:
        vel_data = d.LCEWAP01.data
        vel_qc_data = d.LCEWAP01_QC.data
        vel_name = 'LCEWAP01'
    elif which_vel == 2:
        vel_data = d.LCNSAP01.data
        vel_qc_data = d.LCNSAP01_QC.data
        vel_name = 'LCNSAP01'
    elif which_vel == 3:
        vel_data = d.LRZAAP01.data
        vel_qc_data = d.LRZAAP01_QC.data
        vel_name = 'LRZAAP01'
    elif which_vel == 5:
        vel_data = d.LRZUVP01.data
        vel_qc_data = d.LRZUVP01_QC.data
        vel_name = 'LRZUVP01'
    else:
        ValueError('Invalid value for which_vel kwarg. Choose a value in [1, 2, 3].')

    # Create L2_Python_plots subfolder
    plot_dir = './L2_Python_plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig_name = plot_dir + '{}-{}_{}_{}_binplot_QC.pdf'.format(
        d.station.lower(), str(d.deployment_number), d.serial_number, vel_name)

    bins_per_pg = 4

    # Create pdf file of velocity plots for each bin
    with PdfPages(fig_name) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        for bin_no in range(d.numberOfCells):
            # Limit the number of subplots per each page to 5
            if bin_no != 0 and bin_no % bins_per_pg + 1 == 1:
                # Save the previous fig
                pdf.savefig()
                plt.close()
                fig = plt.figure(figsize=(8.5, 11))
            # Add subplot per bin number
            ax = fig.add_subplot(bins_per_pg, 1, bin_no % bins_per_pg + 1)
            # Plot bad data
            f1 = ax.plot(d.time.data[vel_qc_data[bin_no] == 4], vel_data[bin_no][vel_qc_data[bin_no] == 4],
                         color='r', label='Flag=4')
            # Plot suspicious data if instrument orientation is 'up'
            if d.orientation == 'up':
                f2 = ax.plot(d.time.data[vel_qc_data[bin_no] == 3], vel_data[bin_no][vel_qc_data[bin_no] == 3],
                             color='tab:orange', label='Flag=3')
            # Plot good data
            f3 = ax.plot(d.time.data[vel_qc_data[bin_no] == 1], vel_data[bin_no][vel_qc_data[bin_no] == 1],
                         color='k', label='Flag=1')

            ax.tick_params(axis='both', direction='in', right=True)
            ax.set_title('Bin {}'.format(bin_no + 1))  # +1 since Python indexes from zero

            if bin_no == 0:
                fig.suptitle('{}-{} {} at {} m depth: {} (m s-1)'.format(d.attrs['station'],
                                                                         d.attrs['deployment_number'],
                                                                         d.attrs['serial_number'],
                                                                         d.attrs['instrument_depth'],
                                                                         vel_name), fontweight='semibold')

            # Set the x axis label and legend
            if (bin_no + 1) % bins_per_pg == 0:
                ax.set_xlabel('Time (UTC)')
                ax.legend(loc='lower left')

            # Set x axis label and legend for very last bin if it wasn't a multiple of bins_per_pg
            if bin_no == d.numberOfCells - 1 and (bin_no + 1) % bins_per_pg != 0:
                ax.set_xlabel('Time (UTC)')
                ax.legend(loc='lower left')

        # Save and close the pdf
        pdf.savefig()
        plt.close()

    return


def bad_2_nan(d):
    """
    Apply flags to velocity data.
    Inputs:
        - d: netCDF ADCP file in xarray dataset format
    Outputs:
        None
    """

    print(d.LCEWAP01.data.shape)
    print(d.LCEWAP01_QC.data.shape)

    # Redo setting bad velocity data to nans based on updates from pressure or ocean floor depth analysis
    d.LCEWAP01.data[d.LCEWAP01_QC.data == 4] = np.nan
    d.LCNSAP01.data[d.LCNSAP01_QC.data == 4] = np.nan
    d.LRZAAP01.data[d.LRZAAP01_QC.data == 4] = np.nan

    flag_vb = vb_flag(d)
    if d.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        d.LRZUVP01.data[d.LRZUVP01_QC == 4] = np.nan

    d.attrs['processing_history'] = d.processing_history + " Velocity data that were flagged as bad_data were set" \
                                                           " to nans."

    # Reset velocity data_min and data_max variable attributes
    reset_vel_minmaxes(d=d)
    return


def reset_vel_minmaxes(d):
    """
    Function for re-calculating data_min and data_max variable attributes
    Inputs:
        - d: xarray object created by reading in a netCDF ADCP file with xarray.open_dataset()
    Outputs:
        None
    """

    flag_vb = vb_flag(d)
    if d.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
        var_list = [d.LCEWAP01, d.LCNSAP01, d.LRZAAP01, d.LRZUVP01, d.LERRAP01, d.LCEWAP01_QC, d.LCNSAP01_QC,
                    d.LRZAAP01_QC, d.LRZUVP01_QC]
    else:
        var_list = [d.LCEWAP01, d.LCNSAP01, d.LRZAAP01, d.LERRAP01, d.LCEWAP01_QC, d.LCNSAP01_QC,
                    d.LRZAAP01_QC]

    # Reset mins and maxes
    for var in var_list:
        var.attrs['data_max'] = np.nanmax(var.data)
        var.attrs['data_min'] = np.nanmin(var.data)
    return


def create_nc_L2(f_adcp, dest_dir, f_sbe=None):
    """
    Function for performing the suite of L2 processing methods on netCDF ADCP
    data.
    Inputs:
        - The name of a netCDF ADCP file
        - The name of a netCDF CTD file (required if ADCP file didn't have pressure sensor data
    Outputs:
        - The name of the output netCDF ADCP file
    """

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Set name for new output netCDF
    nc_out_name = os.path.basename(f_adcp).replace('L1', 'L2')
    print(nc_out_name)
    if not dest_dir.endswith('/') or not dest_dir.endswith('\\'):
        out_absolute_name = os.path.abspath(dest_dir + '/' + nc_out_name)
    else:
        out_absolute_name = os.path.abspath(dest_dir + nc_out_name)

    # Open netCDF ADCP file
    nc_adcp = xr.open_dataset(f_adcp)

    # Change value of filename variable to include "L2" instead of "L1"
    nc_adcp = nc_adcp.assign(filename=((), nc_out_name[:-3]))

    # Produce pre-processing plots
    plots_preprocess_L2(nc_adcp)

    if nc_adcp.orientation == 'up':
        # Identify bins through time series where their pressure is negative
        if np.nanmin(nc_adcp.PRESPR01.data) == np.nanmax(nc_adcp.PRESPR01.data):
            warnings.warn('Pressure was calculated from static instrument depth in L1', UserWarning)
            nc_sbe = xr.open_dataset(f_sbe)

            # Calculate ADCP pressure from CTD pressure
            nc_adcp = add_pressure_ctd(nc_adcp=nc_adcp, nc_ctd=nc_sbe, name_ctd=f_sbe)

            # Plot static and CTD-derived pressures
            plot_pres_compare(nc_adcp)

            # Flag bad bins by negative pressure values
            bad_bins = flag_by_pres(d=nc_adcp, use_prexmcat=True)
        else:
            # Flag bad bins by negative pressure values
            bad_bins = flag_by_pres(d=nc_adcp)

        # Identify bins through time series where their backscatter increases, get list of their indices
        susp_bins = flag_by_backsc(d=nc_adcp, bad_bin_list=bad_bins)

        # Make QC plots showing time-averaged negative pressure and backscatter increases
        plot_backscatter_qc(d=nc_adcp)
    else:
        # orientation == 'down'

        # Identify bins through time series that are below the depth of the ocean floor
        bad_bins = flag_below_seafloor(d=nc_adcp)

        # Make QC plots showing time-averaged flagged bins that are below ocean floor depth
        plot_backscatter_qc(d=nc_adcp)

    # Set bad velocity data to nans
    bad_2_nan(d=nc_adcp)

    # Export the dataset object as a new netCDF file
    nc_adcp.to_netcdf(out_absolute_name, mode='w', format='NETCDF4')
    print('Exported L2 netCDF file')

    nc_adcp.close()
    return out_absolute_name


def example_usage_L2():

    # Sample L1 netCDF ADCP file
    f_adcp = './newnc/a1_20050503_20050504_0221m.adcp.L1.nc'
    f_sbe = None

    create_nc_L2(f_adcp)

    return
