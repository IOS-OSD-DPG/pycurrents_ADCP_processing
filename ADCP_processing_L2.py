"""
author: Hana Hourston

Credits:
Plotting code adapted from David Spear (originally in MatLab)
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
import datetime

# Computes principal component direction of u and v
def fmamidir(u, v):
    # Calculate major and minor axes
    ub = np.nanmean(u)
    vb = np.nanmean(v)
    uu = np.nanmean(u**2)
    vv = np.nanmean(v**2)
    uv = np.nanmean(u * v)
    uu = uu - (ub * ub)
    vv = vv - (vb * vb)
    uv = uv - (ub * vb)

    # Solve for the quadratic
    a = 1.0
    b = -(uu + vv)
    c = (uu * vv) - (uv * uv)
    s1 = (-b + np.sqrt((b * b) - (4.0 * a * c))/(2.0 * a))
    s2 = (-b - np.sqrt((b * b) - (4.0 * a * c))/(2.0 * a))
    major = s1
    minor = s2
    if minor > major:
        major = s2
        minor = s1

    return major, minor


def fpcdir(x, y):
    if x.shape != y.shape:
        print('Error: u and v are different sizes!')
        return
    else:
        # Compute major and minor axes
        major, minor = fmamidir(x, y)

        # Compute principal component direction
        u = x
        v = y
        ub = np.nanmean(u)
        vb = np.nanmean(v)
        uu = np.nanmean(u**2)
        uv = np.nanmean(u * v)
        uu = uu - (ub * ub)
        uv = uv - (ub * vb)

        e1 = -uv/(uu - major)
        e2 = 1
        rad_deg = 180/np.pi #conversion factor
        theta = np.arctan2(e1, e2) * rad_deg
        theta = -theta #change rotation angle to be CCW from North

    return theta


def add_attrs_PREXMCAT(dataset):
    fillValue = 1e15
    var = dataset.PREXMCAT
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'decibars'
    var.attrs['_FillValue'] = dataset.PRESPR01._FillValue
    var.attrs['long_name'] = 'pressure calculated from CTD'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = dataset.PRESPR01.sensor_depth
    var.attrs['serial_number'] = dataset.serial_number
    var.attrs['flag_meanings'] = dataset.PRESPR01.flag_meanings
    var.attrs['flag_values'] = dataset.PRESPR01.flag_values
    var.attrs['References'] = dataset.PRESPR01.flag_references
    #var.attrs['legency_GF3_code'] = 'SDN:GF3::PRES'
    #var.attrs['sdn_parameter_name'] = 'Pressure (spatial co-ordinate) exerted by the water body by profiling ' \
    #                                  'pressure sensor and corrected to read zero at sea level'
    #var.attrs['sdn_parameter_name'] = 'Pressure (measured variable) exerted by the water body by semi-fixed ' \
    #                                  'moored SBE MicroCAT (BODC P01)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UPDB'
    var.attrs['sdn_uom_name'] = 'Decibars'
    #ar.attrs['standard_name'] = 'sea_water_pressure'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    return


def add_pressure_SBE(nc_adcp, file_sbe):
    # Function for calculating ADCP pressure from SBE (Sea-Bird CTD) pressure on the same mooring line
    # Requires netCDF-format CTD file
    nc_sbe = xr.open_dataset(file_sbe)

    st_sbe, et_sbe = [nc_sbe.time.data[0], nc_sbe.time.data[-1]]
    st_adcp, et_adcp = [nc_adcp.time.data[0], nc_adcp.time.data[-1]]

    print("ADCP start-time, end-time:", st_adcp, et_adcp, sep=" ")
    print("SBE start-time, end-time:", st_sbe, et_sbe, sep=" ")

    st_adcp_index = np.where(nc_sbe.time.data == st_adcp)
    et_adcp_index = np.where(nc_sbe.time.data == et_adcp)

    # Calculate depth difference between ADCP and SBE
    # Positive if ADCP deeper than SBE
    depth_diff = nc_adcp.instrument_depth - nc_sbe.instrument_depth

    if not st_adcp_index[0].size or not et_adcp_index[0].size:
        # Test if differences are 1 sec long
        for t_i in range(len(nc_sbe.time.data)):
            if np.abs(nc_sbe.time.data[t_i] - nc_adcp.time.data[0]) <= np.timedelta64(1800000000000, 'ns'):
                st_adcp_index = np.array([t_i], dtype='int64')  # same format as above
                break
        for t_j in range(len(nc_adcp.time.data)):
            if np.abs(nc_sbe.time.data[t_j] - nc_adcp.time.data[-1]) <= np.timedelta64(1800000000000, 'ns'):
                et_adcp_index = np.array([t_j], dtype='int64')
                break
        # Re-check st_adcp_index and et_adcp_index
        if not st_adcp_index[0].size:
            warnings.warn("ADCP start-time value not present in SBE time variable", UserWarning)
            # SBE starts before ADCP
            if st_sbe < st_adcp:
                warnings.warn("SBE start-time earlier than ADCP start-time", UserWarning)
            else:
                warnings.warn("SBE start-time later than ADCP start-time", UserWarning)
                # Test if difference off by a second -- in that case ignore
                for t_i in range(len(nc_adcp.time.data)):
                    # If difference equal to or less than 1 sec
                    if np.abs(nc_sbe.time.data[0] - nc_adcp.time.data[t_i]) <= np.timedelta64(1800000000000, 'ns'):
                        st_adcp_index = t_i
                        break
        if not et_adcp_index[0].size:
            warnings.warn("ADCP end-time value not present in SBE time variable", UserWarning)
            # Test if difference off by a second -- in that case ignore
            if et_sbe > et_adcp:
                warnings.warn("SBE end-time later than ADCP end-time", UserWarning)
            else:
                warnings.warn("SBE end-time earlier than ADCP end-time", UserWarning)
                # Needs resolving
    else:
        # Test if sampling interval is the same
        # If SBE sampling interval is more frequent, only take every nth pressure measurement
        # If ADCP sampling interval is more frequent, can't extrapolate
        if nc_adcp.time.data[1]-nc_adcp.time.data[0] < nc_sbe.time.data[1]-nc_sbe.time.data[0]:
            pass
        elif nc_adcp.time.data[1]-nc_adcp.time.data[0] > nc_sbe.time.data[1]-nc_sbe.time.data[0]:
            pass
        else:
            nc_adcp_new = nc_adcp.assign(PREXMCAT=(("time"), nc_sbe.PRESPR01.data[
                                                             st_adcp_index[0][0]:et_adcp_index[0][0]] + depth_diff))

            # Add attributes to PREXMCAT:
            add_attrs_PREXMCAT(nc_adcp_new.PREXMCAT)

    return


def preprocess_plots_L2(d):
    # Preliminary plots

    # Subplot 1/3: Plot avg backscatter against depth

    # First calculate depths
    # depths = np.mean(ncdata.PRESPR01[0,:]) - ncdata.distance  #What Di used
    if d.orientation == 'up':
        depths = float(d.instrument_depth) - d.distance.data
    else:
        depths = float(d.instrument_depth) + d.distance.data

    # Calculate average backscatter (amplitude intensity)
    amp_mean_b1 = np.zeros(int(d.numberOfCells), dtype='float32')
    amp_mean_b2 = np.zeros(int(d.numberOfCells), dtype='float32')
    amp_mean_b3 = np.zeros(int(d.numberOfCells), dtype='float32')
    amp_mean_b4 = np.zeros(int(d.numberOfCells), dtype='float32')

    for i in range(len(amp_mean_b1)):
        amp_mean_b1[i] = np.nanmean(d.TNIHCE01.data[i, :])
        amp_mean_b2[i] = np.nanmean(d.TNIHCE02.data[i, :])
        amp_mean_b3[i] = np.nanmean(d.TNIHCE03.data[i, :])
        amp_mean_b4[i] = np.nanmean(d.TNIHCE04.data[i, :])

    amp = [amp_mean_b1, amp_mean_b2, amp_mean_b3, amp_mean_b4]

    # Make plot and first subplot
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(1, 3, 1)
    beam_no = 1
    colours = ['b', 'g', 'c', 'm']
    for dat, col in zip(amp, colours):
        f1 = ax.plot(dat, depths, label='Beam {}'.format(beam_no), linewidth=1, marker='o', markersize=2, color=col)
        beam_no += 1
    ax.set_ylim(depths[-1], depths[0])
    ax.legend(loc='lower left')
    ax.set_xlabel('Counts')
    ax.set_ylabel('Depth (m)') #Set y axis label for this subplot only out of the 3
    ax.grid()
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.set_title('Mean Backscatter', fontweight='semibold') #subplot title
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

    names = ['LCEWAP01', 'LCNSAP01', 'LRZAAP01']
    vels = [u_mean, v_mean, w_mean]
    for n, dat in zip(names, vels):
        f2 = ax.plot(dat, depths, label=n, linewidth=1, marker='o', markersize=2)
    ax.set_ylim(depths[-1], depths[0]) #set vertical limits
    ax.legend(loc='lower left')
    ax.set_xlabel('Velocity (m/sec)') #Change to cm/sec?
    ax.grid()
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.set_title('Mean Velocity', fontweight='semibold') #subplot title
    # Flip over x-axis if instrument oriented 'up'
    if d.orientation == 'up':
        plt.gca().invert_yaxis()

    # Subplot 3/3: Principal axis

    # Orientation data: pitch ( PTCHGP01), roll (ROLLGP01), heading (HEADCM01)?
    #orientation = -d.HEADCM01.data[0] #convert to clockwise?
    orientation = np.zeros(int(d.numberOfCells), dtype='float32')
    for ibin in range(len(orientation)):
        xx = d.LCEWAP01[ibin, :]
        yy = d.LCNSAP01[ibin, :]
        orientation[ibin] = fpcdir(xx, yy) #convert to CW

    mean_orientation = np.round(np.nanmean(orientation), decimals=1)
    mean_depth = np.nanmean(depths)

    # Make subplot
    ax = fig.add_subplot(1, 3, 3)
    f3 = ax.plot(orientation, depths, linewidth=1, marker='o', markersize=2)
    ax.set_ylim(depths[-1], depths[0]) #set vertical limits
    ax.set_xlabel('Orientation') #Degree symbol
    ax.text(x=mean_orientation, y=mean_depth, s='Mean orientation = ' + str(mean_orientation) + '$^\circ$',
             horizontalalignment='center', verticalalignment='center', fontsize=10)
    ax.grid() #set grid
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.set_title('Principal Axis', fontweight='semibold') #subplot title
    # Flip over x-axis if instrument oriented 'up'
    if d.orientation == 'up':
        plt.gca().invert_yaxis()

    # Create centred figure title
    fig.suptitle('{}-{} {} at {} m depth'.format(d.station, d.deployment_number, d.serial_number,
                                                 d.instrument_depth), fontweight='semibold')

    fig_name = './{}-{}_{}_nc_preprocess.png'.format(
        d.station.lower(), str(d.deployment_number), d.serial_number)
    fig.savefig(fig_name)

    return


def flag_by_pres(d):
    # Flagging: BODC SeaDataNet standards

    # For upward-facing ADCPs, flag backscatter increases near surface
    # For downward-facing ADCPs, flag backscatter increases near ocean floor???

    # 1/2 Pressure or ocean floor:
    # bin20_pres = np.mean(d.PRESPR01[0,:] - d.distance[19]) #from Di's code

    # List of indices where bins start to have negative
    bad_bin_list = np.zeros(shape=d.PRESPR01.data.shape, dtype='int32')
    bad_bin_1 = 0
    # Ignore pressure values that were calculated from static instrument depth
    if d.PRESPR01.attrs['data_min'] != d.PRESPR01.attrs['data_max']:
        if d.orientation == 'up':
            # Flag by negative pressure timestep by timestep
            for t_i in range(len(d.time.data)):
                flag = 0
                # Iterate through bins
                for b_i in range(len(d.distance.data)):  # bin with index 0 == 1st bin
                    # Subtract each bin distance from pressure to determine pressure at each bin depth
                    bin_pres = d.PRESPR01.data[t_i] - d.distance.data[b_i]
                    if bin_pres < 0:
                        # Identify first bin index with negative pressure (index = bin number - 1)
                        bad_bin_1 = b_i
                        # Add bad bin number for this timestep to the array containing bad bins
                        bad_bin_list[t_i] = bad_bin_1
                        flag += 1
                        # Flag velocity data
                        for v_QC in [d.LCEWAP01_QC, d.LCNSAP01_QC, d.LCNSAP01_QC]:
                            v_QC[bad_bin_1:, t_i] = 4 #bad_value flag=4
                    if flag != 0:
                        break
        else:
            # d.orientation == 'down'
            warnings.warn("Pressure values were calculated from static instrument depth in L1", UserWarning)
            # Flag by depth when bin depth > ocean depth
            # Timestep by timestep to account for potential stretch in mooring line over the time series
            for t_i in range(len(d.time.data)):
                flag = 0
                # Find shallowest bin that is within the ocean floor and flag it and all deeper ones == 4
                for b_i in range(len(d.distance.data)):
                    bin_depth = d.instrument_depth + d.distance.data[b_i]
                    if bin_depth > d.water_depth:
                        bad_bin_1 = b_i
                        bad_bin_list[t_i] = bad_bin_1
                        flag += 1
                        # Update velocity QC variables from pressure
                        for v_QC in [d.LCEWAP01_QC, d.LCNSAP01_QC, d.LRZAAP01_QC]:
                            v_QC[bad_bin_1:, t_i] = 4
                    if flag != 0:
                        break
    else:
        # import SBE pressure since pressure sensor missing, and pressure was calculated from static depth
        flag = 0

    d.attrs['processing_history'] = d.processing_history + " Level 2 processing was performed on the data."
    d.attrs['processing_history'] = d.processing_history + " Bins where pressure was negative were flagged as" \
                                                           " bad_value timestep by timestep. An average of the" \
                                                           " farthest {} bins were flagged as bad_value.".format(
        str(len(d.numberOfCells[int(np.round(np.mean(bad_bin_list))):])))

    return bad_bin_list


def flag_by_backsc(d, bad_bin_list):
    # 2/2 Backscatter
    # Note: For files with orientation == 'up' ONLY

    # First take beam-averaged backscatter
    amp_beam_avg = np.zeros(d.TNIHCE01.data.shape, dtype='float32')
    for b_i in range(len(d.TNIHCE01.data)):
        for t_i in range(len(d.TNIHCE01.data[b_i])):
            # Take mean
            amp_beam_avg[b_i, t_i] = np.mean([d.TNIHCE01.data[b_i, t_i], d.TNIHCE02.data[b_i, t_i],
                                              d.TNIHCE03.data[b_i, t_i], d.TNIHCE04.data[b_i, t_i]])

    # Then determine bin where beam-averaged backscatter increases for each timestep
    susp_bin_list = np.zeros(shape=d.PRESPR01.data.shape, dtype='int32')

    # Iterate through time stamps
    for t_i in range(len(d.time.data)):
        flag = 0
        # Iterate through bin numbers
        for b_i in range(1, len(amp_beam_avg[:, t_i])):
            if amp_beam_avg[b_i, t_i] > amp_beam_avg[b_i-1, t_i]:
                susp_bin_1 = b_i
                susp_bin_list[t_i] = susp_bin_1
                flag += 1
                for v_QC in [d.LCEWAP01_QC, d.LCNSAP01_QC, d.LCNSAP01_QC]:
                    v_QC[susp_bin_1:bad_bin_list[t_i], ] = 3  #probably_bad_value
                    v_QC[:susp_bin_1, ] = 2  #probably_good_value flag=2, or good_value flag=1
            if flag != 0:
                break

    d.attrs['processing_history'] = d.processing_history + " Bins where beam-averaged backscatter increases were" \
                                                           " flagged as probably_bad_value timestep by timestep." \
                                                           " An average of the next {} farthest bins where beam-" \
                                                           "averaged backscatter increases were flagged as " \
                                                           "probably_bad_value.".format(
        str(len(d.numberOfCells[int(np.round(np.mean(susp_bin_list))):int(np.round(np.mean(bad_bin_list)))])))

    d.attrs['processing_history'] = d.processing_history + "An average of {} remaining bins were flagged as " \
                                                           "probably_good_value.".format(
        str(len(d.numberOfCells[:int(np.round(np.mean(susp_bin_list)))])))

    return susp_bin_list


def plot_L2_QC(d, bad_bin_list, susp_bin_list=None):
    # Calculate depths from bin distances from instrument
    if d.orientation == 'up':
        depths = float(d.instrument_depth) - d.distance.data
    else:
        depths = float(d.instrument_depth) + d.distance.data

    # Calculate average backscatter (amplitude intensity)
    amp_mean_b1 = np.zeros(int(d.numberOfCells), dtype='float32')
    amp_mean_b2 = np.zeros(int(d.numberOfCells), dtype='float32')
    amp_mean_b3 = np.zeros(int(d.numberOfCells), dtype='float32')
    amp_mean_b4 = np.zeros(int(d.numberOfCells), dtype='float32')

    for i in range(len(amp_mean_b1)):
        amp_mean_b1[i] = np.nanmean(d.TNIHCE01.data[i, :])
        amp_mean_b2[i] = np.nanmean(d.TNIHCE02.data[i, :])
        amp_mean_b3[i] = np.nanmean(d.TNIHCE03.data[i, :])
        amp_mean_b4[i] = np.nanmean(d.TNIHCE04.data[i, :])

    amp = [amp_mean_b1, amp_mean_b2, amp_mean_b3, amp_mean_b4]

    # Colours for plotting
    colours = ['b', 'g', 'c', 'm']

    # Calculate average bad_bin_1 and susp_bin_1 for plotting
    avg_bad_bin_1 = np.mean(bad_bin_list)
    if susp_bin_list is not None:
        avg_susp_bin_1 = np.mean(susp_bin_list)

    # Plot flagged average backscatter (QC):
    # Plot Good, Suspicious, and Bad data separately on the same plot
    # Plot suspicious (flag = 3) and good (flag = 0) backscatter data
    # Plot bad (flag = 4), suspicious (flag = 3) and good (flag = 0) mean velocity data

    # Start the plot
    # Make plot and first subplot

    # 1/2 Backscatter
    fig = plt.figure(figsize=(5, 8.5))  # figsize=(11, 8.5), (8, 8.5) for two subplots
    ax = fig.add_subplot(1, 2, 1)  # (nrows, ncols, index)
    beam_no = 1
    for dat, col in zip(amp, colours):
        f11 = ax.plot(dat[avg_bad_bin_1-1:], depths[avg_bad_bin_1-1:], linewidth=1, marker='o', markersize=2,
                      color='r')
        if susp_bin_list is not None:
            f12 = ax.plot(dat[avg_susp_bin_1-1:avg_bad_bin_1], depths[avg_susp_bin_1-1:avg_bad_bin_1], linewidth=1,
                      marker='o', markersize=2, color='tab:orange')
            f13 = ax.plot(dat[:avg_susp_bin_1], depths[:avg_susp_bin_1], label='Beam {}'.format(beam_no), linewidth=1,
                          marker='o', markersize=2, color=col)
        else:
            f12 = ax.plot(dat[:avg_bad_bin_1], depths[:avg_bad_bin_1], label='Beam {}'.format(beam_no), linewidth=1,
                          marker='o', markersize=2, color=col)
        beam_no += 1

    # Add flag plot text
    ax.text(x=np.max(amp_mean_b1), y=-2, s='Flag=4',
             horizontalalignment='right', verticalalignment='center', fontsize=10)
    if susp_bin_list is not None:
        ax.text(x=np.max(amp_mean_b1), y=depths[int(np.floor(np.mean([(avg_susp_bin_1-1), avg_bad_bin_1])))],
                s='Flag=3', horizontalalignment='right', verticalalignment='center', fontsize=10)

    ax.set_ylim(depths[-1], depths[0])
    ax.legend(loc='lower left')
    ax.set_xlabel('Counts')
    ax.set_ylabel('Depth (m)') #Set y axis label for this subplot only out of the 3
    ax.grid()
    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.set_title('Mean Backscatter', fontweight='semibold') #subplot title
    # Flip over x-axis if instrument oriented 'up'
    if d.orientation == 'up':
        plt.gca().invert_yaxis()

    fig.suptitle('{}-{} {} at {} m depth: L2 QC'.format(d.attrs['station'],
                                                        d.attrs['deployment_number'],
                                                        d.attrs['serial_number'],
                                                        d.attrs['instrument_depth']), fontweight='semibold')

    """
    # 2/2 Velocity
    # Separate each variable based on flags
    v_QC = d.LCEWAP01_QC
    u_mean_bad = u_mean[v_QC[:len(v_QC), 0] == 4] #[bin, ensemble] #########################3
    u_mean_susp = u_mean[v_QC[:len(v_QC), 0] == 3]
    u_mean_good = u_mean[v_QC[:len(v_QC), 0] == 0]
    
    """

    fig_name = './{}-{}_{}_nc_L2.png'.format(
        d.station.lower(), str(d.deployment_number), d.serial_number)
    fig.savefig(fig_name)

    return


def bad_2_nan(d):
    # QC data

    # Redo setting velocity data to nans based on updates from pressure analysis
    d.LCEWAP01[d.LCEWAP01_QC == 4] = np.nan
    d.LCNSAP01[d.LCNSAP01_QC == 4] = np.nan
    d.LRZAAP01[d.LCNSAP01_QC == 4] = np.nan

    d.attrs['processing_history'] = d.processing_history + " Velocity data flagged as bad_data were set to nans."

    return


def main_L2(wd, f):
    # Set working directory
    os.chdir(wd)

    # Set name for new output netCDF
    nc_out_name = os.path.basename(f).replace('L1', 'L2')
    print(nc_out_name)

    # Open netCDF ADCP file
    ncdata = xr.open_dataset(f)

    preprocess_plots_L2(ncdata)

    bad_bins = flag_by_pres(d=ncdata)

    if ncdata.orientation == 'up':
        susp_bins = flag_by_backsc(d=ncdata, bad_bin_list=bad_bins)

    plot_L2_QC(d=ncdata, bad_bin_list=bad_bins, susp_bin_list=susp_bins)

    bad_2_nan(d=ncdata)

    ncdata.to_netcdf(nc_out_name, mode='w', format='NETCDF4')
    ncdata.close()

    return


# my_wd = '/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/ADCP_L2/'
# my_file = '/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/ADCP_L1_geo/muc1_20160712_20170714_0041m.adcp.L1.nc'
#
# main_L2(my_wd, my_file)


##### TESTING #####

my_wd = '/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/ADCP_L2/'
# Downwards-facing, pressure calculated from static instrument depth
# my_file = '/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/ADCP_L1_geo/a1_20050503_20050504_0221m.adcp.L1.nc'

# Upwards-facing, pressure sensor available
my_file = '/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/ADCP_L1_geo/scott1_20170706_20170711_0101m.adcp.L1.nc'

# Set working directory
os.chdir(my_wd)

# Set name for new output netCDF
nc_out_name = os.path.basename(my_file).replace('L1', 'L2')
print(nc_out_name)

# Open netCDF ADCP file
ncdata = xr.open_dataset(my_file)

preprocess_plots_L2(ncdata)

bad_bins = flag_by_pres(d=ncdata)

susp_bins = flag_by_backsc(d=ncdata, bad_bin_list=bad_bins)

plot_L2_QC(d=ncdata, bad_bin_list=bad_bins, susp_bin_list=susp_bins)  # orientation up

plot_L2_QC(d=ncdata, bad_bin_list=bad_bins, susp_bin_list=None)  # orientation down

bad_2_nan(d=ncdata)

ncdata.to_netcdf(nc_out_name, mode='w', format='NETCDF4')
ncdata.close()
