"""
author: Hana Hourston
date: Jan. 23, 2020

about: This script is adapted from Jody Klymak's at https://gist.github.com/jklymak/b39172bd0f7d008c81e32bf0a72e2f09
for L1 processing raw ADCP data.

Contributions from: Di Wan, Eric Firing

NOTES:
# If your raw file came from a NarrowBand instrument, you must also use the create_nc_L1() start_year optional kwarg
  (int type)
# If your raw file has time values out of range, you must also use the create_nc_L1() time_file optional kwarg
# Use the time_file kwarg to read in a csv file containing time entries spanning the range of deployment and using the
  instrument sampling interval

"""

import os
import csv
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import warnings
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pycurrents.adcp.rdiraw import rawfile
from pycurrents.adcp.rdiraw import SysCfg
import pycurrents.adcp.transform as transform
import gsw
import pycurrents_ADCP_processing.add_var2nc as add_var2nc


def mean_orientation(o):
    # orientation, o, is an array
    up = 0
    down = 0
    for i in range(len(o)):
        if o[i]:
            up += 1
        else:
            down += 1
    if up > down:
        return 'up'
    elif down > up:
        return 'down'
    else:
        ValueError('Number of \"up\" orientations equals number of \"down\" orientations in data subset')


def correct_true_north(measured_east, measured_north, metadata_dict):  # change angle to negative of itself
    # Di Wan's magnetic declination correction code: Takes 0 DEG from E-W axis
    # mag_decl: magnetic declination value; float type
    # measured_east: measured Eastward velocity data; array type
    # measured_north: measured Northward velocity data; array type
    angle_rad = -metadata_dict['magnetic_variation'] * np.pi / 180.
    east_true = measured_east * np.cos(angle_rad) - measured_north * np.sin(angle_rad)
    north_true = measured_east * np.sin(angle_rad) + measured_north * np.cos(angle_rad)
    
    metadata_dict['processing_history'] += " Magnetic variation, using average applied; " \
                                           "declination = {}.".format(str(metadata_dict['magnetic_variation']))
    
    return east_true, north_true


def convert_time_var(time_var, number_of_profiles, metadata_dict, origin_year, time_csv):
    # Includes exception handling for bad times
    # time_var: vel.dday; time variable with units in days since the beginning of the year in which measurements 
    #           started being taken by the instrument
    # number_of_profiles: the number of profiles (ensembles) recorded by the instrument over the time series
    # metadata_dict: dictionary object of metadata items
    
    # data.yearbase is an integer of the year that the timeseries starts (e.g., 2016)
    data_origin = pd.Timestamp(str(origin_year) + '-01-01')  # convert to date object

    try:
        # convert time variable to elapsed time since 1970-01-01T00:00:00Z
        t_s = np.array(
            pd.to_datetime(time_var, unit='D', origin=data_origin, utc=True).strftime('%Y-%m-%d %H:%M:%S'),
            dtype='datetime64[s]')
        # DTUT8601 variable: time strings
        t_DTUT8601 = pd.to_datetime(time_var, unit='D', origin=data_origin, utc=True).strftime(
            '%Y-%m-%d %H:%M:%S')  # don't need %Z in strftime
    except OutOfBoundsDatetime or OverflowError:
        print('Using user-created time range')
        t_s = np.zeros(shape=number_of_profiles, dtype='datetime64[s]')
        t_DTUT8601 = np.empty(shape=number_of_profiles, dtype='<U100')
        with open(time_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # Skip headers
            next(csv_reader, None)
            for count, row in enumerate(csv_reader):
                if row[0] == '':
                    pass
                else:
                    t_s[count] = np.datetime64(pd.to_datetime(row[0], utc=True).strftime(
                        '%Y-%m-%d %H:%M:%S'))
                    t_DTUT8601[count] = pd.to_datetime(row[0], utc=True).strftime('%Y-%m-%d %H:%M:%S')

        metadata_dict['processing_history'] += ' OutOfBoundsDateTime exception triggered; used user-generated time range as time data.'
    
    return t_s, t_DTUT8601


def assign_pres(vel_var, metadata_dict):
    # Assign pressure and calculate it froms static instrument depth if ADCP missing pressure sensor
    # vel_var: "vel" variable created from BBWHOS class object, using the command: data.read(varlist=['vel'])
    # metadata_dict: dictionary object of metadata items

    if metadata_dict['model'] == 'wh' or metadata_dict['model'] == 'os' or metadata_dict['model'] == 'sv':
        pres = np.array(vel_var.VL['Pressure'] / 1000, dtype='float32')  # convert decapascal to decibars

        # Calculate pressure based on static instrument depth if missing pressure sensor; extra condition added for zero pressure or weird pressure values
        # Handle no unique modes from statistics.mode(pressure)
        pressure_unique, counts = np.unique(pres, return_counts=True)
        index_of_zero = np.where(pressure_unique == 0)
        print('np.max(counts):', np.max(counts), sep=' ')
        print('counts[index_of_zero]:', counts[index_of_zero], sep=' ')
        print('serial number:', metadata_dict['serialNumber'])

    # Check if model is type missing pressure sensor or if zero is a mode of pressure
    if metadata_dict['model'] == 'bb' or metadata_dict['model'] == 'nb' or np.max(counts) == counts[index_of_zero]:
        p = np.round(gsw.conversions.p_from_z(-metadata_dict['instrument_depth'], metadata_dict['latitude']),
                     decimals=0)  # depth negative because positive is up for this function
        pres = np.repeat(p, len(vel_var.vel1.data))
        metadata_dict['processing_history'] += " Pressure values calculated from static instrument depth ({} m) using " \
                                               "the TEOS-10 75-term expression for specific volume and rounded to {} " \
                                               "significant digits.".format(str(metadata_dict['instrument_depth']), str(len(str(p))))
        warnings.warn('Pressure values calculated from static instrument depth', UserWarning)
    
    return pres


def check_depths(pres, dist, instr_depth, water_depth):
    # Check user-entered instrument_depth and compare with pressure values
    # pres: pressure variable; array type
    # dist: distance variable (contains distance of each bin from ADCP); array type
    # instr_depth: depth of the instrument
    # water_depth: depth of the water

    depths_check = np.mean(pres[:]) - dist
    inst_depth_check = depths_check[0] + dist[0]
    abs_difference = np.absolute(inst_depth_check-instr_depth)
    # Calculate percent difference in relation to total water depth
    if (abs_difference / water_depth * 100) > 0.05:
        warnings.warn("Difference between calculated instrument depth and metadata instrument_depth " \
                      "exceeds 0.05% of the total water depth", UserWarning)
    return


def coordsystem_2enu(vel_var, fixed_leader_var, metadata_dict):
    # Transforms beam and xyz coordinates to enu coordinates
    # vel_var: "vel" variable created from BBWHOS class object, using the command: data.read(varlist=['vel'])
    # fixed_leader_var: "fixed_leader" variable created from BBWHOS class object, using the command: data.read(varlist=['FixedLeader'])
    # metadata_dict: dictionary object of metadata items
    # UHDAS transform functions use a three-beam solution by faking a fourth beam

    if vel_var.trans.coordsystem == 'beam':
        trans = transform.Transform(angle=fixed_leader_var.sysconfig['angle'],
                                    geometry=metadata_dict['beam_pattern'])  #angle is beam angle
        xyze = trans.beam_to_xyz(vel_var.vel.data)
        print(np.shape(xyze))
        enu = transform.rdi_xyz_enu(xyze, vel_var.heading, vel_var.pitch, vel_var.roll,
                                    orientation=metadata_dict['orientation'])
    elif vel_var.trans.coordsystem == 'xyz':
        print(np.shape(vel_var.vel.data))
        enu = transform.rdi_xyz_enu(vel_var.vel.data, vel_var.heading, vel_var.pitch, vel_var.roll,
                                    orientation=metadata_dict['orientation'])
        print(np.shape(enu))
    else:
        ValueError('vel.trans.coordsystem value of {} not recognized. Conversion to enu not available.'.format(
            vel_var.trans.coordsystem))

    print(np.shape(enu))
    # Apply change in coordinates to velocities
    velocity1 = enu[:, :, 0]
    velocity2 = enu[:, :, 1]
    velocity3 = enu[:, :, 2]
    velocity4 = enu[:, :, 3]

    # Round each velocity to 3 decimal places to match the original data
    velocity1 = np.round(velocity1, decimals=3)
    velocity2 = np.round(velocity2, decimals=3)
    velocity3 = np.round(velocity3, decimals=3)
    velocity4 = np.round(velocity4, decimals=3)
    # Make note in processing_history
    metadata_dict['processing_history'] += " The coordinate system was rotated from {} to enu " \
                                           "coordinates.".format(vel_var.trans.coordsystem)
    print('Coordinate system rotated from {} to enu'.format(vel_var.trans.coordsystem))
    vel_var.trans.coordsystem = 'enu'
    metadata_dict['coord_system'] = 'enu'  # Add item to metadata dictionary for coordinate system
    
    return velocity1, velocity2, velocity3, velocity4


def flag_pressure(pres, ens1, ens2, metadata_dict):
    # pres: pressure variable; array type
    # ens1: number of leading bad ensembles from before instrument deployment; int type
    # ens2: number of trailing bad ensembles from after instrument deployment; int type
    # metadata_dict: dictionary object of metadata items

    PRESPR01_QC_var = np.zeros(shape=pres.shape, dtype='float32')
    # 2/2 pressure
    PRESPR01_QC_var[:ens1] = 4
    if ens2 != 0:
        PRESPR01_QC_var[-ens2:] = 4

    # Flag negative pressure values
    PRESPR01_QC_var[pres < 0] = 4

    pres[PRESPR01_QC_var == 4] = np.nan

    metadata_dict['processing_history'] += " Quality control flags set based on SeaDataNet flag scheme from BODC."
    metadata_dict['processing_history'] += " Negative pressure values flagged as \"bad_data\" and set to nan\'s."

    return PRESPR01_QC_var


def flag_velocity(ens1, ens2, number_of_cells, v1, v2, v3, v5=None):
    # Create QC variables containing flag arrays
    # ens1: number of leading bad ensembles from before instrument deployment; int type
    # ens2: number of trailing bad ensembles from after instrument deployment; int type
    # number_of_cells: number of bins
    # v1: Eastward velocity with magnetic declination applied
    # v2: Northward velocity with magnetic declination applied
    # v3: Upwards velocity
    # v5: Upwards velocity from Sentinel V vertical beam; only for Sentinel V instruments

    LCEWAP01_QC_var = np.zeros(shape=v1.shape, dtype='float32')
    LCNSAP01_QC_var = np.zeros(shape=v2.shape, dtype='float32')
    LRZAAP01_QC_var = np.zeros(shape=v3.shape, dtype='float32')

    for qc in [LCEWAP01_QC_var, LCNSAP01_QC_var, LRZAAP01_QC_var]:
        # 0=no_quality_control, 4=value_seems_erroneous
        for bin_num in range(number_of_cells):
            qc[:ens1, bin_num] = 4
            if ens2 != 0:
                qc[-ens2:, bin_num] = 4  # if ens2==0, the slice [-0:] would index the whole array

    # Apply the flags to the data and set bad data to NAs
    print(v1.shape, LCEWAP01_QC_var.shape)
    print(v1)
    print(LCEWAP01_QC_var)
    v1[LCEWAP01_QC_var == 4] = np.nan
    v2[LCNSAP01_QC_var == 4] = np.nan
    v3[LRZAAP01_QC_var == 4] = np.nan
    print('Set u, v, w to nans')
    # Vertical beam velocity flagging for Sentinel V's
    if v5 is None:
        return LCEWAP01_QC_var, LCNSAP01_QC_var, LRZAAP01_QC_var
    else:
        LRZUVP01_QC_var = np.zeros(shape=v5.shape, dtype='float32')
        for bin_num in range(number_of_cells):
            LRZUVP01_QC_var[:ens1, bin_num] = 4
            if ens2 != 0:
                LRZUVP01_QC_var[-ens2:, bin_num] = 4

        return LCEWAP01_QC_var, LCNSAP01_QC_var, LRZAAP01_QC_var, LRZUVP01_QC_var


def add_attrs_2vars_L1(out_obj, metadata_dict, sensor_depth, cell_size, fillValue, pg_flag=0, vb_flag=0, vb_pg_flag=0):
    # out_obj: dataset object produced using the xarray package that will be exported as a netCDF file
    # metadata_dict: dictionary object of metadata items
    # sensor_depth: sensor depth recorded by instrument

    uvw_vel_min = -1000
    uvw_vel_max = 1000

    # Time
    var = out_obj.time
    var.encoding['units'] = "seconds since 1970-01-01T00:00:00Z"
    var.encoding['_FillValue'] = None
    var.attrs['long_name'] = "time"
    var.attrs['cf_role'] = "profile_id"
    var.encoding['calendar'] = "gregorian"
    
    # Bin distances
    var = out_obj.distance
    var.encoding['_FillValue'] = None
    var.attrs['units'] = "m"
    var.attrs['positive'] = 'up' if metadata_dict['orientation'] == 'up' else 'down'
    # var.attrs['long_name'] = "distance"
    var.attrs['long_name'] = "bin_distances_from_ADCP_transducer_along_measurement_axis"
    
    # LCEWAP01: eastward velocity (vel1)
    # all velocities have many of the same attribute values, but not all, so each velocity is done separately
    var = out_obj.LCEWAP01   
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'eastward_sea_water_velocity'
    var.attrs['ancillary_variables'] = 'LCEWAP01_QC'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'u'
    var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset'
    var.attrs['flag_meanings'] = metadata_dict['flag_meaning']
    var.attrs['flag_values'] = metadata_dict['flag_values']
    var.attrs['References'] = metadata_dict['flag_references']
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::EWCT'
    var.attrs['sdn_parameter_name'] = 'Eastward current velocity (Eulerian measurement) in the water body by moored ' \
                                      'acoustic doppler current profiler (ADCP)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['standard_name'] = 'eastward_sea_water_velocity'
    var.attrs['data_max'] = np.nanmax(var.data)
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['valid_max'] = uvw_vel_max
    var.attrs['valid_min'] = uvw_vel_min
    
    # LCNSAP01: northward velocity (vel2)
    var = out_obj.LCNSAP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'northward_sea_water_velocity'
    var.attrs['ancillary_variables'] = 'LCNSAP01_QC'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'v'
    var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset'
    var.attrs['flag_meanings'] = metadata_dict['flag_meaning']
    var.attrs['flag_values'] = metadata_dict['flag_values']
    var.attrs['References'] = metadata_dict['flag_references']
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::NSCT'
    var.attrs['sdn_parameter_name'] = 'Northward current velocity (Eulerian measurement) in the water body by moored ' \
                                      'acoustic doppler current profiler (ADCP)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['standard_name'] = 'northward_sea_water_velocity'
    var.attrs['data_max'] = np.nanmax(var.data)
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['valid_max'] = uvw_vel_max
    var.attrs['valid_min'] = uvw_vel_min
    
    # LRZAAP01: vertical velocity (vel3)
    var = out_obj.LRZAAP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'upward_sea_water_velocity'
    var.attrs['ancillary_variables'] = 'LRZAAP01_QC'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'w'
    var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset'
    var.attrs['flag_meanings'] = metadata_dict['flag_meaning']
    var.attrs['flag_values'] = metadata_dict['flag_values']
    var.attrs['References'] = metadata_dict['flag_references']
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::VCSP'
    var.attrs['sdn_parameter_name'] = 'Upward current velocity (Eulerian measurement) in the water body by moored ' \
                                      'acoustic doppler current profiler (ADCP)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['standard_name'] = 'upward_sea_water_velocity'
    var.attrs['data_max'] = np.nanmax(var.data)
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['valid_max'] = uvw_vel_max
    var.attrs['valid_min'] = uvw_vel_min
    
    # LERRAP01: error velocity (vel4)
    var = out_obj.LERRAP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'error_velocity_in_sea_water'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'e'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::ERRV'
    var.attrs['sdn_parameter_name'] = 'Current velocity error in the water body by moored acoustic doppler current ' \
                                      'profiler (ADCP)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['standard_name'] = 'indicative_error_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'
    var.attrs['data_max'] = np.nanmax(var.data)
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['valid_max'] = 2 * uvw_vel_max
    var.attrs['valid_min'] = 2 * uvw_vel_min
    
    # Velocity variable quality flags
    var = out_obj.LCEWAP01_QC
    var.encoding['dtype'] = 'int'
    var.attrs['_FillValue'] = 0
    var.attrs['long_name'] = 'quality flag for LCEWAP01'
    var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset'
    var.attrs['flag_meanings'] = metadata_dict['flag_meaning']
    var.attrs['flag_values'] = metadata_dict['flag_values']
    var.attrs['References'] = metadata_dict['flag_references']
    var.attrs['data_max'] = np.max(var.data)
    var.attrs['data_min'] = np.min(var.data)
    
    var = out_obj.LCNSAP01_QC
    var.encoding['dtype'] = 'int'
    var.attrs['_FillValue'] = 0
    var.attrs['long_name'] = 'quality flag for LCNSAP01'
    var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset'
    var.attrs['flag_meanings'] = metadata_dict['flag_meaning']
    var.attrs['flag_values'] = metadata_dict['flag_values']
    var.attrs['References'] = metadata_dict['flag_references']
    var.attrs['data_max'] = np.max(var.data)
    var.attrs['data_min'] = np.min(var.data)

    var = out_obj.LRZAAP01_QC
    var.encoding['dtype'] = 'int'
    var.attrs['_FillValue'] = 0
    var.attrs['long_name'] = 'quality flag for LRZAAP01'
    var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset'
    var.attrs['flag_meanings'] = metadata_dict['flag_meaning']
    var.attrs['flag_values'] = metadata_dict['flag_values']
    var.attrs['References'] = metadata_dict['flag_references']
    var.attrs['data_max'] = np.max(var.data)
    var.attrs['data_min'] = np.min(var.data)

    # ELTMEP01: seconds since 1970
    var = out_obj.ELTMEP01
    var.encoding['dtype'] = 'd'
    var.encoding['units'] = 'seconds since 1970-01-01T00:00:00Z'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'time_02'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::N/A'
    var.attrs['sdn_parameter_name'] = 'Elapsed time (since 1970-01-01T00:00:00Z)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UTBB'
    var.attrs['sdn_uom_name'] = 'Seconds'
    var.attrs['standard_name'] = 'time'
    
    # TNIHCE01-4: echo intensity beam 1-4
    var = out_obj.TNIHCE01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_echo_intensity_beam_1'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::BEAM_01'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current ' \
                                      'profiler (ADCP) beam 1'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
    var.attrs['sdn_uom_name'] = 'Counts'
    var.attrs['standard_name'] = 'signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.TNIHCE02
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_echo_intensity_beam_2'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::BEAM_02'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current ' \
                                      'profiler (ADCP) beam 2'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
    var.attrs['sdn_uom_name'] = 'Counts'
    var.attrs['standard_name'] = 'signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.TNIHCE03
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_echo_intensity_beam_3'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::BEAM_03'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current ' \
                                      'profiler (ADCP) beam 3'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
    var.attrs['sdn_uom_name'] = 'Counts'
    var.attrs['standard_name'] = 'signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.TNIHCE04
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_echo_intensity_beam_4'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::BEAM_04'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current ' \
                                      'profiler (ADCP) beam 4'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
    var.attrs['sdn_uom_name'] = 'Counts'
    var.attrs['standard_name'] = 'signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # PCGDAP00 - 4: percent good beam 1-4
    if pg_flag == 1:
        # omit percent good beam data, since it isn't available
        pass
    else:
        var = out_obj.PCGDAP00
        var.encoding['dtype'] = 'float32'
        var.attrs['units'] = 'percent'
        var.attrs['_FillValue'] = fillValue
        var.attrs['long_name'] = 'percent_good_beam_1'
        var.attrs['sensor_type'] = 'adcp'
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = metadata_dict['serialNumber']
        var.attrs['generic_name'] = 'PGd'
        var.attrs['legacy_GF3_code'] = 'SDN:GF3::PGDP_01'
        var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler ' \
                                          'current profiler (ADCP) beam 1'
        var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
        var.attrs['sdn_uom_name'] = 'Percent'
        var.attrs['standard_name'] = 'proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water'
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

        var = out_obj.PCGDAP02
        var.encoding['dtype'] = 'float32'
        var.attrs['units'] = 'percent'
        var.attrs['_FillValue'] = fillValue
        var.attrs['long_name'] = 'percent_good_beam_2'
        var.attrs['sensor_type'] = 'adcp'
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = metadata_dict['serialNumber']
        var.attrs['generic_name'] = 'PGd'
        var.attrs['legacy_GF3_code'] = 'SDN:GF3::PGDP_02'
        var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler ' \
                                          'current profiler (ADCP) beam 2'
        var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
        var.attrs['sdn_uom_name'] = 'Percent'
        var.attrs['standard_name'] = 'proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water'
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

        var = out_obj.PCGDAP03
        var.encoding['dtype'] = 'float32'
        var.attrs['units'] = 'percent'
        var.attrs['_FillValue'] = fillValue
        var.attrs['long_name'] = 'percent_good_beam_3'
        var.attrs['sensor_type'] = 'adcp'
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = metadata_dict['serialNumber']
        var.attrs['generic_name'] = 'PGd'
        var.attrs['legacy_GF3_code'] = 'SDN:GF3::PGDP_03'
        var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler ' \
                                          'current profiler (ADCP) beam 3'
        var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
        var.attrs['sdn_uom_name'] = 'Percent'
        var.attrs['standard_name'] = 'proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water'
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

        var = out_obj.PCGDAP04
        var.encoding['dtype'] = 'float32'
        var.attrs['units'] = 'percent'
        var.attrs['_FillValue'] = fillValue
        var.attrs['long_name'] = 'percent_good_beam_4'
        var.attrs['sensor_type'] = 'adcp'
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = metadata_dict['serialNumber']
        var.attrs['generic_name'] = 'PGd'
        var.attrs['legacy_GF3_code'] = 'SDN:GF3::PGDP_04'
        var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler ' \
                                          'current profiler (ADCP) beam 4'
        var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
        var.attrs['sdn_uom_name'] = 'Percent'
        var.attrs['standard_name'] = 'proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water'
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

    # PTCHGP01: pitch
    var = out_obj.PTCHGP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'degree'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'pitch'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::PTCH'
    var.attrs['sdn_parameter_name'] = 'Orientation (pitch) of measurement platform by inclinometer'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UAAA'
    var.attrs['sdn_uom_name'] = 'Degrees'
    var.attrs['standard_name'] = 'platform_pitch'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # ROLLGP01: roll
    var = out_obj.ROLLGP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'degree'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'roll'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::ROLL'
    var.attrs['sdn_parameter_name'] = 'Orientation (roll angle) of measurement platform by inclinometer ' \
                                      '(second sensor)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UAAA'
    var.attrs['sdn_uom_name'] = 'Degrees'
    var.attrs['standard_name'] = 'platform_roll'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # DISTTRAN: height of sea surface (hght)
    var = out_obj.DISTTRAN
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm'
    var.attrs['_FillValue'] = fillValue
    var.attrs['positive'] = 'up'
    var.attrs['long_name'] = 'height of sea surface'
    var.attrs['generic_name'] = 'height'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::HGHT'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::ULAA'
    var.attrs['sdn_uom_name'] = 'Metres'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # TEMPPR01: transducer temp
    var = out_obj.TEMPPR01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'degree_C'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP Transducer Temp.'
    var.attrs['generic_name'] = 'temp'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::te90'
    var.attrs['sdn_parameter_name'] = 'Temperature of the water body'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UPAA'
    var.attrs['sdn_uom_name'] = 'Celsius degree'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # PPSAADCP: instrument depth (formerly DEPFP01)
    var = out_obj.PPSAADCP
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm'
    var.attrs['_FillValue'] = fillValue
    var.attrs['positive'] = 'down'
    var.attrs['long_name'] = 'instrument depth'
    var.attrs['xducer_offset_from_bottom'] = ''
    var.attrs['bin_size'] = cell_size  # bin size
    var.attrs['generic_name'] = 'depth'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::DEPH'
    var.attrs['sdn_parameter_name'] = 'Depth below surface of the water body'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::ULAA'
    var.attrs['sdn_uom_name'] = 'Metres'
    var.attrs['standard_name'] = 'depth'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # ALONZZ01, longitude
    for var in [out_obj.ALONZZ01, out_obj.longitude]:
        var.encoding['_FillValue'] = None
        var.encoding['dtype'] = 'd'
        var.attrs['units'] = 'degrees_east'
        var.attrs['long_name'] = 'longitude'
        var.attrs['legacy_GF3_code'] = 'SDN:GF3::lon'
        var.attrs['sdn_parameter_name'] = 'Longitude east'
        var.attrs['sdn_uom_urn'] = 'SDN:P06::DEGE'
        var.attrs['sdn_uom_name'] = 'Degrees east'
        var.attrs['standard_name'] = 'longitude'

    # ALATZZ01, latitude
    for var in [out_obj.ALATZZ01, out_obj.latitude]:
        var.encoding['_FillValue'] = None
        var.encoding['dtype'] = 'd'
        var.attrs['units'] = 'degrees_north'
        var.attrs['long_name'] = 'latitude'
        var.attrs['legacy_GF3_code'] = 'SDN:GF3::lat'
        var.attrs['sdn_parameter_name'] = 'Latitude north'
        var.attrs['sdn_uom_urn'] = 'SDN:P06::DEGN'
        var.attrs['sdn_uom_name'] = 'Degrees north'
        var.attrs['standard_name'] = 'latitude'

    # HEADCM01: heading
    var = out_obj.HEADCM01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'degree'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'heading'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::HEAD'
    var.attrs['sdn_parameter_name'] = 'Orientation (horizontal relative to true north) of measurement device {heading}'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UAAA'
    var.attrs['sdn_uom_name'] = 'Degrees'
    var.attrs['standard_name'] = 'platform_orientation'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # PRESPR01: pressure
    var = out_obj.PRESPR01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'dbar'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'pressure'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['ancillary_variables'] = 'PRESPR01_QC'
    var.attrs['comment'] = 'Quality flag indicates negative pressure values in the time series'
    var.attrs['flag_meanings'] = metadata_dict['flag_meaning']
    var.attrs['flag_values'] = metadata_dict['flag_values']
    var.attrs['References'] = metadata_dict['flag_references']
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::PRES'
    var.attrs['sdn_parameter_name'] = 'Pressure (spatial co-ordinate) exerted by the water body by profiling ' \
                                      'pressure sensor and corrected to read zero at sea level'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UPDB'
    var.attrs['sdn_uom_name'] = 'Decibars'
    var.attrs['standard_name'] = 'sea_water_pressure'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # PRESPR01_QC: pressure quality flag
    var = out_obj.PRESPR01_QC
    var.encoding['dtype'] = 'int'
    var.attrs['_FillValue'] = 0
    var.attrs['long_name'] = 'quality flag for PRESPR01'
    var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset and ' \
                           'identification of negative pressure values'
    var.attrs['flag_meanings'] = metadata_dict['flag_meaning']
    var.attrs['flag_values'] = metadata_dict['flag_values']
    var.attrs['References'] = metadata_dict['flag_references']
    var.attrs['data_max'] = np.nanmax(var.data)
    var.attrs['data_min'] = np.nanmin(var.data)
    
    # SVELCV01: sound velocity
    var = out_obj.SVELCV01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'speed of sound'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::SVEL'
    var.attrs['sdn_parameter_name'] = 'Sound velocity in the water body by computation from temperature and ' \
                                      'salinity by unspecified algorithm'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['standard_name'] = 'speed_of_sound_in_sea_water'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # DTUT8601: time values as ISO8601 string, YY-MM-DD hh:mm:ss
    var = out_obj.DTUT8601
    var.encoding['dtype'] = 'U24'  # 24-character string
    var.attrs['note'] = 'time values as ISO8601 string, YY-MM-DD hh:mm:ss'
    var.attrs['time_zone'] = 'UTC'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::time_string'
    var.attrs['sdn_parameter_name'] = 'String corresponding to format \'YYYY-MM-DDThh:mm:ss.sssZ\' or other ' \
                                      'valid ISO8601 string'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::TISO'
    var.attrs['sdn_uom_name'] = 'ISO8601'

    # CMAGZZ01-4: correlation magnitude
    var = out_obj.CMAGZZ01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_1'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'CM'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::CMAG_01'
    var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by ' \
                                      'moored acoustic doppler current profiler (ADCP) beam 1'
    var.attrs['standard_name'] = 'beam_consistency_indicator_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.CMAGZZ02
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_2'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'CM'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::CMAG_02'
    var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by ' \
                                      'moored acoustic doppler current profiler (ADCP) beam 2'
    var.attrs['standard_name'] = 'beam_consistency_indicator_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.CMAGZZ03
    var.attrs['units'] = 'counts'
    var.encoding['dtype'] = 'float32'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_3'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'CM'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::CMAG_03'
    var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by ' \
                                      'moored acoustic doppler current profiler (ADCP) beam 3'
    var.attrs['standard_name'] = 'beam_consistency_indicator_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.CMAGZZ04
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_4'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'CM'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::CMAG_04'
    var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by ' \
                                      'moored acoustic doppler current profiler (ADCP) beam 4'
    var.attrs['standard_name'] = 'beam_consistency_indicator_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)
    # done variables

    # Add Vertical Beam variable attrs for Sentinel V instruments
    if metadata_dict['model'] == 'sv' and vb_flag == 0:
        var = out_obj.LRZUVP01
        var.encoding['dtype'] = 'float32'
        var.attrs['units'] = 'm s-1'
        var.attrs['_FillValue'] = fillValue
        var.attrs['long_name'] = 'upward_sea_water_velocity_by_vertical_beam'
        var.attrs['ancillary_variables'] = 'LRZUVP01_QC'
        var.attrs['sensor_type'] = 'adcp'
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = metadata_dict['serialNumber']
        var.attrs['generic_name'] = 'vv'
        var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset'
        var.attrs['flag_meanings'] = metadata_dict['flag_meaning']
        var.attrs['flag_values'] = metadata_dict['flag_values']
        var.attrs['References'] = metadata_dict['flag_references']
        var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
        var.attrs['sdn_uom_name'] = 'Metres per second'
        var.attrs['standard_name'] = 'upward_sea_water_velocity'
        var.attrs['data_max'] = np.nanmax(var.data)
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['valid_max'] = uvw_vel_max
        var.attrs['valid_min'] = uvw_vel_min

        var = out_obj.LRZUVP01_QC
        var.encoding['dtype'] = 'int'
        var.attrs['_FillValue'] = 0
        var.attrs['long_name'] = 'quality flag for LRZUVP01'
        var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset'
        var.attrs['flag_meanings'] = metadata_dict['flag_meaning']
        var.attrs['flag_values'] = metadata_dict['flag_values']
        var.attrs['References'] = metadata_dict['flag_references']
        var.attrs['data_max'] = np.max(var.data)
        var.attrs['data_min'] = np.min(var.data)

        var = out_obj.TNIHCE05
        var.encoding['dtype'] = 'float32'
        var.attrs['units'] = 'counts'
        var.attrs['_FillValue'] = fillValue
        var.attrs['long_name'] = 'ADCP_echo_intensity_beam_5'
        var.attrs['sensor_type'] = 'adcp'
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = metadata_dict['serialNumber']
        var.attrs['generic_name'] = 'AGC'
        var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current ' \
                                          'profiler (ADCP) vertical beam'
        var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
        var.attrs['sdn_uom_name'] = 'Counts'
        var.attrs['standard_name'] = 'signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water'
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

        var = out_obj.CMAGZZ05
        var.encoding['dtype'] = 'float32'
        var.attrs['units'] = 'counts'
        var.attrs['_FillValue'] = fillValue
        var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_5'
        var.attrs['sensor_type'] = 'adcp'
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = metadata_dict['serialNumber']
        var.attrs['generic_name'] = 'CM'
        var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by ' \
                                          'moored acoustic doppler current profiler (ADCP) vertical beam'
        var.attrs[
            'standard_name'] = 'beam_consistency_indicator_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)
        
        if vb_pg_flag == 0:
            var = out_obj.PCGDAP05
            var.encoding['dtype'] = 'float32'
            var.attrs['units'] = 'percent'
            var.attrs['_FillValue'] = fillValue
            var.attrs['long_name'] = 'percent_good_beam_5'
            var.attrs['sensor_type'] = 'adcp'
            var.attrs['sensor_depth'] = sensor_depth
            var.attrs['serial_number'] = metadata_dict['serialNumber']
            var.attrs['generic_name'] = 'PGd'
            var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler ' \
                                              'current profiler (ADCP) vertical beam'
            var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
            var.attrs['sdn_uom_name'] = 'Percent'
            var.attrs['standard_name'] = 'proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water'
            var.attrs['data_min'] = np.nanmin(var.data)
            var.attrs['data_max'] = np.nanmax(var.data)

    return


def create_meta_dict_L1(adcp_meta):
    """
    Read in a csv metadata file and output in dictionary format
    Inputs:
        - adcp_meta: csv-format file containing metadata for raw ADCP file
    Outputs:
        - meta_dict: a dictionary containing the metadata from the csv file and additional metadata on conventions
    """
    meta_dict = {}
    with open(adcp_meta) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            # extract all metadata from csv file into dictionary -- some items not passed to netCDF file but are extracted anyway
            if row[0] != "Name":
                meta_dict[row[0]] = row[1]
            elif row[0] == '' and row[1] == '':
                warnings.warn('Metadata file contains a blank row; skipping this row', UserWarning)
            elif row[0] != '' and row[1] == '':
                warnings.warn('Metadata item in csv file has blank value; skipping this row '
                              'in metadata file', UserWarning)
            else:
                continue

    # Add conventions metadata to meta_dict
    meta_dict['deployment_type'] = 'Sub Surface'
    meta_dict['flag_meaning'] = 'no_quality_control, good_value, probably_good_value, probably_bad_value, ' \
                                'bad_value, changed_value, value_below_detection, value_in_excess, ' \
                                'interpolated_value, missing_value'
    meta_dict['flag_references'] = 'BODC SeaDataNet'
    meta_dict['flag_values'] = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'
    meta_dict['keywords'] = 'Oceans > Ocean Circulation > Ocean Currents'
    meta_dict['keywords_vocabulary'] = 'GCMD Science Keywords'
    meta_dict['naming_authority'] = 'BODC, MEDS, CF v72'
    meta_dict['variable_code_reference'] = 'BODC P01'
    meta_dict['Conventions'] = "CF-1.8"

    return meta_dict


def nc_create_L1(inFile, file_meta, dest_dir, start_year=None, time_file=None):
    
    # If your raw file came from a NarrowBand instrument, you must also use the create_nc_L1() start_year optional kwarg (int type)
    # If your raw file has time values out of range, you must also use the create_nc_L1() time_file optional kwarg
    # Use the time_file kwarg to read in a csv file containing time entries spanning the range of deployment and using the
    # instrument sampling interval

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Splice file name to get output netCDF file name
    out_name = os.path.basename(inFile)[:-4] + '.adcp.L1.nc'
    print(out_name)
    if not dest_dir.endswith('/') or not dest_dir.endswith('\\'):
        out_absolute_name = os.path.abspath(dest_dir + '/' + out_name)
    else:
        out_absolute_name = os.path.abspath(dest_dir + out_name)

    # Read information from metadata file into a dictionary, called meta_dict
    meta_dict = create_meta_dict_L1(file_meta)
    
    # Assign model, model_long name, and manufacturer
    if meta_dict["instrumentSubtype"].upper() == "WORKHORSE":
        meta_dict['model'] = "wh"
        model_long = "RDI WH Long Ranger"
        meta_dict['manufacturer'] = 'teledyne rdi'
    elif meta_dict["instrumentSubtype"].upper() == "BROADBAND":
        meta_dict['model'] = "bb"
        model_long = "RDI BB"
        meta_dict['manufacturer'] = 'teledyne rdi'
    elif meta_dict["instrumentSubtype"].upper() == "NARROWBAND":
        meta_dict['model'] = "nb"
        model_long = "RDI NB"
        meta_dict['manufacturer'] = 'teledyne rdi'
    elif meta_dict["instrumentSubtype"].upper() == "SENTINEL V":
        meta_dict['model'] = "sv"
        model_long = "RDI SV"
        meta_dict['manufacturer'] = 'teledyne rdi'
    elif meta_dict["instrumentSubtype"].upper() == 'OCEAN SURVEYOR':
        meta_dict['model'] = "os"
        model_long = "RDI OS"
        meta_dict['manufacturer'] = "teledyne rdi"
    else:
        pass

    # Check if model was read into dictionary correctly
    if 'model' not in meta_dict:
        ValueError("instrumentSubtype value of \"{}\" not valid".format(meta_dict['instrumentSubtype']))

    print(meta_dict['model'])
    print('Read in csv metadata file')

    # Read in data and start processing

    # Read in raw ADCP file and model type
    if meta_dict['model'] == 'nb':
        data = rawfile(inFile, meta_dict['model'], trim=True, yearbase=start_year)
    else:
        data = rawfile(inFile, meta_dict['model'], trim=True)
    print('Read in raw data')

    # Extract multidimensional variables from data object: 
    # fixed leader, velocity, amplitude intensity, correlation magnitude, and percent good
    fixed_leader = data.read(varlist=['FixedLeader'])
    vel = data.read(varlist=['Velocity'])
    amp = data.read(varlist=['Intensity'])
    cor = data.read(varlist=['Correlation'])
    pg = data.read(varlist=['PercentGood'])

    # If model == Sentinel V, read in vertical beam data
    if meta_dict['model'] == 'sv':
        # vb_leader = data.read(varlist=['VBLeader'])
        vb_vel = data.read(varlist=['VBVelocity'])
        vb_amp = data.read(varlist=['VBIntensity'])
        vb_cor = data.read(varlist=['VBCorrelation'])
        vb_pg = data.read(varlist=['VBPercentGood'])

    # Create flags if pg data or vb_pg data are missing
    flag_pg = 0
    flag_vb = 0
    flag_vb_pg = 0
    try:
        print(pg.pg1.data[:5])
    except AttributeError:
        flag_pg += 1

    if meta_dict['model'] == 'sv':
        # Test for missing Sentinel V vertical beam data; if true treat file as regular 4-beam file
        try:
            print(vb_vel.vbvel.data[:5])
        except AttributeError:
            flag_vb += 1
        # Test for missing vertical beam percent good data
        try:
            print(vb_pg.vb_pg.data[:5])
        except AttributeError:
            flag_vb_pg += 1

    # Metadata value corrections

    # Convert numeric values to numerics
    meta_dict['country_institute_code'] = int(meta_dict['country_institute_code'])
    
    for key in ['instrument_depth', 'latitude', 'longitude', 'water_depth', 'magnetic_variation']:
        meta_dict[key] = float(meta_dict[key])

    # Add leading zero to serial numbers that have 3 digits
    if len(str(meta_dict['serialNumber'])) == 3:
        meta_dict['serialNumber'] = '0' + str(meta_dict['serialNumber'])
    # Overwrite serial number to include the model: upper returns uppercase
    meta_dict['serialNumber'] = meta_dict['model'].upper() + meta_dict['serialNumber']
    # Add instrument model variable value
    meta_dict['instrumentModel'] = '{} ADCP {}kHz ({})'.format(model_long, data.sysconfig['kHz'],
                                                               meta_dict['serialNumber'])

    # Correct flag_meanings values if they are comma-separated
    if ',' in meta_dict['flag_meaning']:
        flag_meaning_list = [x.strip() for x in meta_dict['flag_meaning'].split(',')]
        meta_dict['flag_meaning'] = np.array(flag_meaning_list, dtype='U{}'.format(
            len(max(flag_meaning_list, key=len))))

    # Convert flag_values from single string to numpy array
    flag_values_list = [x.strip() for x in meta_dict['flag_values'].split(',')]
    meta_dict['flag_values'] = np.array(flag_values_list, dtype='int32')

    # Begin writing processing history, which will be added as a global attribute to the output netCDF file
    meta_dict['processing_history'] = "Metadata read in from log sheet and combined with raw data to export " \
                                      "as netCDF file."

    # Extract metadata from data object

    # Orientation code from Eric Firing
    # Orientation values such as 65535 and 231 cause SysCfg().up to generate an IndexError: list index out of range
    try:
        orientations = [SysCfg(fl).up for fl in fixed_leader.raw.FixedLeader['SysCfg']]
        meta_dict['orientation'] = mean_orientation(orientations)
    except IndexError:
        warnings.warn('Orientation obtained from data.sysconfig[\'up\'] to avoid IndexError: list index out of range', 
                      UserWarning)        
        meta_dict['orientation'] = 'up' if data.sysconfig['up'] else 'down'

    # Retrieve beam pattern
    if data.sysconfig['convex']:
        meta_dict['beam_pattern'] = 'convex'
    else:
        meta_dict['beam_pattern'] = 'concave'

    # Set up dimensions and variables

    time_s, time_DTUT8601 = convert_time_var(time_var=vel.dday, number_of_profiles=data.nprofs, metadata_dict=meta_dict, 
                                             origin_year=data.yearbase, time_csv=time_file)

    # Distance dimension
    distance = np.round(vel.dep.data, decimals=2)

    # Continue setting up variables

    # Convert SoundSpeed from int16 to float32
    sound_speed = np.float32(vel.VL['SoundSpeed'])

    # Convert pressure
    pressure = assign_pres(vel_var=vel, metadata_dict=meta_dict)
    
    # Depth

    # Apply equivalent of swDepth() to depth data: Calculate height from sea pressure using gsw package
    # negative so that depth is positive; units=m
    depth = -np.round(gsw.conversions.z_from_p(p=pressure, lat=meta_dict['latitude']), decimals=2)
    print('Calculated sea surface height from sea pressure using gsw package')
    
    # Check instrument_depth from metadata csv file: compare with pressure values
    check_depths(pressure, distance, meta_dict['instrument_depth'], meta_dict['water_depth']) 

    # Calculate sensor depth of instrument based off mean instrument depth
    sensor_dep = np.nanmean(depth)
    meta_dict['processing_history'] += " Sensor depth and mean depth set to {} based on trimmed depth values.".format(
        str(sensor_dep))

    # Calculate height of sea surface
    if meta_dict['orientation'] == 'up':
        DISTTRAN = np.round(sensor_dep - distance, decimals=2)
    else:
        DISTTRAN = np.round(sensor_dep + distance, decimals=2)

    # Round sensor_dep
    sensor_dep = np.round(sensor_dep, decimals=2)

    # Adjust velocity data

    # Set velocity values of -32768.0 to nans, since -32768.0 is the automatic fill_value for pycurrents
    vel.vel.data[vel.vel.data == -32768.0] = np.nan

    if meta_dict['model'] == 'sv' and flag_vb == 0:
        vb_vel.vbvel.data[vb_vel.vbvel.data == -32768.0] = np.nan

    # Rotate into earth if not in enu already; this makes the netCDF bigger
    # For Sentinel V instruments, transformations are done independently of vertical beam velocity data
    if vel.trans.coordsystem != 'earth' and vel.trans.coordsystem != 'enu':
        vel1, vel2, vel3, vel4 = coordsystem_2enu(vel_var=vel, fixed_leader_var=fixed_leader, metadata_dict=meta_dict)
    else:
        vel1 = vel.vel1.data
        vel2 = vel.vel2.data
        vel3 = vel.vel3.data
        vel4 = vel.vel4.data
        meta_dict['coord_system'] = 'enu'

    # Correct magnetic declination in velocities
    LCEWAP01, LCNSAP01 = correct_true_north(vel1, vel2, meta_dict)

    # Flag data based on cut_lead_ensembles and cut_trail_ensembles

    # Set start and end indices
    e1 = int(meta_dict['cut_lead_ensembles'])  # "ensemble 1"
    e2 = int(meta_dict['cut_trail_ensembles'])  # "ensemble 2"

    # Flag measurements from before deployment and after recovery using e1 and e2
        
    PRESPR01_QC = flag_pressure(pres=pressure, ens1=e1, ens2=e2, metadata_dict=meta_dict)

    if meta_dict['model'] != 'sv' or flag_vb == 1:
        LCEWAP01_QC, LCNSAP01_QC, LRZAAP01_QC = flag_velocity(e1, e2, data.NCells, LCEWAP01, LCNSAP01,
                                                              vel3)
    else:
        LCEWAP01_QC, LCNSAP01_QC, LRZAAP01_QC, LRZUVP01_QC = flag_velocity(e1, e2, data.NCells, LCEWAP01,
                                                                           LCNSAP01, vel3,
                                                                           vb_vel.vbvel.data)

    # Limit variables (depth, temperature, pitch, roll, heading, sound_speed) from before dep. and after rec. of ADCP
    for variable in [depth, vel.temperature, vel.pitch, vel.roll, vel.heading, sound_speed]:
        variable[:e1] = np.nan
        if e2 != 0:
            variable[-e2:] = np.nan

    if e2 != 0:
        meta_dict['processing_history'] += " Velocity, pressure, depth, temperature, pitch, roll, heading, and " \
                                           "sound_speed limited by deployment ({} UTC) and recovery ({} UTC) " \
                                           "times.".format(time_DTUT8601[e1], time_DTUT8601[-e2])
    else:
        meta_dict['processing_history'] += " Velocity, pressure, depth, temperature, pitch, roll, heading, and " \
                                           "sound_speed limited by " \
                                           "deployment ({} UTC) time.".format(time_DTUT8601[e1])

    meta_dict['processing_history'] += ' Level 1 processing was performed on the dataset. This entailed corrections' \
                                       ' for magnetic declination based on an average of the dataset and cleaning ' \
                                       'of the beginning and end of the dataset. The leading {} ensembles and the ' \
                                       'trailing {} ensembles ' \
                                       'were removed from the data set.'.format(meta_dict['cut_lead_ensembles'],
                                                                                meta_dict['cut_trail_ensembles'])

    print('Finished QCing data; making netCDF object next')

    # Make into netCDF file
    
    # Create xarray Dataset object containing all dimensions and variables
    # Sentinel V instruments don't have percent good ('pg') variables
    out = xr.Dataset(coords={'time': time_s, 'distance': distance},
                     data_vars={'LCEWAP01': (['distance', 'time'], LCEWAP01.transpose()),
                                'LCNSAP01': (['distance', 'time'], LCNSAP01.transpose()),
                                'LRZAAP01': (['distance', 'time'], vel3.transpose()),
                                'LERRAP01': (['distance', 'time'], vel4.transpose()),
                                'LCEWAP01_QC': (['distance', 'time'], LCEWAP01_QC.transpose()),
                                'LCNSAP01_QC': (['distance', 'time'], LCNSAP01_QC.transpose()),
                                'LRZAAP01_QC': (['distance', 'time'], LRZAAP01_QC.transpose()),
                                'ELTMEP01': (['time'], time_s),
                                'TNIHCE01': (['distance', 'time'], amp.amp1.transpose()),
                                'TNIHCE02': (['distance', 'time'], amp.amp2.transpose()),
                                'TNIHCE03': (['distance', 'time'], amp.amp3.transpose()),
                                'TNIHCE04': (['distance', 'time'], amp.amp4.transpose()),
                                'CMAGZZ01': (['distance', 'time'], cor.cor1.transpose()),
                                'CMAGZZ02': (['distance', 'time'], cor.cor2.transpose()),
                                'CMAGZZ03': (['distance', 'time'], cor.cor3.transpose()),
                                'CMAGZZ04': (['distance', 'time'], cor.cor4.transpose()),
                                'PTCHGP01': (['time'], vel.pitch),
                                'HEADCM01': (['time'], vel.heading),
                                'ROLLGP01': (['time'], vel.roll),
                                'TEMPPR01': (['time'], vel.temperature),
                                'DISTTRAN': (['distance'], DISTTRAN),
                                'PPSAADCP': (['time'], depth),
                                'ALATZZ01': ([], meta_dict['latitude']),
                                'ALONZZ01': ([], meta_dict['longitude']),
                                'latitude': ([], meta_dict['latitude']),
                                'longitude': ([], meta_dict['longitude']),
                                'PRESPR01': (['time'], pressure),
                                'PRESPR01_QC': (['time'], PRESPR01_QC),
                                'SVELCV01': (['time'], sound_speed),
                                'DTUT8601': (['time'], time_DTUT8601),
                                'filename': ([], out_name[:-3]),
                                'instrument_serial_number': ([], meta_dict['serialNumber']),
                                'instrument_model': ([], meta_dict['instrumentModel'])})

    if flag_pg == 0:
        out = out.assign(PCGDAP00=(('distance', 'time'), pg.pg1.transpose()))
        out = out.assign(PCGDAP02=(('distance', 'time'), pg.pg2.transpose()))
        out = out.assign(PCGDAP03=(('distance', 'time'), pg.pg3.transpose()))
        out = out.assign(PCGDAP04=(('distance', 'time'), pg.pg4.transpose()))

    if meta_dict['model'] == 'sv' and flag_vb == 0:
        out = out.assign(LRZUVP01=(('distance', 'time'), vb_vel.vbvel.data.transpose()))
        out = out.assign(LRZUVP01_QC=(('distance', 'time'), LRZUVP01_QC.transpose()))
        out = out.assign(TNIHCE05=(('distance', 'time'), vb_amp.raw.VBIntensity.transpose()))
        out = out.assign(CMAGZZ05=(('distance', 'time'), vb_cor.VBCorrelation.transpose()))
        if flag_vb_pg == 0:
            out = out.assign(PCGDAP05=(('distance', 'time'), vb_pg.raw.VBPercentGood.transpose()))  # OR vb_pg.VBPercentGood.transpose() ?

    # Add attributes to each variable
    fill_value = 1e+15
    add_attrs_2vars_L1(out_obj=out, metadata_dict=meta_dict, sensor_depth=sensor_dep, cell_size=data.CellSize,
                       fillValue=fill_value, pg_flag=flag_pg, vb_flag=flag_vb, vb_pg_flag=flag_vb_pg)

    # Global attributes

    # Add select meta_dict items as global attributes
    pass_dict_keys = ['cut_lead_ensembles', 'cut_trail_ensembles', 'processing_level', 'model']
    for key, value in meta_dict.items():
        if key in pass_dict_keys:
            pass
        elif key == 'serialNumber':
            out.attrs['serial_number'] = value
        else:
            out.attrs[key] = value

    # Attributes not from metadata file:
    out.attrs['time_coverage_duration'] = vel.dday[-1] - vel.dday[0]
    out.attrs['time_coverage_duration_units'] = "days"
    # ^calculated from start and end times; in days: add time_coverage_duration_units?
    out.attrs['cdm_data_type'] = "station"
    out.attrs['number_of_beams'] = data.NBeams
    # out.attrs['nprofs'] = data.nprofs #number of ensembles
    out.attrs['numberOfCells'] = data.NCells
    out.attrs['pings_per_ensemble'] = data.NPings
    out.attrs['bin1Distance'] = data.Bin1Dist
    # out.attrs['Blank'] = data.Blank #?? blanking distance?
    out.attrs['cellSize'] = data.CellSize
    out.attrs['pingtype'] = data.pingtype
    out.attrs['transmit_pulse_length_cm'] = vel.FL['Pulse']
    out.attrs['instrumentType'] = "adcp"
    out.attrs['manufacturer'] = meta_dict['manufacturer']
    out.attrs['source'] = "Python code: github: pycurrents_ADCP_processing"
    now = datetime.datetime.now()
    out.attrs['date_modified'] = now.strftime("%Y-%m-%d %H:%M:%S")
    out.attrs['_FillValue'] = str(fill_value)
    out.attrs['featureType'] = "profileTimeSeries"
    out.attrs['firmware_version'] = str(vel.FL.FWV) + '.' + str(vel.FL.FWR)  # firmwareVersion
    out.attrs['frequency'] = str(data.sysconfig['kHz'])
    out.attrs['beam_angle'] = str(fixed_leader.sysconfig['angle'])  # beamAngle
    out.attrs['systemConfiguration'] = bin(fixed_leader.FL['SysCfg'])[-8:] + '-' + bin(fixed_leader.FL['SysCfg'])[
                                                                                   :9].replace('b', '')
    out.attrs['sensor_source'] = '{0:08b}'.format(vel.FL['EZ'])  # sensorSource
    out.attrs['sensors_avail'] = '{0:08b}'.format(vel.FL['SA'])  # sensors_avail
    out.attrs['three_beam_used'] = str(vel.trans['threebeam']).upper()  # netCDF4 file format doesn't support bool
    out.attrs['valid_correlation_range'] = vel.FL['LowCorrThresh']  # lowCorrThresh
    out.attrs['minmax_percent_good'] = "100"  # hardcoded in oceNc_create()
    out.attrs['error_velocity_threshold'] = "2000 m s-1"
    out.attrs['false_target_reject_values'] = 50  # falseTargetThresh
    out.attrs['data_type'] = "adcp"
    out.attrs['pred_accuracy'] = 1  # velocityResolution * 1000
    out.attrs['creator_type'] = "person"
    out.attrs['n_codereps'] = vel.FL.NCodeReps
    out.attrs['xmit_lag'] = vel.FL.TransLag
    out.attrs['time_coverage_start'] = time_DTUT8601[e1] + ' UTC'
    out.attrs['time_coverage_end'] = time_DTUT8601[-e2 - 1] + ' UTC'  # -1 is last time entry before cut ones

    # geospatial lat, lon, and vertical min/max calculations
    out.attrs['geospatial_lat_min'] = meta_dict['latitude']
    out.attrs['geospatial_lat_max'] = meta_dict['latitude']
    out.attrs['geospatial_lat_units'] = "degrees_north"
    out.attrs['geospatial_lon_min'] = meta_dict['longitude']
    out.attrs['geospatial_lon_max'] = meta_dict['longitude']
    out.attrs['geospatial_lon_units'] = "degrees_east"

    # sensor_depth is a variable attribute, not a global attribute
    if out.attrs['orientation'] == 'up':
        out.attrs['geospatial_vertical_min'] = sensor_dep - np.nanmax(out.distance.data)
        out.attrs['geospatial_vertical_max'] = sensor_dep - np.nanmin(out.distance.data)
    elif out.attrs['orientation'] == 'down':
        out.attrs['geospatial_vertical_min'] = sensor_dep + np.nanmin(out.distance.data)
        out.attrs['geospatial_vertical_max'] = sensor_dep + np.nanmax(out.distance.data)

    # Export the 'out' object as a netCDF file
    out.to_netcdf(out_absolute_name, mode='w', format='NETCDF4')
    out.close()

    return out_absolute_name


def example_usage_L1():
    # Specify raw ADCP file to create nc file from, along with associated csv metadata file

    # raw .000 file
    raw_file = "./sample_data/a1_20050503_20050504_0221m.000"
    # csv metadata file
    raw_file_meta = "./sample_data/a1_20050503_20050504_0221m_meta_L1.csv"

    dest_dir = 'dest_dir'

    # Create netCDF file
    nc_name = nc_create_L1(inFile=raw_file, file_meta=raw_file_meta, dest_dir=dest_dir, start_year=None, time_file=None)

    # Produce new netCDF file that includes a geographic_area variable
    add_var2nc.add_geo(nc_name, dest_dir)

    return
