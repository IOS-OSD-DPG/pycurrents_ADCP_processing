"""
author: Hana Hourston
date: Jan. 23, 2020

about: This script is adapted from Jody Klymak's at
https://gist.github.com/jklymak/b39172bd0f7d008c81e32bf0a72e2f09
for L1 processing raw ADCP data.

Contributions from: Di Wan, Eric Firing

NOTES:
# If your raw file came from a NarrowBand instrument, you must also use the
  create_nc_L1() start_year optional kwarg (int type)
# If your raw file has time values out of range, you must also use the
  create_nc_L1() time_file optional kwarg
# Use the time_file kwarg to read in a csv file containing time entries
  spanning the range of deployment and using the instrument sampling interval

"""

import os
import csv
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import warnings
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pycurrents.adcp import rdiraw
import pycurrents.adcp.transform as transform
import gsw
from ruamel.yaml import YAML
from pycurrents_ADCP_processing import utils
from shapely.geometry import Point

_FillValue = np.nan
bodc_flag_dict = {
    'no_quality_control': 0, 'good_value': 1, 'probably_good_value': 2, 'probably_bad_value': 3,
    'bad_value': 4, 'changed_value': 5, 'value_below_detection': 6, 'value_in_excess': 7,
    'interpolated_value': 8, 'missing_value': 9
}


def mean_orientation(o: list):
    # orientation, o, is a list of bools with True=up and False=down
    if sum(o) > len(o) / 2:
        return 'up'
    elif sum(o) < len(o) / 2:
        return 'down'
    else:
        ValueError('Number of \"up\" orientations equals number of \"down\" '
                   'orientations in data subset')


def correct_true_north(measured_east, measured_north, meta_dict: dict):
    # change angle to negative of itself
    # Di Wan's magnetic declination correction code: Takes 0 DEG from E-W axis
    # mag_decl: magnetic declination value; float type
    # measured_east: measured Eastward velocity data; array type
    # measured_north: measured Northward velocity data; array type
    angle_rad = -meta_dict['magnetic_variation'] * np.pi / 180.
    east_true = measured_east * np.cos(angle_rad) - measured_north * np.sin(angle_rad)
    north_true = measured_east * np.sin(angle_rad) + measured_north * np.cos(angle_rad)

    # Round to the input number of decimal places
    east_true = np.round(east_true, decimals=3)
    north_true = np.round(north_true, decimals=3)

    meta_dict['processing_history'] += (" Average magnetic declination applied to eastward and northward "
                                        "velocities; declination = {}.").format(meta_dict['magnetic_variation'])

    return east_true, north_true


def convert_time_var(time_var, number_of_profiles, meta_dict: dict, origin_year,
                     time_csv) -> np.ndarray:
    """
    Includes exception handling for bad times
    time_var: vel.dday; time variable with units in days since the beginning
              of the year in which measurements
              started being taken by the instrument
    number_of_profiles: the number of profiles (ensembles) recorded by the
                        instrument over the time series
    metadata_dict: dictionary object of metadata items
    """

    # data.yearbase is an integer of the year that the timeseries starts (e.g., 2016)
    data_origin = pd.Timestamp(str(origin_year) + '-01-01')  # convert to date object

    try:
        # convert time variable to elapsed time since 1970-01-01T00:00:00Z
        t_s = np.array(
            pd.to_datetime(time_var, unit='D', origin=data_origin,
                           utc=True).strftime('%Y-%m-%d %H:%M:%S'),
            dtype='datetime64[s]')
        # # DTUT8601 variable: time strings
        # t_DTUT8601 = pd.to_datetime(time_var, unit='D', origin=data_origin,
        #                             utc=True).strftime(
        #     '%Y-%m-%d %H:%M:%S')  # don't need %Z in strftime
    except OutOfBoundsDatetime or OverflowError:
        print('Using user-created time range')
        t_s = np.zeros(shape=number_of_profiles, dtype='datetime64[s]')
        # t_DTUT8601 = np.empty(shape=number_of_profiles, dtype='<U100')
        with open(time_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # Skip headers
            next(csv_reader, None)
            for count, row in enumerate(csv_reader):
                if row[0] == '':
                    pass
                else:
                    t_s[count] = np.datetime64(
                        pd.to_datetime(row[0], utc=True).strftime(
                            '%Y-%m-%d %H:%M:%S'))
                    # t_DTUT8601[count] = pd.to_datetime(
                    #     row[0], utc=True).strftime('%Y-%m-%d %H:%M:%S')

        meta_dict['processing_history'] += ' OutOfBoundsDateTime exception ' \
                                           'triggered by raw time data; used user-generated time ' \
                                           'range as time data.'

    # Set times out of range to NaNs
    # Get timedelta object with value of one year long
    # Use this duration because ADCPs aren't moored for over 2 years (sometimes just over one year)
    yr_value = pd.Timedelta(365 * 2, unit='D')
    time_median = pd.to_datetime(pd.Series(t_s).astype('int64').median())
    indexer_out_rng = np.where(t_s > time_median + yr_value)[0]

    # Some silly little conversions
    t_df = pd.Series(t_s)
    t_df.loc[indexer_out_rng] = pd.NaT
    t_s = np.array(t_df, dtype='datetime64[s]')

    # t_s[indexer_out_rng] = pd.NaT
    # dtut_df = pd.Series(t_DTUT8601)
    # dtut_df.loc[indexer_out_rng] = 'NaN'  # String type
    # t_DTUT8601 = np.array(dtut_df)

    return t_s  # , t_DTUT8601


def numpy_datetime_to_str(t: np.datetime64):
    return t.astype(str).replace('T', ' ')


def assign_pres(vel_var, meta_dict: dict):
    """
    Assign pressure and calculate it froms static instrument depth if ADCP
    missing pressure sensor
    vel_var: "vel" variable created from BBWHOS class object, using the
    command: data.read(varlist=['vel'])
    metadata_dict: dictionary object of metadata items
    """

    if meta_dict['model'] == 'wh' or meta_dict['model'] == 'sv':
        # convert decapascal to decibars
        pres = np.array(vel_var.VL['Pressure'] / 1000, dtype='float32')

        # Calculate pressure based on static instrument depth if missing
        # pressure sensor; extra condition added for zero pressure or weird
        # pressure values
        # Handle no unique modes from statistics.mode(pressure)
        pressure_unique, counts = np.unique(pres, return_counts=True)
        # index_of_zero = np.where(pressure_unique == 0)
        # print('np.max(counts):', np.max(counts), sep=' ')
        # print('counts[index_of_zero]:', counts[index_of_zero], sep=' ')
        # print('serial number:', metadata_dict['serial_number'])

    # Check if model is type missing pressure sensor or if zero is a mode of pressure
    # Amendment 2023-09-18: change to "if pressure is static for over half the dataset" as the former became a bug
    if meta_dict['model'] == 'bb' or np.max(counts) > len(pres) / 2:  # or np.max(counts) == counts[index_of_zero]:
        p = np.round(gsw.conversions.p_from_z(-meta_dict['instrument_depth'],
                                              meta_dict['latitude']),
                     decimals=1)  # depth negative because positive is up for this function
        pres = np.repeat(p, len(vel_var.vel1.data))
        meta_dict['processing_history'] += " Pressure calculated from static instrument depth ({} m) using " \
                                           "the TEOS-10 75-term expression for specific volume.".format(
            meta_dict['instrument_depth'])
        warnings.warn('Pressure values calculated from static instrument depth', UserWarning)

    return pres


def check_depths(pres, dist, instr_depth: float, water_depth: float):
    """
    Check user-entered instrument_depth and compare with pressure values
    pres: pressure variable; array type
    dist: distance variable (contains distance of each bin from ADCP); array type
    instr_depth: depth of the instrument
    water_depth: depth of the water
    """

    depths_check = np.mean(pres[:]) - dist
    inst_depth_check = depths_check[0] + dist[0]
    abs_difference = np.absolute(inst_depth_check - instr_depth)
    # Calculate percent difference in relation to total water depth
    if (abs_difference / water_depth * 100) > 0.05:
        warnings.warn("Difference between calculated instrument depth and metadata "
                      "instrument_depth exceeds 0.05% of the total water depth",
                      UserWarning)
    return


def coordsystem_2enu(vel_var, fixed_leader_var, meta_dict: dict):
    """
    Transforms beam and xyz coordinates to enu coordinates
    vel_var: "vel" variable created from BBWHOS class object, using the command:
    data.read(varlist=['vel'])
    fixed_leader_var: "fixed_leader" variable created from BBWHOS class object,
    using the command: data.read(varlist=['FixedLeader'])
    metadata_dict: dictionary object of metadata items
    UHDAS transform functions use a three-beam solution by faking a fourth beam
    """

    if vel_var.trans.coordsystem == 'beam':
        trans = transform.Transform(angle=fixed_leader_var.sysconfig['angle'],
                                    geometry=meta_dict['beam_pattern'])  # angle is beam angle
        xyze = trans.beam_to_xyz(vel_var.vel.data)
        # print(np.shape(xyze))
        enu = transform.rdi_xyz_enu(xyze, vel_var.heading, vel_var.pitch, vel_var.roll,
                                    orientation=meta_dict['orientation'])
    elif vel_var.trans.coordsystem == 'xyz':
        # print(np.shape(vel_var.vel.data))
        enu = transform.rdi_xyz_enu(vel_var.vel.data, vel_var.heading, vel_var.pitch, vel_var.roll,
                                    orientation=meta_dict['orientation'])
        # print(np.shape(enu))
    else:
        ValueError('vel.trans.coordsystem value of {} not recognized. Conversion to enu not available.'.format(
            vel_var.trans.coordsystem))

    # print(np.shape(enu))
    # Apply change in coordinates to velocities
    velocity1 = enu[:, :, 0]
    velocity2 = enu[:, :, 1]
    velocity3 = enu[:, :, 2]
    velocity4 = enu[:, :, 3]

    # Make note in processing_history
    meta_dict['processing_history'] += " The coordinate system was transformed from {} to enu.".format(
        vel_var.trans.coordsystem
    )

    vel_var.trans.coordsystem = 'enu'
    meta_dict['coord_system'] = 'enu'  # Add item to metadata dictionary for coordinate system

    return velocity1, velocity2, velocity3, velocity4


def flag_pressure(pres, meta_dict: dict):
    """
    pres: pressure variable; array type
    ens1: number of leading bad ensembles from before instrument deployment; int type
    ens2: number of trailing bad ensembles from after instrument deployment; int type
    metadata_dict: dictionary object of metadata items
    """
    # 'no_quality_control': 0
    PRESPR01_QC_var = np.zeros(shape=pres.shape, dtype='float32')

    # Flag negative pressure values
    PRESPR01_QC_var[pres < 0] = bodc_flag_dict['bad_value']

    # pres[PRESPR01_QC_var == 4] = np.nan

    meta_dict['processing_history'] += " Any negative pressure values were flagged."

    return PRESPR01_QC_var


def flag_velocity(meta_dict: dict, ens1: int, ens2: int, number_of_cells: int, v1, v2, v3, v5=None):
    """
    Create QC variables containing flag arrays
    meta_dict: dictionary of metadata items
    ens1: number of leading bad ensembles from before instrument deployment; int type
    ens2: number of trailing bad ensembles from after instrument deployment; int type
    number_of_cells: number of bins
    v1: Eastward velocity with magnetic declination applied
    v2: Northward velocity with magnetic declination applied
    v3: Upwards velocity
    v5: Upwards velocity from Sentinel V vertical beam; only for Sentinel V instruments
    """

    LCEWAP01_QC_var = np.zeros(shape=v1.shape, dtype='float32')
    LCNSAP01_QC_var = np.zeros(shape=v2.shape, dtype='float32')
    LRZAAP01_QC_var = np.zeros(shape=v3.shape, dtype='float32')

    list_qc_vars = [LCEWAP01_QC_var, LCNSAP01_QC_var, LRZAAP01_QC_var]
    if v5 is not None:
        LRZUVP01_QC_var = np.zeros(shape=v5.shape, dtype='float32')
        list_qc_vars.append(LRZUVP01_QC_var)

    for qc in list_qc_vars:
        # 0=no_quality_control, 4=value_seems_erroneous
        for bin_num in range(number_of_cells):
            qc[:ens1, bin_num] = bodc_flag_dict['bad_value']
            if ens2 != 0:
                # if ens2==0, the slice [-0:] would index the whole array
                qc[-ens2:, bin_num] = bodc_flag_dict['bad_value']

    # Update metadata dict
    meta_dict['processing_history'] += (f' The leading {ens1} and trailing {ens2} velocity ensembles were flagged '
                                        f'based on the SeaDataNet flag scheme from BODC.')

    # # Apply the flags to the data and set bad data to NAs
    # # print(v1.shape, LCEWAP01_QC_var.shape)
    # # print(v1)
    # # print(LCEWAP01_QC_var)
    # v1[LCEWAP01_QC_var == 4] = np.nan
    # v2[LCNSAP01_QC_var == 4] = np.nan
    # v3[LRZAAP01_QC_var == 4] = np.nan
    # # print('Set u, v, w to nans')
    # # Vertical beam velocity flagging for Sentinel V's
    # if v5 is None:
    #     return LCEWAP01_QC_var, LCNSAP01_QC_var, LRZAAP01_QC_var
    # else:
    #     LRZUVP01_QC_var = np.zeros(shape=v5.shape, dtype='float32')
    #     for bin_num in range(number_of_cells):
    #         LRZUVP01_QC_var[:ens1, bin_num] = 4
    #         if ens2 != 0:
    #             LRZUVP01_QC_var[-ens2:, bin_num] = 4

    if v5 is not None:
        return LCEWAP01_QC_var, LCNSAP01_QC_var, LRZAAP01_QC_var, LRZUVP01_QC_var
    else:
        return LCEWAP01_QC_var, LCNSAP01_QC_var, LRZAAP01_QC_var


def read_yml_to_dict(yml_file: str):
    # Read yml file into a list of dictionaries
    with open(yml_file, 'r') as stream:
        yaml = YAML(typ='safe')
        d = yaml.load(stream)

    var_names = [elem['id'] for elem in d]

    var_dict = {key: value for key, value in zip(var_names, d)}

    # Longer text that runs multiple lines is read in as a list
    # with one string element, and the string element ends with a '\n'
    for k1 in var_dict.keys():
        for k2 in var_dict[k1].keys():
            if type(var_dict[k1][k2]) == list:
                if len(var_dict[k1][k2]) == 1:
                    var_dict[k1][k2] = var_dict[k1][k2][0].replace('\n', '')
                else:
                    warnings.warn(
                        'Check YAML file content parsing to list\n' + str(var_dict[k1][k2])
                    )

    for k1 in var_dict.keys():
        var_dict[k1].pop('id')

    return var_dict


def add_attrs_2vars_L1(out_obj: xr.Dataset, meta_dict: dict, sensor_depth,
                       pg_flag=0, vb_flag=0, vb_pg_flag=0):
    """
    out_obj: dataset object produced using the xarray package that will be exported as a netCDF file
    metadata_dict: dictionary object of metadata items
    sensor_depth: sensor depth recorded by instrument
    """
    uvw_vel_min = -1000
    uvw_vel_max = 1000

    yml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'adcp_var_string_attrs.yml')

    var_dict = read_yml_to_dict(yml_file)
    for VAR in var_dict.keys():
        if hasattr(out_obj, VAR):  # Accounts for pg and vb flags
            for att in var_dict[VAR].keys():
                if att in ['dtype', 'calendar']:
                    out_obj[VAR].encoding[att] = var_dict[VAR][att]
                else:
                    out_obj[VAR].attrs[att] = var_dict[VAR][att]

    # Add the rest of the attrs to each variable

    # Time
    var = out_obj.time
    var.encoding['units'] = "seconds since 1970-01-01T00:00:00Z"
    var.encoding['_FillValue'] = _FillValue

    # Bin distances
    var = out_obj.distance
    var.encoding['_FillValue'] = _FillValue  # None
    # var.attrs['units'] = "m"
    var.attrs['positive'] = 'up' if meta_dict['orientation'] == 'up' else 'down'

    # LCEWAP01: eastward velocity (vel1)
    # all velocities have many of the same attribute values, but not all, so each velocity is done separately
    var = out_obj.LCEWAP01
    # var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_max'] = np.nanmax(var.data)
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['valid_max'] = uvw_vel_max
    var.attrs['valid_min'] = uvw_vel_min

    # LCNSAP01: northward velocity (vel2)
    var = out_obj.LCNSAP01
    # var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_max'] = np.nanmax(var.data)
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['valid_max'] = uvw_vel_max
    var.attrs['valid_min'] = uvw_vel_min

    # LRZAAP01: vertical velocity (vel3)
    var = out_obj.LRZAAP01
    # var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_max'] = np.nanmax(var.data)
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['valid_max'] = uvw_vel_max
    var.attrs['valid_min'] = uvw_vel_min

    # LERRAP01: error velocity (vel4)
    var = out_obj.LERRAP01
    # var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_max'] = np.nanmax(var.data)
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['valid_max'] = 2 * uvw_vel_max
    var.attrs['valid_min'] = 2 * uvw_vel_min

    # Velocity variable quality flags
    var = out_obj.LCEWAP01_QC
    # var.encoding['dtype'] = 'int'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['data_max'] = np.max(var.data)
    var.attrs['data_min'] = np.min(var.data)

    var = out_obj.LCNSAP01_QC
    # var.encoding['dtype'] = 'int'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['data_max'] = np.max(var.data)
    var.attrs['data_min'] = np.min(var.data)

    var = out_obj.LRZAAP01_QC
    # var.encoding['dtype'] = 'int'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['data_max'] = np.max(var.data)
    var.attrs['data_min'] = np.min(var.data)

    # ELTMEP01: seconds since 1970
    var = out_obj.ELTMEP01
    var.encoding['units'] = 'seconds since 1970-01-01T00:00:00Z'
    var.attrs['_FillValue'] = _FillValue

    # TNIHCE01-4: echo intensity beam 1-4
    var = out_obj.TNIHCE01
    # var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.TNIHCE02
    # var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.TNIHCE03
    # var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.TNIHCE04
    # var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # PCGDAP00 - 4: percent good beam 1-4
    if pg_flag == 1:
        var = out_obj.PCGDAP00
        # var.attrs['units'] = 'percent'
        var.attrs['_FillValue'] = _FillValue
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = meta_dict['serial_number']
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

        var = out_obj.PCGDAP02
        # var.attrs['units'] = 'percent'
        var.attrs['_FillValue'] = _FillValue
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = meta_dict['serial_number']
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

        var = out_obj.PCGDAP03
        # var.attrs['units'] = 'percent'
        var.attrs['_FillValue'] = _FillValue
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = meta_dict['serial_number']
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

        var = out_obj.PCGDAP04
        # var.attrs['units'] = 'percent'
        var.attrs['_FillValue'] = _FillValue
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = meta_dict['serial_number']
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

    # PTCHGP01: pitch
    var = out_obj.PTCHGP01
    # var.attrs['units'] = 'degree'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # ROLLGP01: roll
    var = out_obj.ROLLGP01
    # var.attrs['units'] = 'degree'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # DISTTRAN: height of sea surface (hght)
    var = out_obj.DISTTRAN
    # var.attrs['units'] = 'm'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # TEMPPR01: transducer temp
    var = out_obj.TEMPPR01
    # var.attrs['units'] = 'degree_C'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # PPSAADCP: instrument depth (formerly DEPFP01)
    var = out_obj.PPSAADCP
    # var.attrs['units'] = 'm'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['bin_size'] = meta_dict['cell_size']  # bin size
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # ALONZZ01, longitude
    for var in [out_obj.ALONZZ01, out_obj.longitude]:
        var.encoding['_FillValue'] = _FillValue  # None
        # var.attrs['units'] = 'degrees_east'

    # ALATZZ01, latitude
    for var in [out_obj.ALATZZ01, out_obj.latitude]:
        var.encoding['_FillValue'] = _FillValue  # None
        # var.attrs['units'] = 'degrees_north'

    # HEADCM01: heading
    var = out_obj.HEADCM01
    # var.attrs['units'] = 'degree'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # PRESPR01: pressure
    var = out_obj.PRESPR01
    # var.attrs['units'] = 'dbar'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # PRESPR01_QC: pressure quality flag
    var = out_obj.PRESPR01_QC
    var.attrs['_FillValue'] = _FillValue
    var.attrs['data_max'] = np.nanmax(var.data)
    var.attrs['data_min'] = np.nanmin(var.data)

    # SVELCV01: sound velocity
    var = out_obj.SVELCV01
    # var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # # DTUT8601: time values as ISO8601 string, YY-MM-DD hh:mm:ss
    # var = out_obj.DTUT8601

    # CMAGZZ01-4: correlation magnitude
    var = out_obj.CMAGZZ01
    # var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.CMAGZZ02
    # var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.CMAGZZ03
    # var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    var = out_obj.CMAGZZ04
    # var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = _FillValue
    var.attrs['sensor_depth'] = sensor_depth
    var.attrs['serial_number'] = meta_dict['serial_number']
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)
    # done variables

    # Add Vertical Beam variable attrs for Sentinel V instruments
    if vb_flag == 1:
        var = out_obj.LRZUVP01
        # var.attrs['units'] = 'm s-1'
        var.attrs['_FillValue'] = _FillValue
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = meta_dict['serial_number']
        var.attrs['data_max'] = np.nanmax(var.data)
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['valid_max'] = uvw_vel_max
        var.attrs['valid_min'] = uvw_vel_min

        var = out_obj.LRZUVP01_QC
        var.attrs['_FillValue'] = _FillValue

        var.attrs['data_max'] = np.max(var.data)
        var.attrs['data_min'] = np.min(var.data)

        var = out_obj.TNIHCE05
        # var.attrs['units'] = 'counts'
        var.attrs['_FillValue'] = _FillValue
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = meta_dict['serial_number']
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

        var = out_obj.CMAGZZ05
        # var.attrs['units'] = 'counts'
        var.attrs['_FillValue'] = _FillValue
        var.attrs['sensor_depth'] = sensor_depth
        var.attrs['serial_number'] = meta_dict['serial_number']
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

        if vb_pg_flag == 1:
            var = out_obj.PCGDAP05
            # var.attrs['units'] = 'percent'
            var.attrs['_FillValue'] = _FillValue
            var.attrs['sensor_depth'] = sensor_depth
            var.attrs['serial_number'] = meta_dict['serial_number']
            var.attrs['data_min'] = np.nanmin(var.data)
            var.attrs['data_max'] = np.nanmax(var.data)

    return


def create_meta_dict_L1(adcp_meta: str) -> dict:
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
        next(csv_reader, None)  # Skip header row
        for row in csv_reader:
            # extract all metadata from csv file into dictionary
            # some items not passed to netCDF file but are extracted anyway
            if row[0] == '' and row[1] == '':
                warnings.warn('Metadata file contains a blank row; skipping this row !')
            elif row[0] != '' and row[1] == '':
                warnings.warn('Metadata item in csv file has blank value; skipping this row '
                              'in metadata file !')
            else:
                meta_dict[row[0]] = row[1]

    # Convert numeric values to numerics

    meta_dict['country_institute_code'] = int(meta_dict['country_institute_code'])

    for key in ['instrument_depth', 'latitude', 'longitude', 'water_depth',
                'magnetic_variation']:
        # Limit especially the lon and lat to 5 decimal places
        meta_dict[key] = np.round(float(meta_dict[key]), 5)

    # Use Geojson definitions for IOS
    meta_dict['geographic_area'] = find_geographic_area_attr(
        lon=meta_dict['longitude'], lat=meta_dict['latitude']
    )

    # Assign model, model_long name, and manufacturer
    if meta_dict['instrument_subtype'].upper().replace(' ', '') in ["WORKHORSE", "LONGRANGER"]:
        # pycurrents does not recognize long rangers as distinct from workhorses
        meta_dict['model'] = "wh"
        meta_dict['manufacturer'] = 'Teledyne RDI'
    elif meta_dict["instrument_subtype"].upper() == "BROADBAND":
        meta_dict['model'] = "bb"
        meta_dict['manufacturer'] = 'Teledyne RDI'
    elif meta_dict["instrument_subtype"].upper() == "SENTINEL V":
        meta_dict['model'] = "sv"
        meta_dict['manufacturer'] = 'Teledyne RDI'
    else:
        ValueError('meta_dict["instrumentSubtype"] not understood:',
                   meta_dict["instrumentSubtype"])

    # Add leading zero to serial numbers that have 3 digits
    if len(str(meta_dict['serial_number'])) == 3:
        meta_dict['serial_number'] = '0' + str(meta_dict['serial_number'])

    # Overwrite serial number to include the model: upper returns uppercase
    meta_dict['serial_number'] = meta_dict['model'].upper() + meta_dict['serial_number']

    return meta_dict


def update_meta_dict_L1(meta_dict: dict, data: rdiraw.FileBBWHOS,
                        fixed_leader: rdiraw.Bunch) -> dict:
    """
    Update metadata dictionary with information from the raw data file
    """
    # Correct the long ranger model after reading in the raw data with pycurrents
    if data.sysconfig['kHz'] == 75 and meta_dict['model'] == 'wh':
        meta_dict['model'] = 'lr'
        meta_dict['instrument_subtype'] = 'Long Ranger'

    # Add instrument model variable value
    meta_dict['instrument_model'] = 'RDI {} ADCP {}kHz ({})'.format(
        meta_dict['model'].upper(), data.sysconfig['kHz'], meta_dict['serial_number'])

    # Extract metadata from data object

    # Orientation code from Eric Firing
    # Orientation values such as 65535 and 231 cause SysCfg().up to generate an
    # IndexError: list index out of range
    try:
        # list of bools; True if upward facing, False if down
        orientations = [rdiraw.SysCfg(fl).up for fl in fixed_leader.raw.FixedLeader['SysCfg']]
        meta_dict['orientation'] = mean_orientation(orientations)
    except IndexError:
        warnings.warn('Orientation obtained from data.sysconfig[\'up\'] to avoid '
                      'IndexError: list index out of range', UserWarning)
        meta_dict['orientation'] = 'up' if data.sysconfig['up'] else 'down'

    # Retrieve beam pattern, which is needed for coordsystem_2enu() transform
    if data.sysconfig['convex']:
        meta_dict['beam_pattern'] = 'convex'
    else:
        meta_dict['beam_pattern'] = 'concave'

    meta_dict['cell_size'] = data.CellSize  # Written to global attributes later

    # Begin writing processing history, which will be added as a global attribute to the output netCDF file
    meta_dict['processing_history'] = "Metadata read in from CSV file."

    return meta_dict


def find_geographic_area_attr(lon: float, lat: float):
    # Geojson definitions for IOS
    json_file = 'ios_polygons.geojson'
    json_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), json_file)
    # json_file = os.path.realpath(json_file)
    polygons_dict = utils.read_geojson(json_file)
    return utils.find_geographic_area(polygons_dict, Point(lon, lat))


def nc_create_L1(inFile, file_meta, dest_dir, time_file=None, verbose=False):
    """About:
    Perform level 1 processing on a raw ADCP file and export it as a netCDF file
    :param inFile: full file name of raw ADCP file
    :param file_meta: full file name of csv metadata file associated with inFile
    :param dest_dir: string type; name of folder in which files will be output
    :param time_file: full file name of csv file containing user-generated time data;
                required if inFile has garbled out-of-range time data
    :param verbose: If True then print out progress statements
    """

    # If your raw file has time values out of range, you must also use the
    # create_nc_L1() time_file optional kwarg
    # Use the time_file kwarg to read in a csv file containing time entries
    # spanning the range of deployment and using the
    # instrument sampling interval

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Splice file name to get output netCDF file name
    out_name = os.path.basename(inFile)[:-4] + '.adcp.L1.nc'

    if verbose:
        print(out_name)

    if not dest_dir.endswith('/') or not dest_dir.endswith('\\'):
        out_absolute_name = os.path.abspath(dest_dir + '/' + out_name)
    else:
        out_absolute_name = os.path.abspath(dest_dir + out_name)

    # Read information from metadata file into a dictionary, called meta_dict
    meta_dict = create_meta_dict_L1(file_meta)

    if verbose:
        print('Read in csv metadata file')

    # ------------------------Read in data and start processing--------------------

    # Read in raw ADCP file and model type
    data = rdiraw.rawfile(inFile, meta_dict['model'], trim=True)

    if verbose:
        print('Read in raw data')

    # Extract multidimensional variables from data object: 
    # fixed leader, velocity, amplitude intensity, correlation magnitude, and
    # percent good
    fixed_leader = data.read(varlist=['FixedLeader'])
    vel = data.read(varlist=['Velocity'])
    amp = data.read(varlist=['Intensity'])
    cor = data.read(varlist=['Correlation'])
    pg = data.read(varlist=['PercentGood'])

    # Create flags if pg (percent good) data or
    # vb_pg (vertical beam percent good) data are *missing*
    # flag_pg = 0
    # flag_vb = 0
    # flag_vb_pg = 0
    # try:
    #     # Create throwaway test variable to test for variable availability
    #     test_var = pg.pg1.data[:5]
    # except AttributeError:
    #     flag_pg += 1
    # --------Create flags if variables are *present* NOT missing
    flag_pg = 1 if 'pg1' in pg.keys() else 0
    flag_vb = 0
    flag_vb_pg = 0

    # If model == Sentinel V, read in vertical beam data
    if meta_dict['model'] == 'sv':
        # vb_leader = data.read(varlist=['VBLeader'])
        vb_vel = data.read(varlist=['VBVelocity'])
        vb_amp = data.read(varlist=['VBIntensity'])
        vb_cor = data.read(varlist=['VBCorrelation'])
        vb_pg = data.read(varlist=['VBPercentGood'])

        # Test for missing Sentinel V vertical beam data; if true treat file as
        # regular 4-beam file
        if 'vbvel' in vb_vel.keys():
            flag_vb += 1
            # try:
        #     # Vertical beam velocity data also available from vb_vel.raw.VBVelocity
        #     # but it's multiplied by 1e3 (to make int type)
        #     test_var = vb_vel.vbvel.data[:5]
        # except AttributeError:
        #     flag_vb += 1
        # Test for missing vertical beam percent good data
        try:
            test_var = vb_pg.raw.VBPercentGood[:5]
            flag_vb_pg += 1
        except AttributeError:
            pass

    # Metadata value corrections using info from the raw data file
    meta_dict = update_meta_dict_L1(meta_dict, data, fixed_leader)

    # --------------------Set up dimensions and variables-----------------

    # Initialize dictionary to hold all the variables
    var_dict = {}

    # Time dimension
    var_dict['time'] = convert_time_var(
        time_var=vel.dday, number_of_profiles=data.nprofs, meta_dict=meta_dict,
        origin_year=data.yearbase, time_csv=time_file
    )

    # Distance dimension
    var_dict['distance'] = np.round(vel.dep.data, decimals=2)

    # Amplitude
    var_dict['TNIHCE01'] = amp.amp1.transpose()
    var_dict['TNIHCE02'] = amp.amp2.transpose()
    var_dict['TNIHCE03'] = amp.amp3.transpose()
    var_dict['TNIHCE04'] = amp.amp4.transpose()
    if flag_vb == 1:
        var_dict['TNIHCE05'] = vb_amp.raw.VBIntensity.transpose()

    # Correlation magnitude
    var_dict['CMAGZZ01'] = cor.cor1.transpose()
    var_dict['CMAGZZ02'] = cor.cor2.transpose()
    var_dict['CMAGZZ03'] = cor.cor3.transpose()
    var_dict['CMAGZZ04'] = cor.cor4.transpose()
    if flag_vb == 1:
        var_dict['CMAGZZ05'] = vb_cor.raw.VBCorrelation.transpose()

    if flag_pg == 1:
        var_dict['PCGDAP00'] = pg.pg1.transpose()
        var_dict['PCGDAP02'] = pg.pg2.transpose()
        var_dict['PCGDAP03'] = pg.pg3.transpose()
        var_dict['PCGDAP04'] = pg.pg4.transpose()

    if flag_vb_pg == 1:
        var_dict['PCGDAP05'] = vb_pg.raw.VBPercentGood.transpose()

    var_dict['ELTMEP01'] = var_dict['time']  # todo remove
    var_dict['TEMPPR01'] = vel.temperature
    var_dict['PTCHGP01'] = vel.pitch
    var_dict['ROLLGP01'] = vel.roll
    var_dict['HEADCM01'] = vel.heading

    # Convert SoundSpeed from int16 to float32
    var_dict['SVELCV01'] = np.float32(vel.VL['SoundSpeed'])

    # Convert pressure
    var_dict['PRESPR01'] = assign_pres(vel_var=vel, meta_dict=meta_dict)

    # Dimensionless vars
    var_dict['ALATZZ01'] = meta_dict['latitude']
    var_dict['ALONZZ01'] = meta_dict['longitude']
    var_dict['latitude'] = meta_dict['latitude']
    var_dict['longitude'] = meta_dict['longitude']
    var_dict['filename'] = out_name[:-3]  # do not include .nc suffix
    var_dict['instrument_serial_number'] = meta_dict['serial_number']
    var_dict['instrument_model'] = meta_dict['instrument_model']
    var_dict['geographic_area'] = meta_dict['geographic_area']

    # ------------------------Adjust velocity data-------------------------

    # Set velocity values of -32768.0 to nans, since -32768.0 is the automatic
    # fill_value for pycurrents
    vel.vel.data[vel.vel.data == vel.vel.fill_value] = _FillValue

    if meta_dict['model'] == 'sv' and flag_vb == 0:
        vb_vel.vbvel.data[vb_vel.vbvel.data == vb_vel.vbvel.fill_value] = _FillValue

    # Rotate into earth if not in enu already; this makes the netCDF bigger
    # For Sentinel V instruments, transformations are done independently of vertical
    # beam velocity data
    if vel.trans.coordsystem not in ['earth', 'enu']:
        old_coordsystem = vel.trans.coordsystem
        vel1, vel2, vel3, vel4 = coordsystem_2enu(
            vel_var=vel, fixed_leader_var=fixed_leader, meta_dict=meta_dict)

        if verbose:
            print('Coordinate system rotated from {} to enu'.format(old_coordsystem))
    else:
        vel1 = vel.vel1.data
        vel2 = vel.vel2.data
        vel3 = vel.vel3.data
        vel4 = vel.vel4.data
        meta_dict['coord_system'] = 'enu'

    # Correct magnetic declination in velocities and round to the input number of decimal places
    LCEWAP01, LCNSAP01 = correct_true_north(vel1, vel2, meta_dict)

    var_dict['LCEWAP01'] = LCEWAP01.transpose()
    var_dict['LCNSAP01'] = LCNSAP01.transpose()
    var_dict['LRZAAP01'] = vel3.transpose()
    var_dict['LERRAP01'] = vel4.transpose()
    if flag_vb == 1:
        var_dict['LRZUVP01'] = vb_vel.vbvel.data.transpose()

    # -----------Truncate time series variables before computing derived variables-----------

    # Set start and end indices
    e1 = int(meta_dict['cut_lead_ensembles'])  # "ensemble 1"
    e2 = int(meta_dict['cut_trail_ensembles'])  # "ensemble 2"

    trunc_func_2d = lambda arr: arr[:, e1:-e2] if e2 != 0 else arr[:, e1:]
    trunc_func_1d = lambda arr: arr[e1:-e2] if e2 != 0 else arr[e1:]

    # Flag measurements from before deployment and after recovery using e1 and e2
    old_time_series_len = len(var_dict['time'])
    for key in var_dict.keys():
        if type(var_dict[key]) == np.ndarray:
            if len(var_dict[key].shape) == 2:
                var_dict[key] = trunc_func_2d(var_dict[key])
            elif len(var_dict[key]) == old_time_series_len:
                var_dict[key] = trunc_func_1d(var_dict[key])

    # todo change {e1} to equivalent of ({} UTC).format(DTUT8601[e1])?
    # some data after deployment and before recovery are also sometimes cut - statements not accurate
    if e1 != 0:
        meta_dict['processing_history'] += f' Leading {e1} ensembles from before deployment discarded.'
    if e2 != 0:
        meta_dict['processing_history'] += f' Trailing {e2} ensembles from after recovery discarded.'

    # --------------------------------Additional flagging--------------------------------

    var_dict['PRESPR01_QC'] = flag_pressure(pres=var_dict['PRESPR01'], meta_dict=meta_dict)

    for velocity in ['LCEWAP01', 'LCNSAP01', 'LRZAAP01', 'LRZUVP01']:
        if velocity in var_dict.keys():
            var_dict[f'{velocity}_QC'] = np.zeros(shape=var_dict[velocity].shape)

    # -----------------------------Compute derived variables------------------------------

    # Apply equivalent of swDepth() to depth data: Calculate height from sea pressure
    # using gsw package. Negative so that depth is positive; units=m
    var_dict['PPSAADCP'] = -np.round(
        gsw.conversions.z_from_p(p=var_dict['PRESPR01'], lat=meta_dict['latitude']), decimals=2
    )

    meta_dict['processing_history'] += (" Time series sea surface height calculated from pressure "
                                        "using the TEOS-10 75-term expression for specific volume.")

    if verbose:
        print('Calculated sea surface height from sea pressure using the gsw package')

    # Check instrument_depth from metadata csv file: compare with pressure values
    check_depths(pres=var_dict['PRESPR01'], dist=var_dict['distance'],
                 instr_depth=meta_dict['instrument_depth'], water_depth=meta_dict['water_depth'])

    # Calculate sensor depth of instrument based off mean instrument depth
    sensor_dep = np.nanmean(var_dict['PPSAADCP'])

    # Calculate height of sea surface
    if meta_dict['orientation'] == 'up':
        var_dict['DISTTRAN'] = np.round(sensor_dep - var_dict['distance'], decimals=2)
    else:
        var_dict['DISTTRAN'] = np.round(sensor_dep + var_dict['distance'], decimals=2)

    # Round sensor_dep
    sensor_dep = np.round(sensor_dep, decimals=2)

    meta_dict['processing_history'] += f" Sensor depth set to the mean of trimmed depth values: {sensor_dep} m."

    if verbose:
        print('Finished QCing data; making netCDF object next')

    # -------------------------Make into netCDF file---------------------------

    # Create xarray Dataset object containing all dimensions and variables
    # Sentinel V instruments don't have percent good ('pg') variables
    # out = xr.Dataset(coords={'time': time_s, 'distance': distance},
    #                  data_vars={'LCEWAP01': (['distance', 'time'], LCEWAP01.transpose()),
    #                             'LCNSAP01': (['distance', 'time'], LCNSAP01.transpose()),
    #                             'LRZAAP01': (['distance', 'time'], vel3.transpose()),
    #                             'LERRAP01': (['distance', 'time'], vel4.transpose()),
    #                             'LCEWAP01_QC': (['distance', 'time'], LCEWAP01_QC.transpose()),
    #                             'LCNSAP01_QC': (['distance', 'time'], LCNSAP01_QC.transpose()),
    #                             'LRZAAP01_QC': (['distance', 'time'], LRZAAP01_QC.transpose()),
    #                             'ELTMEP01': (['time'], time_s),
    #                             'TNIHCE01': (['distance', 'time'], amp.amp1.transpose()),
    #                             'TNIHCE02': (['distance', 'time'], amp.amp2.transpose()),
    #                             'TNIHCE03': (['distance', 'time'], amp.amp3.transpose()),
    #                             'TNIHCE04': (['distance', 'time'], amp.amp4.transpose()),
    #                             'CMAGZZ01': (['distance', 'time'], cor.cor1.transpose()),
    #                             'CMAGZZ02': (['distance', 'time'], cor.cor2.transpose()),
    #                             'CMAGZZ03': (['distance', 'time'], cor.cor3.transpose()),
    #                             'CMAGZZ04': (['distance', 'time'], cor.cor4.transpose()),
    #                             'PTCHGP01': (['time'], vel.pitch),
    #                             'HEADCM01': (['time'], vel.heading),
    #                             'ROLLGP01': (['time'], vel.roll),
    #                             'TEMPPR01': (['time'], vel.temperature),
    #                             'DISTTRAN': (['distance'], DISTTRAN),
    #                             'PPSAADCP': (['time'], depth),
    #                             'ALATZZ01': ([], meta_dict['latitude']),
    #                             'ALONZZ01': ([], meta_dict['longitude']),
    #                             'latitude': ([], meta_dict['latitude']),
    #                             'longitude': ([], meta_dict['longitude']),
    #                             'PRESPR01': (['time'], pressure),
    #                             'PRESPR01_QC': (['time'], PRESPR01_QC),
    #                             'SVELCV01': (['time'], sound_speed),
    #                             'DTUT8601': (['time'], time_DTUT8601),
    #                             'filename': ([], out_name[:-3]),  # do not include .nc suffix
    #                             'instrument_serial_number': ([], meta_dict['serial_number']),
    #                             'instrument_model': ([], meta_dict['instrument_model']),
    #                             'geographic_area': ([], meta_dict['geographic_area'])})

    out = xr.Dataset(coords={'time': var_dict['time'], 'distance': var_dict['distance']})

    variable_order = ['LCEWAP01', 'LCNSAP01', 'LRZAAP01', 'LERRAP01', 'LRZUVP01',
                      'LCEWAP01_QC', 'LCNSAP01_QC', 'LRZAAP01_QC', 'LRZUVP01_QC'
                                                                   'TNIHCE01', 'TNIHCE02', 'TNIHCE03', 'TNIHCE04',
                      'TNIHCE05',
                      'CMAGZZ01', 'CMAGZZ02', 'CMAGZZ03', 'CMAGZZ04', 'CMAGZZ05'
                                                                      'PCGDAP00', 'PCGDAP02', 'PCGDAP03', 'PCGDAP04',
                      'PCGDAP05',
                      'ELTMEP01', 'DISTTRAN', 'PPSAADCP', 'PRESPR01',
                      'ALATZZ01', 'ALONZZ01', 'latitude', 'longitude',
                      'PTCHGP01', 'HEADCM01', 'ROLLGP01',
                      'TEMPPR01', 'SVELCV01',
                      'filename', 'instrument_serial_number', 'instrument_model',
                      'geographic_area']

    for key in variable_order:
        if key in var_dict.keys():
            if len(np.shape(var_dict[key])) == 0:
                out[key] = ((), var_dict[key])
            elif len(np.shape(var_dict[key])) == 1:
                if len(var_dict[key]) == len(var_dict['distance']):
                    out[key] = (('distance'), var_dict[key])
                elif len(var_dict[key]) == len(var_dict['time']):
                    out[key] = (('time'), var_dict[key])
            elif len(np.shape(var_dict[key])) == 2:
                out[key] = (('distance', 'time'), var_dict[key])
            else:
                warnings.warn(f'Shape of variable {key} not compatible')

    if verbose:
        print(out.data_vars)  # Check that all available vars have been added

    # Add variable-specific attributes

    add_attrs_2vars_L1(out_obj=out, meta_dict=meta_dict, sensor_depth=sensor_dep,
                       pg_flag=flag_pg, vb_flag=flag_vb, vb_pg_flag=flag_vb_pg)

    # ----------------------Global attributes----------------------

    # Add select meta_dict items as global attributes
    pass_dict_keys = ['cut_lead_ensembles', 'cut_trail_ensembles', 'processing_level', 'model']
    for key, value in meta_dict.items():
        if key not in pass_dict_keys:  # Exclude certain items in the dictionary
            out.attrs[key] = value

    # Rest of attributes not from metadata file:
    out.attrs['time_coverage_duration'] = vel.dday[-1] - vel.dday[0]
    out.attrs['time_coverage_duration_units'] = "days"
    # ^calculated from start and end times; in days: add time_coverage_duration_units?
    out.attrs['cdm_data_type'] = "station"
    out.attrs['number_of_beams'] = data.NBeams
    # out.attrs['nprofs'] = data.nprofs #number of ensembles
    out.attrs['number_of_cells'] = data.NCells
    out.attrs['pings_per_ensemble'] = data.NPings
    out.attrs['bin_1_distance'] = data.Bin1Dist
    out.attrs['cell_size'] = data.CellSize
    out.attrs['ping_type'] = data.pingtype
    out.attrs['transmit_pulse_length_cm'] = vel.FL['Pulse']
    out.attrs['instrument_type'] = "adcp"
    out.attrs['manufacturer'] = meta_dict['manufacturer']
    out.attrs['source'] = "Python code: github: pycurrents_ADCP_processing"
    now = datetime.datetime.now()
    out.attrs['date_modified'] = now.strftime("%Y-%m-%d %H:%M:%S")
    out.attrs['_FillValue'] = _FillValue  # str(fill_value)
    out.attrs['feature_type'] = "profileTimeSeries"
    out.attrs['firmware_version'] = str(vel.FL.FWV) + '.' + str(vel.FL.FWR)  # firmwareVersion
    out.attrs['frequency'] = str(data.sysconfig['kHz'])
    out.attrs['beam_angle'] = str(fixed_leader.sysconfig['angle'])  # beamAngle
    out.attrs['system_configuration'] = bin(fixed_leader.FL['SysCfg'])[-8:] + '-' + bin(fixed_leader.FL['SysCfg'])[
                                                                                    :9].replace('b', '')
    out.attrs['sensor_source'] = '{0:08b}'.format(vel.FL['EZ'])  # sensorSource
    out.attrs['sensors_avail'] = '{0:08b}'.format(vel.FL['SA'])  # sensors_avail
    out.attrs['three_beam_used'] = str(vel.trans['threebeam']).upper()  # netCDF4 file format doesn't support bool
    out.attrs['valid_correlation_range'] = vel.FL['LowCorrThresh']  # lowCorrThresh
    out.attrs['min_percent_good'] = fixed_leader.FL['PGMin']
    out.attrs['blank'] = '{} m'.format(fixed_leader.FL['Blank'] / 100)  # convert cm to m
    out.attrs['error_velocity_threshold'] = "{} mm s-1".format(fixed_leader.FL['EVMax'])
    tpp_min = '{0:0>2}'.format(fixed_leader.FL['TPP_min'])
    tpp_sec = '{0:0>2}'.format(fixed_leader.FL['TPP_sec'])
    tpp_hun = '{0:0>2}'.format(fixed_leader.FL['TPP_hun'])
    out.attrs['time_ping'] = '{}:{}.{}'.format(tpp_min, tpp_sec, tpp_hun)
    out.attrs['false_target_reject_values'] = '{} counts'.format(fixed_leader.FL['WA'])  # falseTargetThresh
    out.attrs['data_type'] = "adcp"
    # out.attrs['pred_accuracy'] = 1  # velocityResolution * 1000
    out.attrs['creator_type'] = "person"
    out.attrs['n_codereps'] = vel.FL.NCodeReps
    out.attrs['xmit_lag'] = vel.FL.TransLag
    out.attrs['xmit_length'] = fixed_leader.FL['Pulse']
    out.attrs['time_coverage_start'] = numpy_datetime_to_str(var_dict['time'][e1]) + ' UTC'
    out.attrs['time_coverage_end'] = numpy_datetime_to_str(var_dict['time'][-e2 - 1]) + ' UTC'

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


def example_L1_1():
    """
    Specify raw ADCP file to create nc file from, along with associated csv metadata file
    """
    # raw .000 file
    raw_file = "./sample_data/a1_20050503_20050504_0221m.000"
    # csv metadata file
    raw_file_meta = "./sample_data/a1_20050503_20050504_0221m_meta_L1.csv"

    dest_dir = 'dest_dir'

    # Create netCDF file
    nc_name = nc_create_L1(inFile=raw_file, file_meta=raw_file_meta, dest_dir=dest_dir)

    # # DEPRECProduce new netCDF file that includes a geographic_area variable
    # geo_name = add_var2nc.add_geo(nc_name, dest_dir)

    # return nc_name, geo_name
    return nc_name


def example_L1_2():
    """Specify raw ADCP file to create nc file from, along with associated csv metadata file
    AND time file"""

    # raw .000 file
    raw_file = "./sample_data/scott2_20160711_20170707_0040m.pd0"
    # csv metadata file
    raw_file_meta = "./sample_data/scott2_20160711_20170707_0040m_meta_L1.csv"
    # csv time file
    scott_time = './sample_data/scott2_20160711_20170707_0040m_time.csv'

    dest_dir = 'dest_dir'

    # Create netCDF file
    nc_name = nc_create_L1(inFile=raw_file, file_meta=raw_file_meta, dest_dir=dest_dir, time_file=scott_time)

    # # DEPRECProduce new netCDF file that includes a geographic_area variable
    # geo_name = add_var2nc.add_geo(nc_name, dest_dir)

    # return nc_name, geo_name
    return nc_name
