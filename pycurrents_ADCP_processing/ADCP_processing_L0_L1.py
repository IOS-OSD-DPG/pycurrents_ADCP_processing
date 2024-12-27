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
from pycurrents_ADCP_processing import plot_westcoast_nc_LX as pw
from shapely.geometry import Point

# from datetime import datetime, timezone

_FillValue = np.nan

BODC_FLAG_DICT = {
    'no_quality_control': 0, 'good_value': 1, 'probably_good_value': 2, 'probably_bad_value': 3,
    'bad_value': 4, 'changed_value': 5, 'value_below_detection': 6, 'value_in_excess': 7,
    'interpolated_value': 8, 'missing_value': 9
}

# 2023-12-11 removed 'ELTMEP01', 'ALATZZ01', 'ALONZZ01'
VARIABLE_ORDER = [
    'LCEWAP01', 'LCNSAP01', 'VEL_MAGNETIC_EAST', 'VEL_MAGNETIC_NORTH',
    'LRZAAP01', 'LERRAP01', 'LRZUVP01',
    'LCEWAP01_QC', 'LCNSAP01_QC', 'LRZAAP01_QC', 'LRZUVP01_QC',
    'TNIHCE01', 'TNIHCE02', 'TNIHCE03', 'TNIHCE04', 'TNIHCE05',
    'CMAGZZ01', 'CMAGZZ02', 'CMAGZZ03', 'CMAGZZ04', 'CMAGZZ05',
    'PCGDAP00', 'PCGDAP02', 'PCGDAP03', 'PCGDAP04', 'PCGDAP05',
    'DISTTRAN', 'PPSAADCP', 'PRESPR01', 'PRESPR01_QC',
    'latitude', 'longitude',
    'PTCHGP01', 'HEADCM01', 'ROLLGP01', 'TEMPPR01', 'SVELCV01',
    'filename', 'instrument_serial_number', 'instrument_model',
    'instrument_depth', 'water_depth', 'geographic_area'
]


def correct_true_north(measured_east, measured_north, meta_dict: dict, num_decimals: int = 3):
    # change angle to negative of itself
    # Di Wan's magnetic declination correction code: Takes 0 DEG from E-W axis
    # mag_decl: magnetic declination value; float type
    # measured_east: measured Eastward velocity data; array type
    # measured_north: measured Northward velocity data; array type
    angle_rad = -meta_dict['magnetic_variation'] * np.pi / 180.
    east_true = measured_east * np.cos(angle_rad) - measured_north * np.sin(angle_rad)
    north_true = measured_east * np.sin(angle_rad) + measured_north * np.cos(angle_rad)

    # Round to the input number of decimal places
    east_true = np.round(east_true, decimals=num_decimals)
    north_true = np.round(north_true, decimals=num_decimals)

    meta_dict['processing_history'] += (" Average magnetic declination applied to eastward and northward "
                                        "velocities; declination = {}.").format(meta_dict['magnetic_variation'])

    return east_true, north_true


def convert_time_var(time_var, number_of_profiles, meta_dict: dict, origin_year: int,
                     time_csv=None) -> np.ndarray:
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
        t_s = pd.to_datetime(time_var, unit='D', origin=data_origin, utc=True)
        # t_s = np.array(
        #     pd.to_datetime(time_var, unit='D', origin=data_origin, utc=True).strftime('%Y-%m-%d %H:%M:%S'),
        #     dtype='datetime64[ns]'
        # )
        # # DTUT8601 variable: time strings
        # t_DTUT8601 = pd.to_datetime(time_var, unit='D', origin=data_origin,
        #                             utc=True).strftime(
        #     '%Y-%m-%d %H:%M:%S')  # don't need %Z in strftime
    except OutOfBoundsDatetime or OverflowError:
        warnings.warn('OutOfBoundsDateTime exception triggered by raw time data', UserWarning)
        t_s = np.zeros(shape=number_of_profiles, dtype='datetime64[ns]')
        # t_DTUT8601 = np.empty(shape=number_of_profiles, dtype='<U100')
        if type(time_csv) is str:
            with open(time_csv) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                # Skip headers
                next(csv_reader, None)
                for count, row in enumerate(csv_reader):
                    if row[0] == '':
                        pass
                    else:
                        t_s[count] = np.datetime64(
                            pd.to_datetime(row[0], utc=True).strftime('%Y-%m-%d %H:%M:%S')
                        )
                        # t_DTUT8601[count] = pd.to_datetime(
                        #     row[0], utc=True).strftime('%Y-%m-%d %H:%M:%S')
        else:
            warnings.warn(f'User did not provide valid time_csv with value {time_csv}; '
                          f'attempting to generate replacement time data. To generate time data yourself, '
                          f'see /pycurrents_ADCP_processing/generate_time_range.py.', UserWarning)

            # Median is robust to outliers
            # Round frequency to nearest second
            median_period = pd.Timedelta(
                np.nanmedian(np.diff(time_var)), unit='day'
            ).round('s').total_seconds()
            # List of accepted units here (use secondly):
            # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
            median_period = f'{utils.round_to_int(median_period)}S'  # convert to string

            # Assume the first timestamp is correct
            date_range = pd.date_range(
                start=pd.to_datetime(
                    time_var[0], unit='D', origin=data_origin, utc=True
                ).strftime('%Y-%m-%d %H:%M:%S'),
                periods=len(time_var),
                freq=median_period
            )

            t_s = np.array([np.datetime64(t) for t in date_range])

        meta_dict['processing_history'] += (' OutOfBoundsDateTime exception triggered by raw time data; '
                                            'generated time range with Python to use as time data.')

    # Set times out of range to NaNs
    # Get timedelta object with value of one year long
    # Use this duration because ADCPs aren't moored for over 2 years (sometimes just over one year)
    yr_value = pd.Timedelta(365 * 2, unit='D')

    try:
        # Works for pandas 2.x
        time_median = pd.Series(t_s).median()
    except TypeError:
        # Works for pandas 1.x
        time_median = pd.to_datetime(pd.Series(t_s).astype('int64').median(), utc=True)

    try:
        indexer_out_rng = np.where(t_s > time_median + yr_value)[0]
    except TypeError:
        indexer_out_rng = np.where(
            pd.to_datetime(t_s, utc=True) > time_median + yr_value
        )[0]

    # Some silly little conversions
    t_df = pd.Series(t_s)
    t_df.loc[indexer_out_rng] = pd.NaT
    t_s = np.array(t_df, dtype='datetime64[ns]')

    # t_s[indexer_out_rng] = pd.NaT
    # dtut_df = pd.Series(t_DTUT8601)
    # dtut_df.loc[indexer_out_rng] = 'NaN'  # String type
    # t_DTUT8601 = np.array(dtut_df)

    return t_s  # , t_DTUT8601


def assign_pressure(vel_var, meta_dict: dict, level):
    """
    Assign pressure and calculate it froms static instrument depth if ADCP
    missing pressure sensor
    vel_var: "vel" variable created from BBWHOS class object, using the
        command, data.read(varlist=['vel'])
    meta_dict: dictionary object of metadata items
    level: processing level 0 or 1
    """

    static_pressure_flag = 0

    if meta_dict['model'] in ['wh', 'lr', 'sv']:
        # convert decapascal to decibars
        pres = np.array(vel_var.VL['Pressure'] / 1000, dtype='float32')

        # Calculate pressure based on static instrument depth if missing
        # pressure sensor; extra condition added for zero pressure or weird
        # pressure values
        # Handle no unique modes from statistics.mode(pressure)
        pressure_unique, counts = np.unique(pres, return_counts=True)
        static_pressure_flag += 1 if np.max(counts) > len(pres) / 2 else 0

    # Check if model is type missing pressure sensor or if zero is a mode of pressure
    # Amendment 2023-09-18: change to "if pressure is static for over half the dataset" as the former became a bug
    if meta_dict['model'] == 'bb' or static_pressure_flag == 1:  # or np.max(counts) == counts[index_of_zero]:
        if level == 1:
            p = np.round(
                gsw.conversions.p_from_z(
                    -meta_dict['instrument_depth'],
                    meta_dict['latitude']
                ),
                decimals=1
            )  # depth negative because positive is up for this function
            pres = np.repeat(p, len(vel_var.vel1.data))
            meta_dict['processing_history'] += " Pressure calculated from static instrument depth ({} m) using " \
                                               "the TEOS-10 75-term expression for specific volume.".format(
                meta_dict['instrument_depth'])
            warnings.warn('Pressure values calculated from static instrument depth', UserWarning)
        elif level == 0:
            # Do not calculate pressure for level 0 if it's not available
            warnings.warn('No pressure data available (no field of name Pressure)')
            pres = None

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
    metadata_dict: dictionary object of metadata items
    """
    # 'no_quality_control': 0
    PRESPR01_QC_var = np.zeros(shape=pres.shape, dtype='float32')

    # Flag negative pressure values
    PRESPR01_QC_var[pres < 0] = BODC_FLAG_DICT['bad_value']

    # pres[PRESPR01_QC_var == 4] = np.nan

    meta_dict['processing_history'] += " Any negative pressure values were flagged."

    return PRESPR01_QC_var


def flag_velocity_DEPREC(meta_dict: dict, ens1: int, ens2: int, number_of_cells: int, v1, v2, v3, v5=None):
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
            qc[:ens1, bin_num] = BODC_FLAG_DICT['bad_value']
            if ens2 != 0:
                # if ens2==0, the slice [-0:] would index the whole array
                qc[-ens2:, bin_num] = BODC_FLAG_DICT['bad_value']

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
                        'Check YAML file content parsing to list\n' + str(var_dict[k1][k2]), UserWarning
                    )

    for k1 in var_dict.keys():
        var_dict[k1].pop('id')

    return var_dict


def add_attrs_2vars(out_obj: xr.Dataset, meta_dict: dict):  # sensor_depth, pg_flag=0, vb_flag=0, vb_pg_flag=0
    """
    out_obj: dataset object produced using the xarray package that will be exported as a netCDF file
    metadata_dict: dictionary object of metadata items
    """
    # uvw_vel_min = -1000
    # uvw_vel_max = 1000

    yml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'adcp_var_string_attrs.yml')

    attr_dict = read_yml_to_dict(yml_file)
    for VAR in attr_dict.keys():
        if hasattr(out_obj, VAR):  # Accounts for pg and vb flags
            for att in attr_dict[VAR].keys():
                if att in ['dtype', 'calendar']:
                    out_obj[VAR].encoding[att] = attr_dict[VAR][att]
                else:
                    out_obj[VAR].attrs[att] = attr_dict[VAR][att]

    # Add the rest of the attrs to each variable

    # Time
    var = out_obj.time
    var.encoding['units'] = "seconds since 1970-01-01T00:00:00Z"
    var.encoding['_FillValue'] = _FillValue

    # Bin distances
    var = out_obj.distance
    var.encoding['_FillValue'] = _FillValue  # None
    var.attrs['positive'] = 'up' if meta_dict['orientation'] == 'up' else 'down'

    for var_name in VARIABLE_ORDER:
        if hasattr(out_obj, var_name):
            if var_name == var_name.upper():
                # Add a few more attrs to *select* vars
                out_obj[var_name].attrs['_FillValue'] = _FillValue
                out_obj[var_name].attrs['data_max'] = np.nanmax(out_obj[var_name].data)
                out_obj[var_name].attrs['data_min'] = np.nanmin(out_obj[var_name].data)
            elif var_name in ['latitude', 'longitude']:
                out_obj[var_name].attrs['_FillValue'] = _FillValue

    return


def create_meta_dict(adcp_meta: str, level) -> dict:
    """
    Read in a csv metadata file and output in dictionary format
    Inputs:
        - adcp_meta: csv-format file containing metadata for raw ADCP file
        - level: processing level to apply, 0 or 1 (default 1)
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
                warnings.warn('Metadata file contains a blank row; skipping this row !', UserWarning)
            elif row[0] != '' and row[1] == '':
                warnings.warn(f'Metadata item {row[0]} in csv file has blank value', UserWarning)
                meta_dict[row[0]] = None
            else:
                meta_dict[row[0]] = row[1]

    # Convert numeric values to numerics

    meta_dict['country_institute_code'] = int(meta_dict['country_institute_code'])

    for key in ['instrument_depth', 'latitude', 'longitude', 'water_depth',
                'magnetic_variation', 'recovery_lat', 'recovery_lon']:
        if key in meta_dict:
            if meta_dict[key] is not None:
                meta_dict[key] = np.round(float(meta_dict[key]), 5)
        else:
            meta_dict[key] = None

    meta_dict['processing_level'] = str(level)

    # Use Geojson definitions for IOS
    meta_dict['geographic_area'] = utils.find_geographic_area_attr(
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
        ValueError(f'meta_dict["instrumentSubtype"] not understood: {meta_dict["instrumentSubtype"]}')

    # Edit the serial number if needed
    if meta_dict['serial_number'] is None:
        meta_dict['serial_number'] = 'Unknown'
    elif len(str(meta_dict['serial_number'])) == 3:
        # Add leading zero to serial numbers that have 3 digits
        meta_dict['serial_number'] = '0' + str(meta_dict['serial_number'])

    # Overwrite serial number to include the model: upper returns uppercase
    if meta_dict['serial_number'] != 'Unknown':
        meta_dict['serial_number'] = meta_dict['model'].upper() + meta_dict['serial_number']

    # Begin writing processing history, which will be added as a global attribute to the output netCDF file
    meta_dict['processing_history'] = "Metadata read in from CSV file."

    return meta_dict


def matlab_index_to_python(ind):
    """
    Matlab and R indexing starts at 1 and includes the end
    Python indexing starts at 0 and excludes the end
    """
    return ind - 1


def python_index_to_matlab(ind):
    return ind + 1


def update_meta_dict(meta_dict: dict, data: rdiraw.FileBBWHOS,
                        fixed_leader: rdiraw.Bunch, level) -> dict:
    """
    Update metadata dictionary with information from the raw data file
    """
    # Correct the long ranger model after reading in the raw data with pycurrents
    if data.sysconfig['kHz'] == 75 and meta_dict['model'] == 'wh':
        meta_dict['instrument_subtype'] = 'Workhorse Long Ranger'
        meta_dict['serial_number'] = meta_dict['serial_number'].replace('WH', 'LR')
        meta_dict['instrument_model'] = 'RDI {} Long Ranger ADCP {}kHz'.format(
            meta_dict['model'].upper(), data.sysconfig['kHz']
        )
        warnings.warn('Workhorse ADCP identified as 75 kHz Long Ranger', UserWarning)
    else:
        # Add instrument model variable value
        # meta_dict['instrument_model'] = 'RDI {} ADCP {}kHz ({})'.format(
        #     meta_dict['model'].upper(), data.sysconfig['kHz'], meta_dict['serial_number'])
        meta_dict['instrument_model'] = 'RDI {} ADCP {}kHz'.format(
            meta_dict['model'].upper(), data.sysconfig['kHz']
        )

    # Extract metadata from data object

    # Orientation code from Eric Firing
    # Orientation values such as 65535 and 231 cause SysCfg().up to generate an
    # IndexError: list index out of range
    try:
        # list of bools; True if upward facing, False if down
        orientations = [rdiraw.SysCfg(fl).up for fl in fixed_leader.raw.FixedLeader['SysCfg']]
        meta_dict['orientation'] = utils.mean_orientation(orientations)
    except IndexError:
        warnings.warn('Orientation obtained from data.sysconfig[\'up\'] to avoid '
                      'IndexError: list index out of range', UserWarning)
        meta_dict['orientation'] = 'up' if data.sysconfig['up'] else 'down'

    # Retrieve beam pattern, which is needed for coordsystem_2enu() transform
    if data.sysconfig['convex']:
        meta_dict['beam_pattern'] = 'convex'
    else:
        meta_dict['beam_pattern'] = 'concave'

    if level == 1:
        # Convert segment indices from string/int to list and from matlab start at 1 to python start at 0
        flag_start = 0
        flag_end = 0
        for segment_indices in ['segment_start_indices', 'segment_end_indices']:
            if segment_indices in meta_dict.keys():
                if type(meta_dict[segment_indices]) == str and ',' in meta_dict[segment_indices]:
                    # Multiple segments
                    meta_dict[segment_indices] = [
                        matlab_index_to_python(int(ind))
                        for ind in meta_dict[segment_indices].replace('"', '').split(',')
                    ]
                elif type(meta_dict[segment_indices]) == int:
                    # Replaces numbers of ensembles to cut
                    meta_dict[segment_indices] = [matlab_index_to_python(meta_dict[segment_indices])]
                elif meta_dict[segment_indices] is None:
                    if segment_indices == 'segment_start_indices':
                        flag_start += 1
                    else:
                        flag_end += 1
            else:
                if segment_indices == 'segment_start_indices':
                    flag_start += 1
                else:
                    flag_end += 1

        if flag_start > 0:
            if 'cut_lead_ensembles' in meta_dict.keys() and meta_dict['cut_lead_ensembles'] is not None:
                meta_dict['segment_start_indices'] = [int(meta_dict['cut_lead_ensembles'])]
            else:
                meta_dict['segment_start_indices'] = [0]

        if flag_end > 0:
            if 'cut_trail_ensembles' in meta_dict.keys() and meta_dict['cut_trail_ensembles'] is not None:
                meta_dict['segment_end_indices'] = [
                    matlab_index_to_python(len(fixed_leader['raw']['FixedLeader']) - int(meta_dict['cut_trail_ensembles']))
                ]
            else:
                meta_dict['segment_end_indices'] = [matlab_index_to_python(len(fixed_leader['raw']['FixedLeader']))]

    return meta_dict


def get_segment_start_end_idx_depth_DEPREC(
        segment_starts_ends, time_series_depth: np.ndarray,
        datetime_pd: np.ndarray, original_instr_depth: float
):
    """
    Get start and end indices of time series segments and their corresponding instrument depths.
    Calculate instrument depth from the pressure during subsequent segments using gsw.

    segment_starts_ends: dict or None
    time_series_depth: PPSAADCP
    """

    # Assume the first segment instrument depth was already correct
    # before the mooring was displaced, so pass the original depth here

    # Create flag
    ds_is_split = True

    # Convert the user-input segment starts and ends into usable format
    if type(segment_starts_ends) == dict:
        # Determine the type of the objects in the list
        if all([type(k) == str for k in segment_starts_ends.keys()]):
            # key is a datetime in string format 'YYYYMMDD HH:MM:SS'
            segment_starts_ends = {
                pd.to_datetime(k): pd.to_datetime(v)
                for k, v in segment_starts_ends.items()
            }
            # Convert it to an index (integer type)
            idx = {}
            for k, v in segment_starts_ends.items():
                idx_k = np.where(
                    abs(datetime_pd - pd.to_datetime(k)
                        ) == min(abs(datetime_pd - pd.to_datetime(k)))
                )[0]
                print(idx_k)
                idx_v = np.where(
                    abs(datetime_pd - pd.to_datetime(v)
                        ) == min(abs(datetime_pd - pd.to_datetime(v)))
                )[0]
                idx[idx_k[0]] = idx_v[0]
            segment_start_end_idx = idx
        elif all([type(k) == int for k in segment_starts_ends.keys()]):
            segment_start_end_idx = segment_starts_ends
    elif segment_starts_ends is None:
        segment_start_end_idx = {0: len(datetime_pd) - 1}  # {0: -1}??
        ds_is_split = False
    else:
        ValueError(
            f'Segment start and end items are type {type(segment_starts_ends)} and must be dict or None'
        )

    # Add 1 to the end index because python indexing is not inclusive
    for k, v in segment_start_end_idx.items():
        segment_start_end_idx[k] = v + 1

    # Get the mean instrument depth for each segment
    segment_instrument_depths = np.zeros(
        len(segment_start_end_idx), dtype='float32')

    for i, k, v in zip(range(len(segment_start_end_idx)),
                       segment_start_end_idx.keys(),
                       segment_start_end_idx.values()):
        # Keep original instrument_depth for first segment
        if i == 0:
            segment_instrument_depths[i] = original_instr_depth
        else:
            segment_instrument_depths[i] = np.round(np.nanmean(time_series_depth[k: v]), 1)

    return segment_start_end_idx, segment_instrument_depths, ds_is_split


def get_segment_instrument_depths(
        start_indices, end_indices, time_series_depth: np.ndarray, start_instrument_depth: float
):
    """
    Compute instrument depth from the average time series depth PPSAADCP for each segment
    start_instrument_depth is the depth provided by the user in the metadata csv file
    """
    # Compute new instrument_depth including an offset amount
    depth_correction = np.round(
        np.nanmean(time_series_depth[start_indices[0]: end_indices[0]]) - start_instrument_depth,
        1
    )

    instrument_depths = np.zeros(len(start_indices))
    instrument_depths[0] = start_instrument_depth

    for i in range(1, len(start_indices)):
        instrument_depths[i] = np.round(
            np.nanmean(time_series_depth[start_indices[i]: end_indices[i]]) - depth_correction,
            1
        )

    # Check for any repeating depth values
    if len(np.unique([utils.round_to_int(x) for x in instrument_depths])) < len(instrument_depths):
        warnings.warn(f'Segment depths round to same int value: depths = {instrument_depths}')

    return instrument_depths, depth_correction


def make_dataset_from_subset(
        ds: xr.Dataset, start_idx: int, end_idx: int,
        instrument_depth: float, depth_correction: float, num_segments: int,
        time_of_strike: str, recovery_lat=None, recovery_lon=None
):
    """
    Create an xarray dataset to contain data from a single segment
    """

    # Need to subtract 1 from en_idx because +1 is added in
    # get_user_segment_start_end_idx_depth() for inclusive ranges
    # but ranges are not called here
    new_filename = '{}_{}_{}_{}m_L1.adcp.nc'.format(
        ds.station.lower(),
        utils.numpy_datetime_to_str_utc(ds.time.data[start_idx])[:10].replace('-', ''),
        utils.numpy_datetime_to_str_utc(ds.time.data[end_idx])[:10].replace('-', ''),
        f'000{utils.round_to_int(instrument_depth)}'[-4:]
    )

    # Add variables
    var_dict = {}
    for key in ds.data_vars.keys():
        if key == 'filename':
            var_dict[key] = ([], new_filename)
        elif key == 'latitude' and type(recovery_lat) == float:
            # print(recovery_lat, recovery_lon)
            var_dict[key] = ([], recovery_lat)
        elif key == 'longitude' and type(recovery_lon) == float:
            var_dict[key] = ([], recovery_lon)
        elif 'time' in ds[key].coords:
            if 'distance' in ds[key].coords:
                var_dict[key] = (['distance', 'time'], ds[key].data[:, start_idx:end_idx])
            else:
                var_dict[key] = (['time'], ds[key].data[start_idx:end_idx])
        elif key == 'DISTTRAN':
            sensor_depth = np.nanmean(ds['PPSAADCP'].data[start_idx:end_idx])
            DISTTRAN = compute_sea_surface_height(
                orientation=ds.attrs['orientation'],
                sensor_depth=sensor_depth,
                distance=ds.distance.data,
                meta_dict=None
            )
            var_dict[key] = (['distance'], DISTTRAN)
        elif key == 'instrument_depth':
            var_dict[key] = ([], instrument_depth)  # Update new instrument depth
        else:
            if ds[key].data.dtype == np.dtype('float32'):
                var_dict[key] = ([], np.round(float(ds[key].data), 1))
            else:
                var_dict[key] = ([], str(ds[key].data))

    dsout = xr.Dataset(
        coords={'time': ds.time.data[start_idx:end_idx],
                'distance': ds.distance.data}, data_vars=var_dict
    )

    # Add attributes and encoding back to the coordinates and variables
    for coord in dsout.coords.keys():
        for encoding, encoding_val in ds[coord].encoding.items():
            dsout[coord].encoding[encoding] = encoding_val
        for attr, attr_val in ds[coord].attrs.items():
            dsout[coord].attrs[attr] = attr_val

    for var in dsout.data_vars.keys():
        # try:
        #     dsout[var].encoding['_FillValue'] = ds[var].encoding['_FillValue']
        # except KeyError:
        #     pass
        for encoding, encoding_val in ds[var].encoding.items():
            dsout[var].encoding[encoding] = encoding_val
        for attr, attr_val in ds[var].attrs.items():
            # Recalculate data min and max
            if attr == 'data_min':
                dsout[var].attrs[attr] = np.nanmin(dsout[var].data)
            elif attr == 'data_max':
                dsout[var].attrs[attr] = np.nanmax(dsout[var].data)
            # Update sensor depth for each segment
            elif attr == 'sensor_depth':
                dsout[var].attrs[attr] = instrument_depth
            else:
                dsout[var].attrs[attr] = attr_val

    # Add global attributes
    for key, value in ds.attrs.items():
        dsout.attrs[key] = value

    ns_to_days = 1. / (60 * 60 * 24 * 1e9)  # nanoseconds to days

    geospatial_vertical_min, geospatial_vertical_max = utils.geospatial_vertical_extrema(
        dsout.orientation, dsout.instrument_depth.data, dsout.distance.data
    )

    # duration must be in decimal days format
    dsout.attrs['time_coverage_duration'] = get_time_duration(
        float(dsout.time.data[-1] - dsout.time.data[0]) * ns_to_days
    )
    # dsout.attrs['source'] = 'https://github.com/IOS-OSD-DPG/pycurrents_ADCP_processing'
    # string format
    dsout.attrs['time_coverage_start'] = utils.numpy_datetime_to_str_utc(dsout.time.data[0])
    dsout.attrs['time_coverage_end'] = utils.numpy_datetime_to_str_utc(dsout.time.data[-1])
    dsout.attrs['geospatial_vertical_min'] = geospatial_vertical_min
    dsout.attrs['geospatial_vertical_max'] = geospatial_vertical_max
    dsout.attrs['date_created'] = datetime.datetime.now(datetime.timezone.utc).strftime(
        '%Y-%m-%d %H:%M:%S UTC'
    )

    # Update processing_history
    dsout.attrs['processing_history'] += (f" Dataset split into {num_segments} segments due to a mooring "
                                          f"strike(s) at {time_of_strike}. New instrument depth calculated "
                                          f"as the sum of the mean instrument depth for this segment and an "
                                          f"offset of {depth_correction}m, where the offset is the "
                                          f"difference between the mean instrument depth for segment 1 and "
                                          f"the instrument depth input by the user.")

    # Convert start and end idx back to within the context of the original dataset length
    # leading_ens_cut, trailing_ens_cut = utils.parse_processing_history(dsout.attrs['processing_history'])
    segment_start_idx = python_index_to_matlab(start_idx)
    segment_end_idx = python_index_to_matlab(end_idx)

    dsout.attrs['processing_history'] += (f" From the original time series with length {len(ds.time.data)},"
                                          f" the segment start index = {segment_start_idx} and the"
                                          f" segment end index = {segment_end_idx}.")

    if type(recovery_lat) == float:
        dsout.attrs['geospatial_lat_min'] = recovery_lat
        dsout.attrs['geospatial_lat_max'] = recovery_lat
        dsout.attrs['processing_history'] += ' Latitude updated with coordinates from recovery cruise.'
    if type(recovery_lon) == float:
        dsout.attrs['geospatial_lon_min'] = recovery_lon
        dsout.attrs['geospatial_lon_max'] = recovery_lon
        dsout.attrs['processing_history'] += ' Longitude updated with coordinates from recovery cruise.'

    return dsout, new_filename


def split_ds_by_pressure(input_ds: xr.Dataset, segment_starts: list, segment_ends: list,
                         dest_dir: str, recovery_lat=None, recovery_lon=None, verbose=False):
    """
    Split dataset by pressure changes if the mooring was hit and displaced
    :param input_ds: input ADCP dataset
    :param segment_starts: segment start indices; list of ints
    :param segment_ends: segment end indices; list of ints
    :param recovery_lat: latitude recorded on recovery cruise; optional
    :param recovery_lon: longitude recorded on recovery cruise; optional
    :param dest_dir: destination directory for output files
    :param verbose: print out progress statements if True; default False
    """
    # Use the input instrument_depth for the first segment
    segment_instr_depths, depth_correction = get_segment_instrument_depths(
        segment_starts, segment_ends, time_series_depth=input_ds.PPSAADCP.data,
        start_instrument_depth=input_ds.instrument_depth.data
    )

    num_segments = len(segment_instr_depths)

    # Join times of splits if there were more than one mooring strike for writing to processing_history
    time_of_split = ' & '.join(
        [utils.numpy_datetime_to_str_utc(t) for t in input_ds.time.data[np.array(segment_ends[:-1]) + 1]]
    )

    # Initialize list to hold the file names of all netcdf files to be output
    netcdf_filenames = []

    # Make a plot of pressure before splitting up the dataset
    if verbose:
        print('Plotting pressure before splitting dataset...')
    pw.plot_adcp_pressure(input_ds, dest_dir=dest_dir, is_pre_split=True)

    # Iterate through all the segments and create a netCDF file from each
    for st_idx, en_idx, i in zip(segment_starts, segment_ends, range(num_segments)):
        if verbose:
            print(f'Segment {i + 1}: index {st_idx} to {en_idx}')

        # Generate as many file names as there are segments of data
        # only use the "date" part of the datetime and not the time portion
        # format the depth to 4 string characters by adding zeros if necessary

        # split the dataset by creating subsets from the original
        # Only apply the recovery lat and lon to the last segment of data
        if i < len(segment_instr_depths) - 1:
            ds_segment, out_segment_name = make_dataset_from_subset(
                input_ds, st_idx, en_idx, segment_instr_depths[i], depth_correction,
                num_segments, time_of_strike=time_of_split
            )
        else:
            # The last segment of data where i == len(segment_instr_depths) - 1
            ds_segment, out_segment_name = make_dataset_from_subset(
                input_ds, st_idx, en_idx, segment_instr_depths[i], depth_correction,
                num_segments, time_of_strike=time_of_split,
                recovery_lat=recovery_lat, recovery_lon=recovery_lon
            )

        if verbose:
            print('New netCDF file:', out_segment_name)

        # print(ds_segment.attrs)
        # print(ds_segment)

        # File name
        absolute_segment_name = os.path.join(dest_dir, out_segment_name)

        netcdf_filenames.append(absolute_segment_name)

        # Export the dataset object as a new netCDF file
        ds_segment.to_netcdf(absolute_segment_name, mode='w', format='NETCDF4')

        ds_segment.close()

    return netcdf_filenames


def compute_sea_surface_height(orientation: 'str', sensor_depth: float, distance: np.ndarray,
                               meta_dict=None):
    if orientation == 'up':
        DISTTRAN = np.round(sensor_depth - distance, decimals=2)
        history = ' DISTTRAN (sea-surface height) calculated as sensor depth minus distance for upward-facing ADCPs.'
    else:
        DISTTRAN = np.round(sensor_depth + distance, decimals=2)
        history = ' DISTTRAN (sea-surface height) calculated as sensor depth plus distance for downward-facing ADCPs.'

    if meta_dict is not None:
        meta_dict['processing_history'] += f" Sensor depth set to the mean of trimmed depth values."
        meta_dict['processing_history'] += history
    return DISTTRAN


def truncate_time_series_ends(var_dict: dict, meta_dict: dict):
    # Set start and end indices
    if 'segment_start_indices' in meta_dict.keys():
        # Take the first start index and the last end index of these list-type values
        e1 = meta_dict['segment_start_indices'][0]
        # Need to subtract 1 since len(var_dict['time']) >= last index of var_dict['time'] + 1
        # otherwise e2 can never be == 0
        e2 = len(var_dict['time']) - meta_dict['segment_end_indices'][-1] - 1
    else:
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

    # some data after deployment and before recovery are also sometimes cut - statements not accurate
    # If these are changed, update utils.parse_processing_history() !!!!
    if e1 != 0:
        meta_dict['processing_history'] += f' Leading {e1} ensembles from before deployment discarded.'
    if e2 != 0:
        meta_dict['processing_history'] += f' Trailing {e2} ensembles from after recovery discarded.'

    # Update segment_start_indices and segment_end_indices in case of mooring strike to start from zero
    # Take max of start ind with zero in case ind < e1
    meta_dict['segment_start_indices'] = [np.max([ind - e1, 0]) for ind in meta_dict['segment_start_indices']]
    meta_dict['segment_end_indices'] = [ind - e1 for ind in meta_dict['segment_end_indices']]

    return var_dict


def get_time_resolution(time_data):
    """
    Get resolution of time dimension in format "HH:MM:SS"
    """
    dt = pd.Timedelta(((time_data[2] - time_data[0]) / 2))
    # return dt[-8:]
    return str(dt).split(' ')[2]  # Remove day component of Timedelta that is before the HH:MM:SS part


def get_time_duration(days: float):
    """Get time duration in format "123 days, HH:MM:SS" """
    duration = str(pd.Timedelta(days, unit='day')).split('.')[0]
    # Second step to add comma after "days"
    return duration.replace('days ', 'days, ')


def nc_create_L0_L1(in_file, file_meta, dest_dir, level=1, time_file=None, verbose=False):
    """About:
    Perform level 0 or 1 processing on a raw ADCP file and export it as a netCDF file
    :param in_file: full file name of raw ADCP file
    :param file_meta: full file name of csv metadata file associated with in_file
    :param dest_dir: string type; name of folder in which files will be output
    :param level: processing level to use, default to 1 but can be zero
    :param time_file: full file name of csv file containing user-generated time data;
                required if inFile has garbled out-of-range time data
    :param verbose: If True then print out progress statements
    """

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Splice file name to get output netCDF file name
    out_name = os.path.basename(in_file)[:-4] + f'_L{level}.adcp.nc'

    if verbose:
        print(out_name)

    # Read information from metadata file into a dictionary, called meta_dict
    meta_dict = create_meta_dict(file_meta, level)

    if verbose:
        print('Read in csv metadata file')

    # ------------------------Read in data and start processing--------------------

    # Read in raw ADCP file and model type
    data = rdiraw.rawfile(in_file, meta_dict['model'], trim=True)

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
    # --------Create flags if variables are *present*, NOT if they are missing
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
    meta_dict = update_meta_dict(meta_dict, data, fixed_leader, level)

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

    # Amplitude todo set all these to float32?
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

    # var_dict['ELTMEP01'] = var_dict['time']
    var_dict['TEMPPR01'] = vel.temperature
    var_dict['PTCHGP01'] = vel.pitch
    var_dict['ROLLGP01'] = vel.roll
    var_dict['HEADCM01'] = vel.heading

    # Convert SoundSpeed from int16 to float32??
    var_dict['SVELCV01'] = vel.VL['SoundSpeed']

    # Convert pressure
    var_dict['PRESPR01'] = assign_pressure(vel_var=vel, meta_dict=meta_dict, level=level)
    if var_dict['PRESPR01'] is None:
        # Drop pressure if no data and L1 processing
        _ = var_dict.pop('PRESPR01')

    # Dimensionless vars
    # var_dict['ALATZZ01'] = meta_dict['latitude']
    # var_dict['ALONZZ01'] = meta_dict['longitude']
    var_dict['latitude'] = meta_dict['latitude']
    var_dict['longitude'] = meta_dict['longitude']
    var_dict['filename'] = out_name[:-3]  # do not include .nc suffix
    var_dict['instrument_serial_number'] = meta_dict['serial_number']
    var_dict['instrument_model'] = meta_dict['instrument_model']
    var_dict['geographic_area'] = meta_dict['geographic_area']
    var_dict['instrument_depth'] = meta_dict['instrument_depth']
    var_dict['water_depth'] = meta_dict['water_depth']

    # ------------------------Adjust velocity data-------------------------

    # Set velocity values of -32768.0 to nans, since -32768.0 is the automatic
    # fill_value for pycurrents
    vel.vel.data[vel.vel.data == vel.vel.fill_value] = _FillValue

    if meta_dict['model'] == 'sv' and flag_vb == 1:
        vb_vel.vbvel.data[vb_vel.vbvel.data == vb_vel.vbvel.fill_value] = _FillValue

    # Rotate into earth if not in enu already; this makes the netCDF bigger
    # For Sentinel V instruments, transformations are done independently of vertical
    # beam velocity data
    if level == 1 and vel.trans.coordsystem not in ['earth', 'enu']:
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
        if vel.trans.coordsystem in ['earth', 'enu']:
            meta_dict['coord_system'] = 'enu'
        else:
            meta_dict['coord_system'] = vel.trans.coordsystem

    # Correct magnetic declination in velocities and round to the input number of decimal places
    vel_num_decimals = 3
    if level == 1:
        LCEWAP01, LCNSAP01 = correct_true_north(vel1, vel2, meta_dict, vel_num_decimals)

        if verbose:
            print('Applied magnetic declination to north and east velocities')

        var_dict['LCEWAP01'] = LCEWAP01.transpose()
        var_dict['LCNSAP01'] = LCNSAP01.transpose()

        if flag_vb == 1:
            var_dict['LRZUVP01'] = np.round(vb_vel.vbvel.data.transpose(), vel_num_decimals)
    else:
        # Level 0
        var_dict['VEL_MAGNETIC_EAST'] = np.round(vel1.transpose(), vel_num_decimals)
        var_dict['VEL_MAGNETIC_NORTH'] = np.round(vel2.transpose(), vel_num_decimals)

    # Upwards and error velocities the same for both L0 and L1
    var_dict['LRZAAP01'] = np.round(vel3.transpose(), vel_num_decimals)
    var_dict['LERRAP01'] = np.round(vel4.transpose(), vel_num_decimals)

    # ---------Truncate time series variables before computing derived variables---------

    # Skip this step if dataset will be segmented later
    if level == 1 and len(meta_dict['segment_start_indices']) <= 1:
        var_dict = truncate_time_series_ends(var_dict, meta_dict)

    # --------------------------------Additional flagging--------------------------------

    # Only for L1 not L0
    if level == 1:
        var_dict['PRESPR01_QC'] = flag_pressure(pres=var_dict['PRESPR01'], meta_dict=meta_dict)

        for velocity in ['LCEWAP01', 'LCNSAP01', 'LRZAAP01', 'LRZUVP01']:
            if velocity in var_dict.keys():
                var_dict[f'{velocity}_QC'] = np.zeros(shape=var_dict[velocity].shape)

    # -----------------------------Compute derived variables------------------------------

    if level == 1:
        # Apply equivalent of swDepth() to depth data: Calculate height from sea pressure
        # using gsw package. Negative so that depth is positive; units=m
        var_dict['PPSAADCP'] = -np.round(
            gsw.conversions.z_from_p(p=var_dict['PRESPR01'], lat=meta_dict['latitude']), decimals=2
        )

        meta_dict['processing_history'] += (" Time series sea surface height calculated from pressure "
                                            "using the TEOS-10 75-term expression for specific volume.")

        if verbose:
            print('Calculated sea surface height from sea pressure using the gsw package')

        # Calculate sensor depth of instrument based off mean instrument depth
        sensor_dep = np.round(np.nanmean(var_dict['PPSAADCP']), decimals=2)

        # Calculate height of sea surface
        var_dict['DISTTRAN'] = compute_sea_surface_height(
            meta_dict['orientation'], sensor_dep, var_dict['distance'], meta_dict
        )

    if 'PRESPR01' in var_dict.keys():
        # Check instrument_depth from metadata csv file: compare with pressure values
        check_depths(pres=var_dict['PRESPR01'], dist=var_dict['distance'],
                     instr_depth=meta_dict['instrument_depth'], water_depth=meta_dict['water_depth'])

    if verbose:
        print('Finished QCing data; making netCDF object next')

    # -------------------------Make into netCDF file---------------------------

    out = xr.Dataset(coords={'time': var_dict['time'], 'distance': var_dict['distance']})

    for key in VARIABLE_ORDER:
        if key in var_dict.keys():
            # Convert dimensional vars to float32 to avoid
            # xarray.Dataset.to_netcdf() ValueError: cannot convert float NaN to integer
            if len(np.shape(var_dict[key])) == 0:
                out[key] = ((), var_dict[key])
            elif len(np.shape(var_dict[key])) == 1:
                if len(var_dict[key]) == len(var_dict['distance']):
                    out[key] = (('distance'), np.float32(var_dict[key]))
                elif len(var_dict[key]) == len(var_dict['time']):
                    out[key] = (('time'), np.float32(var_dict[key]))
            elif len(np.shape(var_dict[key])) == 2:
                out[key] = (('distance', 'time'), np.float32(var_dict[key]))
            else:
                warnings.warn(f'Shape of variable {key} not compatible', UserWarning)

    # if verbose:
    #     print(out.data_vars)  # Check that all available vars have been added

    # Add variable-specific attributes
    add_attrs_2vars(out_obj=out, meta_dict=meta_dict)  # sensor_depth=sensor_dep, flags....

    # ----------------------Global attributes----------------------

    # Add select meta_dict items as global attributes
    # remove serial number and geographic area duplication between vars and global attrs
    pass_dict_keys = ['cut_lead_ensembles', 'cut_trail_ensembles', 'model',  # 'processing_level',
                      'segment_start_indices', 'segment_end_indices', 'recovery_lat', 'recovery_lon',
                      'serial_number', 'geographic_area', 'water_depth', 'instrument_depth']

    # accepted_netcdf_dtypes = [str, int, float, list, tuple, np.ndarray]

    for key, value in meta_dict.items():
        # Exclude certain items in the dictionary
        if key not in pass_dict_keys and meta_dict[key] is not None:
            out.attrs[key] = value
        elif meta_dict[key] is None:
            # Do not write to netcdf file
            warnings.warn(f'Metadata item {key} with value {meta_dict[key]} not supported by netCDF')

    # Rest of global attributes not from metadata file:
    out.attrs['standard_name_vocabulary'] = 'CF Standard Name Table v29'
    out.attrs['Conventions'] = 'COARDS, CF-1.7, ACDD-1.3'
    out.attrs['deployment_type'] = 'Sub Surface'
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
    out.attrs['instrument_type'] = "ADCP"
    out.attrs['manufacturer'] = meta_dict['manufacturer']
    out.attrs['source'] = "Python code: GitHub: pycurrents_ADCP_processing"
    now = datetime.datetime.now()
    out.attrs['date_created'] = now.strftime("%Y-%m-%d %H:%M:%S")  # renamed date_modified
    out.attrs['_FillValue'] = _FillValue  # str(fill_value)
    out.attrs['featureType'] = "profileTimeSeries"
    out.attrs['firmware_version'] = str(vel.FL.FWV) + '.' + str(vel.FL.FWR)  # firmwareVersion
    out.attrs['frequency'] = str(data.sysconfig['kHz'])
    out.attrs['beam_angle'] = str(fixed_leader.sysconfig['angle'])  # beamAngle
    out.attrs['system_configuration'] = bin(fixed_leader.FL['SysCfg'])[-8:] + '-' + bin(fixed_leader.FL['SysCfg'])[
                                                                                    :9].replace('b', '')
    out.attrs['sensor_source'] = '{0:08b}'.format(vel.FL['EZ'])  # sensorSource
    out.attrs['sensors_avail'] = '{0:08b}'.format(vel.FL['SA'])  # sensors_avail
    out.attrs['three_beam_used'] = str(vel.trans['threebeam']).upper()  # netCDF4 file format doesn't support bool
    out.attrs['bin_mapping'] = str(vel.trans['binmap']).upper()
    out.attrs['valid_correlation_range'] = vel.FL['LowCorrThresh']  # lowCorrThresh
    out.attrs['min_percent_good'] = fixed_leader.FL['PGMin']
    out.attrs['blank'] = '{} m'.format(fixed_leader.FL['Blank'] / 100)  # convert cm to m
    out.attrs['error_velocity_threshold'] = "{} mm s-1".format(fixed_leader.FL['EVMax'])
    tpp_min = '{0:0>2}'.format(fixed_leader.FL['TPP_min'])
    tpp_sec = '{0:0>2}'.format(fixed_leader.FL['TPP_sec'])
    tpp_hun = '{0:0>2}'.format(fixed_leader.FL['TPP_hun'])
    out.attrs['time_ping'] = '{}:{}.{}'.format(tpp_min, tpp_sec, tpp_hun)
    out.attrs['false_target_reject_values'] = '{} counts'.format(fixed_leader.FL['WA'])  # falseTargetThresh
    # out.attrs['data_type'] = "adcp"
    # out.attrs['pred_accuracy'] = 1  # velocityResolution * 1000
    # out.attrs['creator_type'] = "person"
    out.attrs['n_codereps'] = vel.FL.NCodeReps
    out.attrs['xmit_lag'] = vel.FL.TransLag
    out.attrs['xmit_length'] = fixed_leader.FL['Pulse']
    out.attrs['time_coverage_start'] = utils.numpy_datetime_to_str_utc(var_dict['time'][0])
    out.attrs['time_coverage_end'] = utils.numpy_datetime_to_str_utc(var_dict['time'][-1])
    # out.attrs['time_coverage_duration'] = vel.dday[-1] - vel.dday[0]
    # New format:
    out.attrs['time_coverage_duration'] = get_time_duration(vel.dday[-1] - vel.dday[0])
    # out.attrs['time_coverage_duration_units'] = "days"
    out.attrs['time_coverage_resolution'] = get_time_resolution(out.time.data)

    # geospatial lat, lon, and vertical min/max calculations
    out.attrs['geospatial_lat_min'] = meta_dict['latitude']
    out.attrs['geospatial_lat_max'] = meta_dict['latitude']
    out.attrs['geospatial_lat_units'] = "degrees_north"
    out.attrs['geospatial_lon_min'] = meta_dict['longitude']
    out.attrs['geospatial_lon_max'] = meta_dict['longitude']
    out.attrs['geospatial_lon_units'] = "degrees_east"

    # # sensor_depth was removed as a variable attribute
    # if level == 1:
    #     depth = sensor_dep
    # else:
    #     depth = meta_dict['instrument_depth']

    out.attrs['geospatial_vertical_min'], out.attrs['geospatial_vertical_max'] = utils.geospatial_vertical_extrema(
        out.attrs['orientation'], sensor_depth=meta_dict['instrument_depth'], distance=out.distance.data
    )

    if verbose:
        print(out)  # for testing

    # -----------------Split dataset by pressure changes if any--------------

    # User inputs indices or datetimes
    # returns a dictionary of the start and end indices of format {start: end}
    # and a numpy array of depths with length equal to the dictionary
    if level == 1 and len(meta_dict['segment_start_indices']) > 1:
        nc_names = split_ds_by_pressure(
            input_ds=out, segment_starts=meta_dict['segment_start_indices'],
            segment_ends=meta_dict['segment_end_indices'],
            dest_dir=dest_dir, recovery_lat=meta_dict['recovery_lat'],
            recovery_lon=meta_dict['recovery_lon'], verbose=verbose
        )
    else:
        nc_names = [os.path.join(dest_dir, out_name)]
        out.to_netcdf(nc_names[0], mode='w', format='NETCDF4')

    out.close()

    return nc_names


def example1(level=1, verbose=False):
    """
    Specify raw ADCP file to create nc file from, along with associated csv metadata file
    """
    # raw .000 file
    raw_file = "./sample_data/a1_20050503_20050504_0221m.000"
    # csv metadata file
    raw_file_meta = "./sample_data/a1_20050503_20050504_0221m_metadata.csv"

    dest_dir = 'dest_dir'

    # Create netCDF file
    nc_name = nc_create_L0_L1(
        in_file=raw_file, file_meta=raw_file_meta, dest_dir=dest_dir, level=level, verbose=verbose
    )

    # # DEPRECProduce new netCDF file that includes a geographic_area variable
    # geo_name = add_var2nc.add_geo(nc_name, dest_dir)

    # return nc_name, geo_name
    return nc_name


def example2(level=1, verbose=False):
    """Specify raw ADCP file to create nc file from, along with associated csv metadata file
    AND time file"""

    # raw .000 file
    raw_file = "./sample_data/scott2_20160711_20170707_0040m.pd0"
    # csv metadata file
    raw_file_meta = "./sample_data/scott2_20160711_20170707_0040m_metadata.csv"
    # csv time file
    scott_time = './sample_data/scott2_20160711_20170707_0040m_time.csv'

    dest_dir = 'dest_dir'

    # Create netCDF file
    nc_name = nc_create_L0_L1(
        in_file=raw_file, file_meta=raw_file_meta, dest_dir=dest_dir, level=level, time_file=scott_time, verbose=verbose
    )

    # # DEPRECProduce new netCDF file that includes a geographic_area variable
    # geo_name = add_var2nc.add_geo(nc_name, dest_dir)

    # return nc_name, geo_name
    return nc_name
