"""
author: Hana Hourston
date: Jan. 23, 2020

about: This script is adapted from Jody Klymak's at https://gist.github.com/jklymak/b39172bd0f7d008c81e32bf0a72e2f09
for L1 processing raw ADCP data.

Contributions from: Di Wan, Eric Firing

User input (3 places) needed after correct_true_north() function
"""

import os
import glob as glob
import fnmatch
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


def mean_orientation(o):
    # orientation is an array
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
        print('Warning: Number of \"up\" orientations equals number of \"down\" orientations in data subset.')


# Di Wan's magnetic declination correction code: Takes 0 DEG from E-W axis
def correct_true_north(mag_decl, measured_east, measured_north):  # change angle to negative of itself
    angle_rad = -mag_decl * np.pi / 180.
    east_true = measured_east * np.cos(angle_rad) - measured_north * np.sin(angle_rad)
    north_true = measured_east * np.sin(angle_rad) + measured_north * np.cos(angle_rad)
    return east_true, north_true


# User input

# wd = 'your/wd/here'
# os.chdir(wd)

# Specify raw ADCP file to create nc file from, along with associated csv metadata file

# 1) raw .000 file
raw_file = "./sample_data/a1_20050503_20050504_0221m.000"
# 2) csv metadata file
raw_file_meta = "./sample_data/a1_20050503_20050504_0221m_meta_L1.csv"

# If your raw file came from a NarrowBand instrument, you must also use the start_year optional kwarg (int type)

# If your raw file has time values out of range, you must also use the time_file optionalkwarg
# This should be a csv file with time entries spanning the range of deployment


def nc_create_L1(inFile, file_meta, start_year=None, time_file=None):
    
    # Splice file name to get output netCDF file name
    outname = os.path.basename(inFile)[:-4] + '.adcp.L1.nc'
    print(outname)

    # Get model and timestamp from raw file csv metadata file
    # Instrument frequency is not included in the metadata files, so not included in "model". Frequency optional, apparently
    meta_dict = {}
    model = ""
    with open(file_meta) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            # extract all metadata from csv file into dictionary -- some items not passed to netCDF file but are extracted anyway
            if row[0] != "Name":
                meta_dict[row[0]] = row[1]
                # create variables for model
                if row[0] == "instrumentSubtype":
                    if row[1] == "Workhorse":
                        model = "wh"
                        model_long = "RDI WH Long Ranger"
                        manufacturer = 'teledyne rdi'
                    elif row[1] == "Broadband":
                        model = "bb"
                        model_long = "RDI BB"
                        manufacturer = 'teledyne rdi'
                    elif row[1] == "Narrowband":
                        model = "nb"
                        model_long = "RDI NB"
                        manufacturer = 'teledyne rdi'
                    elif row[1] == "Sentinel V":  # missing from documentation
                        model = "sv"
                        model_long = "RDI SV"
                        manufacturer = 'teledyne rdi'
                    elif row[1] == 'Ocean Surveyor' or row[1] == 'ocean surveyor':
                        model = "os"
                        model_long = "RDI OS"
                        manufacturer = "teledyne rdi"
                    else:
                        continue
                elif row[0] == '' and row[1] == '':
                    warnings.warn('Metadata file contains a blank row; skipping this row', UserWarning)
                elif row[0] != '' and row[1] == '':
                    warnings.warn('Metadata item in csv file has blank value; skipping this row '
                                  'in metadata file', UserWarning)
                else:
                    continue
    
    if model == "":
        ValueError("No valid instrumentSubtype value detected")
        
    print('Read in csv metadata file')

    # Read in data and start processing

    # Read in raw ADCP file and model type
    if model == 'nb':
        data = rawfile(inFile, model, trim=True, yearbase=start_year)
    else:
        data = rawfile(inFile, model, trim=True)
    print('Read in raw data')

    # Extract multidimensional variables from data object: fixed leader, velocity, amplitude intensity, correlation magnitude, and percent good
    fixed_leader = data.read(varlist=['FixedLeader'])
    vel = data.read(varlist=['Velocity'])
    amp = data.read(varlist=['Intensity'])
    cor = data.read(varlist=['Correlation'])
    pg = data.read(varlist=['PercentGood'])

    # Metadata value corrections

    # Convert numeric values to numerics
    meta_dict['country_institute_code'] = int(meta_dict['country_institute_code'])
    meta_dict['instrument_depth'] = float(meta_dict['instrument_depth'])
    meta_dict['latitude'] = float(meta_dict['latitude'])
    meta_dict['longitude'] = float(meta_dict['longitude'])
    meta_dict['water_depth'] = float(meta_dict['water_depth'])
    if 'magnetic_variation' in meta_dict:
        meta_dict['magnetic_variation'] = float(meta_dict['magnetic_variation'])
    else:
        pass
        #meta_dict['magnetic_variation'] = float(magnetic_variation)

    # Add leading zero to serial numbers that have 3 digits
    if len(str(meta_dict['serialNumber'])) == 3:
        meta_dict['serialNumber'] = '0' + str(meta_dict['serialNumber'])
    # Overwrite serial number to include the model: upper returns uppercase
    meta_dict['serialNumber'] = model.upper() + meta_dict['serialNumber']
    # Add instrument model variable value
    meta_dict['instrumentModel'] = '{} ADCP {}kHz ({})'.format(model_long, data.sysconfig['kHz'],
                                                               meta_dict['serialNumber'])

    # Begin writing processing history, which will be added as a global attribute to the output netCDF file
    processing_history = "Metadata read in from log sheet and combined with raw data to export as netCDF file."

    # Extract metadata from data object

    # Orientation code from Eric Firing
    # Orientation values such as 65535 and 231 cause SysCfg().up to generate an IndexError: list index out of range
    # Such values include 65535 and 231
    try:
        orientations = [SysCfg(fl).up for fl in fixed_leader.raw.FixedLeader['SysCfg']]
        orientation = mean_orientation(orientations)
    except IndexError:
        warnings.warn('Orientation obtained from data.sysconfig[\'up\'] to avoid IndexError: list index out of range', 
                      UserWarning)        
        orientation = 'up' if data.sysconfig['up'] else 'down'

    # Retrieve beam pattern
    if data.sysconfig['convex']:
        beamPattern = 'convex'
    else:
        beamPattern = ''

    # Get timestamp from "data" object just created
    # In R, the start time was obtained from the "adp" object created within R
    # data.yearbase is an integer of the year that the timeseries starts (e.g., 2016)
    data_origin = pd.Timestamp(str(data.yearbase) + '-01-01')  # convert to date object

    # Set up dimensions and variables

    try:
        # convert time variable to elapsed time since 1970-01-01T00:00:00Z
        time_us = np.array(
            pd.to_datetime(vel.dday, unit='D', origin=data_origin, utc=True).strftime('%Y-%m-%d %H:%M:%S'),
            dtype='datetime64[s]')
        # DTUT8601 variable: time strings
        time_DTUT8601 = pd.to_datetime(vel.dday, unit='D', origin=data_origin, utc=True, errors='coerce').strftime(
            '%Y-%m-%d %H:%M:%S')  # don't need %Z in strftime
    except OutOfBoundsDatetime or OverflowError:
        print('Using user-created time range')
        time_s = np.zeros(shape=data.nprofs, dtype='datetime64[s]')
        time_DTUT8601 = np.empty(shape=data.nprofs, dtype='<U100')
        with open(time_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # Skip headers
            next(csv_reader, None)
            for count, row in enumerate(csv_reader):
                if row[0] == '':
                    pass
                else:
                    time_s[count] = np.datetime64(pd.to_datetime(row[0], utc=True).strftime(
                        '%Y-%m-%d %H:%M:%S'))
                    time_DTUT8601[count] = pd.to_datetime(row[0], utc=True).strftime('%Y-%m-%d %H:%M:%S')

        processing_history = processing_history + ' OutOfBoundsDateTime exception triggered; used user-' \
                                                  'generated time range as time data.'

    # Distance dimension
    distance = np.round(vel.dep.data, decimals=2)

    # Continue setting up variables

    # Convert SoundSpeed from int16 to float32
    sound_speed = np.float32(vel.VL['SoundSpeed'])

    # Convert pressure
    if model == 'wh' or model == 'os' or model == 'sv':
        pressure = np.array(vel.VL['Pressure'] / 1000, dtype='float32')  # convert decapascal to decibars

        # Calculate pressure based on static instrument depth if missing pressure sensor; extra condition added for zero pressure or weird pressure values
        # Handle no unique modes from statistics.mode(pressure)
        pressure_unique, counts = np.unique(pressure, return_counts=True)
        index_of_zero = np.where(pressure_unique == 0)
        print('np.max(counts):', np.max(counts), sep=' ')
        print('counts[index_of_zero]:', counts[index_of_zero], sep=' ')
        print('serial number:', meta_dict['serialNumber'])

    if model == 'bb' or model == 'nb' or np.max(counts) == counts[index_of_zero]:  # if zero is a mode of pressure
        p = np.round(gsw.conversions.p_from_z(-meta_dict['instrument_depth'], meta_dict['latitude']),
                     decimals=0)  # depth negative because positive is up for this function
        pressure = np.repeat(p, len(vel.vel1.data))
        processing_history = processing_history + " Pressure values calculated from static instrument depth ({} m) using " \
                                                  "the TEOS-10 75-term expression for specific volume and rounded to {} " \
                                                  "significant digits.".format(str(meta_dict['instrument_depth']),
                                                                               num2words(len(str(p))))
        warnings.warn('Pressure values calculated from static instrument depth', UserWarning)

    # Check instrument_depth from metadata csv file: compare with pressure values
    depths_check = np.mean(pressure[:]) - distance
    inst_depth_check = depths_check[0] + distance[0]
    abs_difference = np.absolute(inst_depth_check-meta_dict['instrument_depth'])
    # Calculate percent difference in relation to total water depth
    if (abs_difference / meta_dict['water_depth'] * 100) > 0.05:
        warnings.warn(message="Difference between calculated instrument depth and metadata instrument_depth "
                              "exceeds 0.05% of the total water depth", category=UserWarning)

    # Adjust velocity data

    # Set velocity values of -32768.0 to nans, since -32768.0 is the automatic fill_value for pycurrents
    vel.vel.data[vel.vel.data == -32768.0] = np.nan

    # Rotate into earth if not in enu already; this makes the netCDF bigger
    if vel.trans.coordsystem != 'earth' and vel.trans.coordsystem != 'enu':
        trans = transform.Transform(angle=fixed_leader.sysconfig['angle'], geometry=beamPattern)  #angle is beam angle
        xyze = trans.beam_to_xyz(vel.vel.data)
        print(np.shape(xyze))
        enu = transform.rdi_xyz_enu(xyze, vel.heading, vel.pitch, vel.roll, orientation=orientation)
        print(np.shape(enu))
        # Apply change in coordinates to velocities
        vel1 = xr.DataArray(enu[:, :, 0], dims=['time', 'distance'])
        vel2 = xr.DataArray(enu[:, :, 1], dims=['time', 'distance'])
        vel3 = xr.DataArray(enu[:, :, 2], dims=['time', 'distance'])
        vel4 = xr.DataArray(enu[:, :, 3], dims=['time', 'distance'])
        processing_history = processing_history + " The coordinate system was rotated into enu coordinates."
        meta_dict['coord_system'] = 'enu'  # Add item to metadata dictionary for coordinate system
        print('Coordinate system rotated from {} to enu'.format(vel.trans.coordsystem))
    else:
        vel1 = vel.vel1
        vel2 = vel.vel2
        vel3 = vel.vel3
        vel4 = vel.vel4
        meta_dict['coord_system'] = 'enu'

    # Correct magnetic declination in velocities
    # meta_dict['magnetic_variation'] = magnetic_variation
    LCEWAP01, LCNSAP01 = correct_true_north(meta_dict['magnetic_variation'], vel1.data, vel2.data)
    processing_history += " Magnetic variation, using average applied; declination = {}.".format(
        str(meta_dict['magnetic_variation']))

    # Flag velocity data based on cut_lead_ensembles and cut_trail_ensembles

    # Set start and end indices
    e1 = int(meta_dict['cut_lead_ensembles'])  # "ensemble 1"
    e2 = int(meta_dict['cut_trail_ensembles'])  # "ensemble 2"

    # Create QC variables containing flag arrays
    LCEWAP01_QC = np.zeros(shape=LCEWAP01.shape, dtype='float32')
    LCNSAP01_QC = np.zeros(shape=LCNSAP01.shape, dtype='float32')
    LRZAAP01_QC = np.zeros(shape=vel3.data.shape, dtype='float32')

    for qc in [LCEWAP01_QC, LCNSAP01_QC, LRZAAP01_QC]:
        # 0=no_quality_control, 4=value_seems_erroneous
        for b in range(data.NCells):
            qc[:e1, b] = 4
            if e2 != 0:
                qc[-e2:, b] = 4  # if e2==0, the slice [-0:] would index the whole array

    # Apply the flags to the data and set bad data to NAs
    LCEWAP01[LCEWAP01_QC == 4] = np.nan
    LCNSAP01[LCNSAP01_QC == 4] = np.nan
    vel3.data[LRZAAP01_QC == 4] = np.nan
    processing_history += " Quality control flags set based on SeaDataNet flag scheme from BODC."

    # Depth

    # Apply equivalent of swDepth() to depth data: Calculate height from sea pressure using gsw package
    # negative so that depth is positive; units=m
    depth = -gsw.conversions.z_from_p(p=pressure, lat=meta_dict['latitude'])
    print('Calculated height from sea pressure using gsw package')

    # Limit variables (depth, pressure, temperature, pitch, roll, heading, sound_speed) from before dep. and after rec. of ADCP
    depth[:e1] = np.nan
    pressure[:e1] = np.nan
    vel.temperature[:e1] = np.nan
    vel.pitch[:e1] = np.nan
    vel.roll[:e1] = np.nan
    vel.heading[:e1] = np.nan
    sound_speed[:e1] = np.nan
    if e2 != 0:
        depth[-e2:] = np.nan
        pressure[-e2:] = np.nan
        vel.temperature[-e2:] = np.nan
        vel.pitch[-e2:] = np.nan
        vel.roll[-e2:] = np.nan
        vel.heading[-e2:] = np.nan
        sound_speed[-e2:] = np.nan

        processing_history += " Depth pressure, temperature, pitch, roll, heading, and sound_speed limited by " \
                              "deployment ({} UTC) and recovery ({} UTC) " \
                              "times.".format(time_DTUT8601[e1], time_DTUT8601[-e2])
    else:
        processing_history += " Depth pressure, temperature, pitch, roll, heading, and sound_speed limited by " \
                              "deployment ({} UTC) time.".format(time_DTUT8601[e1])

    # Calculate sensor depth of instrument based off mean instrument transducer depth
    sensor_dep = np.nanmean(depth)
    processing_history += " Sensor depth and mean depth set to {} based on trimmed depth values.".format(
        str(sensor_dep))

    # Calculate height of sea surface: bin height minus sensor depth
    DISTTRAN = distance - sensor_dep

    processing_history += ' Level 1 processing was performed on the dataset. This entailed corrections for magnetic ' \
                          'declination based on an average of the dataset and cleaning of the beginning and end of ' \
                          'the dataset. No QC was carried out. The leading {} ensembles and the trailing {} ensembles ' \
                          'were removed from the data set.'.format(meta_dict['cut_lead_ensembles'],
                                                                   meta_dict['cut_trail_ensembles'])

    print('Finished QCing data; making netCDF object next')

    # Make into netCDF file

    # Create xarray Dataset object containing all dimensions and variables
    out = xr.Dataset(coords={'time': time_us, 'distance': distance},
                     data_vars={'LCEWAP01': (['distance', 'time'], LCEWAP01.transpose()),
                                'LCNSAP01': (['distance', 'time'], LCNSAP01.transpose()),
                                'LRZAAP01': (['distance', 'time'], vel3.data.transpose()),
                                'LERRAP01': (['distance', 'time'], vel4.data.transpose()),
                                'LCEWAP01_QC': (['distance', 'time'], LCEWAP01_QC.transpose()),
                                'LCNSAP01_QC': (['distance', 'time'], LCNSAP01_QC.transpose()),
                                'LRZAAP01_QC': (['distance', 'time'], LRZAAP01_QC.transpose()),
                                'ELTMEP01': (['time'], time_us),
                                'TNIHCE01': (['distance', 'time'], amp.amp1.transpose()),
                                'TNIHCE02': (['distance', 'time'], amp.amp2.transpose()),
                                'TNIHCE03': (['distance', 'time'], amp.amp3.transpose()),
                                'TNIHCE04': (['distance', 'time'], amp.amp4.transpose()),
                                'CMAGZZ01': (['distance', 'time'], cor.cor1.transpose()),
                                'CMAGZZ02': (['distance', 'time'], cor.cor2.transpose()),
                                'CMAGZZ03': (['distance', 'time'], cor.cor3.transpose()),
                                'CMAGZZ04': (['distance', 'time'], cor.cor4.transpose()),
                                'PCGDAP00': (['distance', 'time'], pg.pg1.transpose()),
                                'PCGDAP02': (['distance', 'time'], pg.pg2.transpose()),
                                'PCGDAP03': (['distance', 'time'], pg.pg3.transpose()),
                                'PCGDAP04': (['distance', 'time'], pg.pg4.transpose()),
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
                                'SVELCV01': (['time'], sound_speed),
                                'DTUT8601': (['time'], time_DTUT8601),
                                'filename': ([], outname[:-3]),
                                'instrument_serial_number': ([], meta_dict['serialNumber']),
                                'instrument_model': ([], meta_dict['instrumentModel'])})

    # Add attributes to each variable

    fillValue = 1e+15
    uvw_vel_min = -1000
    uvw_vel_max = 1000

    # Time
    var = out.time
    var.encoding['units'] = "seconds since 1970-01-01T00:00:00Z"
    var.encoding['_FillValue'] = None  # omits fill value from time dim; otherwise would be included with value NaN
    var.attrs['long_name'] = "time"
    var.attrs['cf_role'] = "profile_id"
    var.encoding['calendar'] = "gregorian"

    # Bin distances
    var = out.distance
    var.encoding['_FillValue'] = None
    var.attrs['units'] = "metres"
    # var.attrs['long_name'] = "distance"
    var.attrs['long_name'] = "bin_distances_from_ADCP_transducer_along_measurement_axis"

    # LCEWAP01: eastward velocity (vel1)
    # all velocities have many of the same attribute values, but not all, so each velocity is done separately
    var = out.LCEWAP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm/sec'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'eastward_sea_water_velocity'
    var.attrs['ancillary_variables'] = 'LCEWAP01_QC'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'u'
    var.attrs['flag_meanings'] = meta_dict['flag_meaning']
    var.attrs['flag_values'] = meta_dict['flag_values']
    var.attrs['References'] = meta_dict['flag_references']
    var.attrs['legency_GF3_code'] = 'SDN:GF3::EWCT'
    var.attrs['sdn_parameter_name'] = 'Eastward current velocity (Eulerian measurement) in the water body by moored ' \
                                      'acoustic doppler current profiler (ADCP)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['standard_name'] = 'eastward_sea_water_velocity'
    var.attrs['data_max'] = np.round(np.nanmax(LCEWAP01), decimals=2)
    var.attrs['data_min'] = np.round(np.nanmin(LCEWAP01), decimals=2)
    var.attrs['valid_max'] = uvw_vel_max
    var.attrs['valid_min'] = uvw_vel_min

    # LCNSAP01: northward velocity (vel2)
    var = out.LCNSAP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm/sec'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'northward_sea_water_velocity'
    var.attrs['ancillary_variables'] = 'LCNSAP01_QC'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'v'
    var.attrs['flag_meanings'] = meta_dict['flag_meaning']
    var.attrs['flag_values'] = meta_dict['flag_values']
    var.attrs['References'] = meta_dict['flag_references']
    var.attrs['legency_GF3_code'] = 'SDN:GF3::NSCT'
    var.attrs['sdn_parameter_name'] = 'Northward current velocity (Eulerian measurement) in the water body by moored ' \
                                      'acoustic doppler current profiler (ADCP)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['standard_name'] = 'northward_sea_water_velocity'
    var.attrs['data_max'] = np.round(np.nanmax(LCNSAP01), decimals=2)
    var.attrs['data_min'] = np.round(np.nanmin(LCNSAP01), decimals=2)
    var.attrs['valid_max'] = uvw_vel_max
    var.attrs['valid_min'] = uvw_vel_min

    # LRZAAP01: vertical velocity (vel3)
    var = out.LRZAAP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm/sec'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'upward_sea_water_velocity'
    var.attrs['ancillary_variables'] = 'LRZAAP01_QC'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'w'
    var.attrs['flag_meanings'] = meta_dict['flag_meaning']
    var.attrs['flag_values'] = meta_dict['flag_values']
    var.attrs['References'] = meta_dict['flag_references']
    var.attrs['legency_GF3_code'] = 'SDN:GF3::VCSP'
    var.attrs['sdn_parameter_name'] = 'Upward current velocity (Eulerian measurement) in the water body by moored ' \
                                      'acoustic doppler current profiler (ADCP)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['standard_name'] = 'upward_sea_water_velocity'
    var.attrs['data_max'] = np.nanmax(vel3)
    var.attrs['data_min'] = np.nanmin(vel3)
    var.attrs['valid_max'] = uvw_vel_max
    var.attrs['valid_min'] = uvw_vel_min

    # LERRAP01: error velocity (vel4)
    var = out.LERRAP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm/sec'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'error_velocity_in_sea_water'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'e'
    var.attrs['flag_meanings'] = meta_dict['flag_meaning']
    var.attrs['flag_values'] = meta_dict['flag_values']
    var.attrs['References'] = meta_dict['flag_references']
    var.attrs['legency_GF3_code'] = 'SDN:GF3::ERRV'
    var.attrs['sdn_parameter_name'] = 'Current velocity error in the water body by moored acoustic doppler current ' \
                                      'profiler (ADCP)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['data_max'] = np.nanmax(vel4)
    var.attrs['data_min'] = np.nanmin(vel4)
    var.attrs['valid_max'] = 2 * uvw_vel_max
    var.attrs['valid_min'] = 2 * uvw_vel_min

    # Velocity variable quality flags
    var = out.LCEWAP01_QC
    var.encoding['dtype'] = 'int'
    var.attrs['_FillValue'] = 0
    var.attrs['long_name'] = 'quality flag for LCEWAP01'
    var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset'
    var.attrs['flag_meanings'] = meta_dict['flag_meaning']
    var.attrs['flag_values'] = meta_dict['flag_values']
    var.attrs['References'] = meta_dict['flag_references']
    var.attrs['data_max'] = np.nanmax(LCEWAP01_QC)
    var.attrs['data_min'] = np.nanmin(LCEWAP01_QC)

    var = out.LCNSAP01_QC
    var.encoding['dtype'] = 'int'
    var.attrs['_FillValue'] = 0
    var.attrs['long_name'] = 'quality flag for LCNSAP01'
    var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset'
    var.attrs['flag_meanings'] = meta_dict['flag_meaning']
    var.attrs['flag_values'] = meta_dict['flag_values']
    var.attrs['References'] = meta_dict['flag_references']
    var.attrs['data_max'] = np.nanmax(LCNSAP01_QC)
    var.attrs['data_min'] = np.nanmin(LCNSAP01_QC)

    var = out.LRZAAP01_QC
    var.encoding['dtype'] = 'int'
    var.attrs['_FillValue'] = 0
    var.attrs['long_name'] = 'quality flag for LRZAAP01'
    var.attrs['comment'] = 'Quality flag resulting from cleaning of the beginning and end of the dataset'
    var.attrs['flag_meanings'] = meta_dict['flag_meaning']
    var.attrs['flag_values'] = meta_dict['flag_values']
    var.attrs['References'] = meta_dict['flag_references']
    var.attrs['data_max'] = np.nanmax(LRZAAP01_QC)
    var.attrs['data_min'] = np.nanmin(LRZAAP01_QC)

    # ELTMEP01: seconds since 1970
    var = out.ELTMEP01
    var.encoding['dtype'] = 'd'
    var.encoding['units'] = 'seconds since 1970-01-01T00:00:00Z'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'time_02'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::N/A'
    var.attrs['sdn_parameter_name'] = 'Elapsed time (since 1970-01-01T00:00:00Z)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UTBB'
    var.attrs['sdn_uom_name'] = 'Seconds'
    var.attrs['standard_name'] = 'time'

    # TNIHCE01-4: echo intensity beam 1-4
    var = out.TNIHCE01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_echo_intensity_beam_1'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::BEAM_01'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current ' \
                                      'profiler (ADCP) beam 1'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
    var.attrs['sdn_uom_name'] = 'Counts'
    var.attrs['data_min'] = np.nanmin(amp.amp1)
    var.attrs['data_max'] = np.nanmax(amp.amp1)

    var = out.TNIHCE02
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_echo_intensity_beam_2'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::BEAM_02'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current ' \
                                      'profiler (ADCP) beam 2'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
    var.attrs['sdn_uom_name'] = 'Counts'
    var.attrs['data_min'] = np.nanmin(amp.amp2)
    var.attrs['data_max'] = np.nanmax(amp.amp2)

    var = out.TNIHCE03
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_echo_intensity_beam_3'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::BEAM_03'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current ' \
                                      'profiler (ADCP) beam 3'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
    var.attrs['sdn_uom_name'] = 'Counts'
    var.attrs['data_min'] = np.nanmin(amp.amp3)
    var.attrs['data_max'] = np.nanmax(amp.amp3)

    var = out.TNIHCE04
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_echo_intensity_beam_4'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::BEAM_04'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current ' \
                                      'profiler (ADCP) beam 4'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
    var.attrs['sdn_uom_name'] = 'Counts'
    var.attrs['data_min'] = np.nanmin(amp.amp4)
    var.attrs['data_max'] = np.nanmax(amp.amp4)

    # PCGDAP00 - 4: percent good beam 1-4
    var = out.PCGDAP00
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'percent'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'percent_good_beam_1'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'PGd'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::PGDP_01'
    var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler ' \
                                      'current profiler (ADCP) beam 1'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
    var.attrs['sdn_uom_name'] = 'Percent'
    var.attrs['data_min'] = np.nanmin(pg.pg1)
    var.attrs['data_max'] = np.nanmax(pg.pg1)

    var = out.PCGDAP02
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'percent'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'percent_good_beam_2'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'PGd'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::PGDP_02'
    var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler ' \
                                      'current profiler (ADCP) beam 2'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
    var.attrs['sdn_uom_name'] = 'Percent'
    var.attrs['data_min'] = np.nanmin(pg.pg2)
    var.attrs['data_max'] = np.nanmax(pg.pg2)

    var = out.PCGDAP03
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'percent'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'percent_good_beam_3'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'PGd'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::PGDP_03'
    var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler ' \
                                      'current profiler (ADCP) beam 3'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
    var.attrs['sdn_uom_name'] = 'Percent'
    var.attrs['data_min'] = np.nanmin(pg.pg3)
    var.attrs['data_max'] = np.nanmax(pg.pg3)

    var = out.PCGDAP04
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'percent'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'percent_good_beam_4'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'PGd'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::PGDP_04'
    var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler ' \
                                      'current profiler (ADCP) beam 4'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
    var.attrs['sdn_uom_name'] = 'Percent'
    var.attrs['data_min'] = np.nanmin(pg.pg4)
    var.attrs['data_max'] = np.nanmax(pg.pg4)

    # PTCHGP01: pitch
    var = out.PTCHGP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'degrees'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'pitch'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::PTCH'
    var.attrs['sdn_parameter_name'] = 'Orientation (pitch) of measurement platform by inclinometer'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UAAA'
    var.attrs['sdn_uom_name'] = 'Degrees'
    var.attrs['standard_name'] = 'platform_pitch_angle'
    var.attrs['data_min'] = np.nanmin(vel.pitch)
    var.attrs['data_max'] = np.nanmax(vel.pitch)

    # ROLLGP01: roll
    var = out.ROLLGP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'degrees'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'roll'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::ROLL'
    var.attrs['sdn_parameter_name'] = 'Orientation (roll angle) of measurement platform by inclinometer (second sensor)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UAAA'
    var.attrs['sdn_uom_name'] = 'Degrees'
    var.attrs['standard_name'] = 'platform_roll_angle'
    var.attrs['data_min'] = np.nanmin(vel.roll)
    var.attrs['data_max'] = np.nanmax(vel.roll)

    # DISTTRAN: height of sea surface (hght)
    var = out.DISTTRAN
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'height of sea surface'
    var.attrs['generic_name'] = 'height'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['legency_GF3_code'] = 'SDN:GF3::HGHT'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::ULAA'
    var.attrs['sdn_uom_name'] = 'Metres'
    var.attrs['data_min'] = np.nanmin(DISTTRAN)
    var.attrs['data_max'] = np.nanmax(DISTTRAN)

    # TEMPPR01: transducer temp
    var = out.TEMPPR01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'degrees celsius'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP Transducer Temp.'
    var.attrs['generic_name'] = 'temp'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['legency_GF3_code'] = 'SDN:GF3::te90'
    var.attrs['sdn_parameter_name'] = 'Temperature of the water body'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UPAA'
    var.attrs['sdn_uom_name'] = 'Celsius degree'
    var.attrs['data_min'] = np.nanmin(vel.temperature)
    var.attrs['data_max'] = np.nanmax(vel.temperature)

    # PPSAADCP: instrument depth (formerly DEPFP01)
    var = out.PPSAADCP
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'instrument depth'
    var.attrs['xducer_offset_from_bottom'] = ''
    var.attrs['bin_size'] = data.CellSize  # bin size
    var.attrs['generic_name'] = 'depth'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['legency_GF3_code'] = 'SDN:GF3::DEPH'
    var.attrs['sdn_parameter_name'] = 'Depth below surface of the water body'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::ULAA'
    var.attrs['sdn_uom_name'] = 'Metres'
    var.attrs['standard_name'] = 'depth'
    var.attrs['data_min'] = np.nanmin(depth)
    var.attrs['data_max'] = np.nanmax(depth)

    # ALONZZ01, longitude
    for var in [out.ALONZZ01, out.longitude]:
        var.encoding['_FillValue'] = None
        var.encoding['dtype'] = 'd'
        var.attrs['units'] = 'degrees_east'
        var.attrs['long_name'] = 'longitude'
        var.attrs['legency_GF3_code'] = 'SDN:GF3::lon'
        var.attrs['sdn_parameter_name'] = 'Longitude east'
        var.attrs['sdn_uom_urn'] = 'SDN:P06::DEGE'
        var.attrs['sdn_uom_name'] = 'Degrees east'
        var.attrs['standard_name'] = 'longitude'

    # ALATZZ01, latitude
    for var in [out.ALATZZ01, out.latitude]:
        var.encoding['_FillValue'] = None
        var.encoding['dtype'] = 'd'
        var.attrs['units'] = 'degrees_north'
        var.attrs['long_name'] = 'latitude'
        var.attrs['legency_GF3_code'] = 'SDN:GF3::lat'
        var.attrs['sdn_parameter_name'] = 'Latitude north'
        var.attrs['sdn_uom_urn'] = 'SDN:P06::DEGN'
        var.attrs['sdn_uom_name'] = 'Degrees north'
        var.attrs['standard_name'] = 'latitude'

    # HEADCM01: heading
    var = out.HEADCM01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'degrees'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'heading'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['legency_GF3_code'] = 'SDN:GF3::HEAD'
    var.attrs['sdn_parameter_name'] = 'Orientation (horizontal relative to true north) of measurement device {heading}'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UAAA'
    var.attrs['sdn_uom_name'] = 'Degrees'
    var.attrs['data_min'] = np.nanmin(vel.heading)
    var.attrs['data_max'] = np.nanmax(vel.heading)

    # PRESPR01: pressure
    var = out.PRESPR01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'decibars'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'pressure'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['legency_GF3_code'] = 'SDN:GF3::PRES'
    var.attrs['sdn_parameter_name'] = 'Pressure (spatial co-ordinate) exerted by the water body by profiling ' \
                                      'pressure sensor and corrected to read zero at sea level'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UPDB'
    var.attrs['sdn_uom_name'] = 'Decibars'
    var.attrs['standard_name'] = 'sea_water_pressure'
    var.attrs['data_min'] = np.nanmin(pressure)
    var.attrs['data_max'] = np.nanmax(pressure)

    # SVELCV01: sound velocity
    var = out.SVELCV01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm/sec'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'speed of sound'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['legency_GF3_code'] = 'SDN:GF3::SVEL'
    var.attrs['sdn_parameter_name'] = 'Sound velocity in the water body by computation from temperature and ' \
                                      'salinity by unspecified algorithm'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['standard_name'] = 'speed_of_sound_in_sea_water'
    var.attrs['data_min'] = np.nanmin(sound_speed)
    var.attrs['data_max'] = np.nanmax(sound_speed)

    # DTUT8601: time values as ISO8601 string, YY-MM-DD hh:mm:ss
    var = out.DTUT8601
    var.encoding['dtype'] = 'U24'  # 24-character string
    var.attrs['note'] = 'time values as ISO8601 string, YY-MM-DD hh:mm:ss'
    var.attrs['time_zone'] = 'UTC'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::time_string'
    var.attrs['sdn_parameter_name'] = 'String corresponding to format \'YYYY-MM-DDThh:mm:ss.sssZ\' or other ' \
                                      'valid ISO8601 string'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::TISO'
    var.attrs['sdn_uom_name'] = 'ISO8601'

    # CMAGZZ01-4: correlation magnitude
    var = out.CMAGZZ01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_1'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'CM'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::CMAG_01'
    var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by ' \
                                      'moored acoustic doppler current profiler (ADCP) beam 1'
    var.attrs['data_min'] = np.nanmin(cor.cor1)
    var.attrs['data_max'] = np.nanmax(cor.cor1)

    var = out.CMAGZZ02
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_2'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'CM'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::CMAG_02'
    var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by ' \
                                      'moored acoustic doppler current profiler (ADCP) beam 2'
    var.attrs['data_min'] = np.nanmin(cor.cor2)
    var.attrs['data_max'] = np.nanmax(cor.cor2)

    var = out.CMAGZZ03
    var.attrs['units'] = 'counts'
    var.encoding['dtype'] = 'float32'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_3'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'CM'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::CMAG_03'
    var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by ' \
                                      'moored acoustic doppler current profiler (ADCP) beam 3'
    var.attrs['data_min'] = np.nanmin(cor.cor3)
    var.attrs['data_max'] = np.nanmax(cor.cor3)

    var = out.CMAGZZ04
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'counts'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_4'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['sensor_depth'] = sensor_dep
    var.attrs['serial_number'] = meta_dict['serialNumber']
    var.attrs['generic_name'] = 'CM'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::CMAG_04'
    var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by ' \
                                      'moored acoustic doppler current profiler (ADCP) beam 4'
    var.attrs['data_min'] = np.nanmin(cor.cor4)
    var.attrs['data_max'] = np.nanmax(cor.cor4)
    # done variables

    # Global attributes

    # Attributes not from metadata file:
    out.attrs['processing_history'] = processing_history
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
    out.attrs['manufacturer'] = manufacturer
    out.attrs['source'] = "R code: adcpProcess, github:"
    now = datetime.datetime.now()
    out.attrs['date_modified'] = now.strftime("%Y-%m-%d %H:%M:%S")
    out.attrs['_FillValue'] = str(fillValue)
    out.attrs['featureType'] = "profileTimeSeries"
    out.attrs['firmware_version'] = str(vel.FL.FWV) + '.' + str(vel.FL.FWR)  # firmwareVersion
    out.attrs['frequency'] = str(data.sysconfig['kHz'])
    out.attrs['beam_pattern'] = beamPattern
    out.attrs['beam_angle'] = str(fixed_leader.sysconfig['angle'])  # beamAngle
    out.attrs['systemConfiguration'] = bin(fixed_leader.FL['SysCfg'])[-8:] + '-' + bin(fixed_leader.FL['SysCfg'])[
                                                                                   :9].replace('b', '')
    out.attrs['sensor_source'] = '{0:08b}'.format(vel.FL['EZ'])  # sensorSource
    out.attrs['sensors_avail'] = '{0:08b}'.format(vel.FL['SA'])  # sensors_avail
    out.attrs['three_beam_used'] = str(vel.trans['threebeam']).upper()  # netCDF4 file format doesn't support bool
    out.attrs['valid_correlation_range'] = vel.FL['LowCorrThresh']  # lowCorrThresh
    out.attrs['minmax_percent_good'] = "100"  # hardcoded in oceNc_create()
    out.attrs['error_velocity_threshold'] = "2000 m/sec"
    out.attrs['false_target_reject_values'] = 50  # falseTargetThresh
    out.attrs['data_type'] = "adcp"
    out.attrs['pred_accuracy'] = 1  # velocityResolution * 1000
    out.attrs['Conventions'] = "CF-1.9"
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
    out.attrs['orientation'] = orientation

    # sensor_depth is a variable attribute, not a global attribute
    if out.attrs['orientation'] == 'up':
        out.attrs['geospatial_vertical_min'] = sensor_dep - np.nanmax(distance)
        out.attrs['geospatial_vertical_max'] = sensor_dep - np.nanmin(distance)
    elif out.attrs['orientation'] == 'down':
        out.attrs['geospatial_vertical_min'] = sensor_dep + np.nanmin(distance)
        out.attrs['geospatial_vertical_max'] = sensor_dep + np.nanmax(distance)

    # Add select meta_dict items as global attributes
    for key, value in meta_dict.items():
        if key == 'cut_lead_ensembles' or key == 'cut_trail_ensembles' or key == 'processing_level':
            pass
        elif key == 'serialNumber':
            out.attrs['serial_number'] = value
        else:
            out.attrs[key] = value

    # Export the 'out' object as a netCDF file
    print(outname)

    out.to_netcdf(outname, mode='w', format='NETCDF4')
    out.close()

    return


# Call function
nc_create_L1(inFile=raw_file, file_meta=raw_file_meta, start_year=None, time_file=None)

