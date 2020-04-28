"""
author: Hana Hourston
date: Jan. 23, 2020

about: This script is adapted from Jody Klymak's at https://gist.github.com/jklymak/b39172bd0f7d008c81e32bf0a72e2f09
for L1 processing raw ADCP data.

Contributions from: Di Wan, Eric Firing

User input (4 places) needed after var_to2d() function
"""

import os
import csv
import numpy as np
import xarray as xr
import pandas as pd
import datetime
from pycurrents.adcp.rdiraw import rawfile
from pycurrents.adcp.rdiraw import SysCfg
import pycurrents.adcp.transform as transform
import gsw
from num2words import num2words
import statistics


# This prints out the FileBBWHOS() function code. rdiraw.rawfile() calls rdiraw.FileBBWHOS()
# import inspect
# print(inspect.getsource(rdiraw.FileBBWHOS))


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


# Di Wan's magnetic declination correction code: ADJUST ANGLE: TAKES 0 DEG FROM E-W AXIS
def correct_true_north(mag_decl, measured_east, measured_north): #change angle to negative of itself
    angle_rad = -mag_decl * np.pi/180.
    east_true = measured_east * np.cos(angle_rad) - measured_north * np.sin(angle_rad)
    north_true = measured_east * np.sin(angle_rad) + measured_north * np.cos(angle_rad)
    return east_true, north_true


# Reshape 3d numeric variables to include 'station' dimension
def var_to3d(variable):
    return np.reshape(variable.transpose(), (1, len(vel.dep), len(variable))) # (16709, 31) to (1, 31, 16709)

# Reshape 2d numeric variables to include 'station' dimension
def var_to2d(variable):
    return np.reshape(variable, (1, len(variable)))


# User input

wd = 'your/wd/here'
os.chdir(wd)

# Specify raw ADCP file to create nc file from, along with associated csv metadata file and
# average magnetic declination over the timeseries

# 1) raw .000 file
inFile = 'your/path/here'
# 2) csv metadata file
file_meta = 'your/path/here'
# 3) average magnetic declination over the time series
magnetic_variation = ''


# Begin making netCDF file

# Splice file name to get output netCDF file name
outname = os.path.basename(inFile)[:-4] + '.adcp.L1.nc'; print(outname)

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
                elif row[1] == "Sentinel V":  #missing from documentation
                    model = "sv"
                    model_long = "RDI SV"
                    manufacturer = 'teledyne rdi'
                elif row[1] == "Multi-Cell Doppler Current Profiler":
                    model = ""
                    model_long = "Sontek MC DCP"
                    manufacturer = 'sontek'
                else:
                    continue
            else:
                continue
if model == "":
    print("No valid instrumentSubtype detected")


# Read in raw ADCP file and model type
data = rawfile(inFile, model)

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
meta_dict['station_number'] = int(meta_dict['station_number'])
meta_dict['water_depth'] = float(meta_dict['water_depth'])

# Update naming_authority from CF v52 to CF v72
meta_dict['naming_authority'] = 'BODC, MEDS, CF v72'

# Add leading zero to serial numbers that have 3 digits
if len(str(meta_dict['serialNumber'])) == 3:
    meta_dict['serialNumber'] = '0' + str(meta_dict['serialNumber'])
# Overwrite serial number to include the model: upper returns uppercase
meta_dict['serialNumber'] = model.upper() + meta_dict['serialNumber']
# Add instrument model variable value
meta_dict['instrumentModel'] = '{} ADCP {}kHz ({})'.format(model_long, data.sysconfig['kHz'], meta_dict['serialNumber'])

# Begin writing processing history, which will be added as a global attribute to the output netCDF file
processing_history = "Metadata read in from log sheet and combined with raw data to export as netCDF file."


# Extract metadata from data object

# Orientation code from Eric Firing:
orientations = [SysCfg(fl).up for fl in fixed_leader.raw.FixedLeader['SysCfg']]
orientation = mean_orientation(orientations)

# Retrieve beam pattern
if data.sysconfig['convex']:
    beamPattern = 'convex'
else:
    beamPattern = ''

# Get timestamp from "data" object just created
# In R, the start time was obtained from the "adp" object created within R
# data.yearbase is an integer of the year that the timeseries starts (e.g., 2016)
data_origin = pd.Timestamp(str(data.yearbase) + '-01-01') #convert to date object


# Set up dimensions and variables

# convert time variable to elapsed time since 1970-01-01T00:00:00Z; dtype='datetime64[ns]'
time_us = np.array(pd.to_datetime(vel.dday, unit='D', origin=data_origin, utc=True).strftime('%Y-%m-%d %H:%M:%S.%f'), dtype='datetime64')

# Station dimension
station = np.array([meta_dict['station_number']]) # Should dimensions be integers or arrays?

# nchar dimension; for consistency with R ADCP package -- can't add it as a dim if no variable has it as a dim
# nchar = np.array([range(24)])

# DTUT8601 should have dtype='|S23' ? this is where nchar=23 comes in?
time_DTUT8601 = pd.to_datetime(vel.dday, unit='D', origin=data_origin, utc=True).strftime('%Y-%m-%d %H:%M:%S') #don't need %Z in strftime

# Convert pressure
pressure = np.array(vel.VL['Pressure'] / 1000, dtype='float32') #convert decapascal to decibars

# Calculate pressure based on static instrument depth if missing pressure sensor; extra condition added for zero pressure or weird pressure values
if model == 'bb' or model == 'nb' or statistics.mode(pressure) == 0:
    z = meta_dict['instrument_depth']
    p = round(gsw.conversions.p_from_z(-meta_dict['instrument_depth'], meta_dict['latitude']), ndigits=0) #depth negative because positive is up for this function
    pressure = np.repeat(p, len(vel.vel1.data))
    processing_history = processing_history + " Pressure values calculated from static instrument depth ({} m) using " \
                                              "the TEOS-10 75-term expression for specific volume and rounded to {} " \
                                              "significant digits.".format(str(meta_dict['instrument_depth']), num2words(len(str(p))))


# Adjust velocity data

# Set velocity values of -32768.0 to nans, since they are NAs in R but they are -32768.0 here for some reason
vel.vel1.data[vel.vel1.data == -32768.0] = np.nan
vel.vel2.data[vel.vel2.data == -32768.0] = np.nan
vel.vel3.data[vel.vel3.data == -32768.0] = np.nan
vel.vel4.data[vel.vel4.data == -32768.0] = np.nan


# Rotate into earth **IF not in enu already; this makes the netCDF bigger
if vel.trans.coordsystem != 'earth' or vel.trans.coordsystem != 'enu':
    trans = transform.Transform(angle=vel.FL.BeamAngle, geometry=beamPattern) #angle is beam angle
    xyze = trans.beam_to_xyz(vel.vel) #
    print(np.shape(xyze))
    enu = transform.rdi_xyz_enu(xyze, vel.heading, vel.pitch, vel.roll, orientation='up')
    print(np.shape(enu))
    # Apply change in coordinates to velocities
    vel.vel1.data = xr.DataArray(enu[:, :, 0], dims=['station', 'distance', 'time'])
    vel.vel2.data = xr.DataArray(enu[:, :, 1], dims=['station', 'distance', 'time'])
    vel.vel3 = xr.DataArray(enu[:, :, 2], dims=['station', 'distance', 'time'])
    vel.vel4 = xr.DataArray(enu[:, :, 3], dims=['station', 'distance', 'time'])
    processing_history = processing_history + " The coordinate system was rotated into enu coordinates."
    meta_dict['coord_system'] = 'enu' #Add item to metadata dictionary for coordinate system
else:
    meta_dict['coord_system'] = 'enu'


# Correct magnetic declination in velocities; code from Di Wan
meta_dict['magnetic_variation'] = magnetic_variation
LCEWAP01, LCNSAP01 = correct_true_north(meta_dict['magnetic_variation'], vel.vel1.data, vel.vel2.data)
#LCEWAP01, LCNSAP01 = correct_true_north(10, vel.vel1.data, vel.vel2.data)
processing_history += " Magnetic variation, using average applied; declination = {}.".format(str(meta_dict['magnetic_variation']))


# Flag velocity data based on cut_lead_ensembles and cut_trail_ensembles

# Set start and end indices
e1 = int(meta_dict['cut_lead_ensembles']) #"ensemble 1"
e2 = int(meta_dict['cut_trail_ensembles']) #"ensemble 2"

# Create QC variables containing flag arrays
LCEWAP01_QC = np.zeros(shape=LCEWAP01.shape, dtype='float32')
LCNSAP01_QC = np.zeros(shape=LCNSAP01.shape, dtype='float32')
LRZAAP01_QC = np.zeros(shape=vel.vel3.data.shape, dtype='float32')

for qc in [LCEWAP01_QC, LCNSAP01_QC, LRZAAP01_QC]:
    # 0=no_quality_control, 4=value_seems_erroneous
    for b in range(data.NCells):
        qc[:e1, b] = 4
        qc[-e2:, b] = 4

# Apply the flags to the data and set bad data to NAs
LCEWAP01[LCEWAP01_QC == 4] = np.nan
LCNSAP01[LCNSAP01_QC == 4] = np.nan
vel.vel3.data[LRZAAP01_QC == 4] = np.nan
processing_history += " Quality control flags set based on SeaDataNet flag scheme from BODC."


# Depth

# Apply equivalent of swDepth() to depth data: Calculate height from sea pressure using gsw package
depth = -gsw.conversions.z_from_p(p=pressure, lat=meta_dict['latitude']) #negative so that depth is positive; units=m

# Limit depth by deployment and recovery times
# Create array of flags for depth time-limited
depth_tl = np.zeros(shape=depth.shape)
depth_tl[:e1] = 4
depth_tl[-e2:] = 4
depth[depth_tl == 4] = np.nan
processing_history += " Depth limited by deployment ({} UTC) and recovery ({} UTC) times.".format(time_DTUT8601[e1], time_DTUT8601[-e2])

# Calculate sensor depth of instrument based off mean instrument transducer depth
sensor_dep = np.nanmean(depth)
processing_history += " Sensor depth and mean depth set to {} based on trimmed depth values.".format(str(sensor_dep))

# Calculate height of sea surface: bin height minus sensor depth
DISTTRAN = vel.dep - sensor_dep

# Limit variables (pressure, temperature, pitch, roll, heading, soundspeed) from before deployment and after recovery of ADCP
pressure[:e1] = np.nan
pressure[-e2:] = np.nan
vel.temperature[:e1] = np.nan
vel.temperature[-e2:] = np.nan
vel.pitch[:e1] = np.nan
vel.pitch[-e2:] = np.nan
vel.roll[:e1] = np.nan
vel.roll[-e2:] = np.nan
vel.heading[:e1] = np.nan
vel.heading[-e2:] = np.nan
vel.VL['SoundSpeed'][:e1] = np.nan
vel.VL['SoundSpeed'][-e2:] = np.nan


processing_history += 'Level 1 processing was performed on the dataset. This entailed corrections for magnetic ' \
                      'declination based on an average of the dataset and cleaning of the beginning and end of ' \
                      'the dataset. No QC was carried out. The leading {} ensembles and the trailing {} ensembles ' \
                      'were removed from the data set.'.format(meta_dict['cut_lead_ensembles'], meta_dict['cut_trail_ensembles'])


# Make into netCDF file

# Create xarray Dataset object containing all dimensions and variables
out = xr.Dataset(coords={'time': time_us, 'distance': vel.dep, 'station': station},
                 data_vars={'LCEWAP01': (['station', 'distance', 'time'], var_to3d(LCEWAP01)),
                            'LCNSAP01': (['station', 'distance', 'time'], var_to3d(LCNSAP01)),
                            'LRZAAP01': (['station', 'distance', 'time'], var_to3d(vel.vel3.data)),
                            'LERRAP01': (['station', 'distance', 'time'], var_to3d(vel.vel4.data)),
                            'LCEWAP01_QC': (['station', 'distance', 'time'], var_to3d(LCEWAP01_QC)),
                            'LCNSAP01_QC': (['station', 'distance', 'time'], var_to3d(LCNSAP01_QC)),
                            'LRZAAP01_QC': (['station', 'distance', 'time'], var_to3d(LRZAAP01_QC)),
                            'ELTMEP01': (['station', 'time'], var_to2d(time_us)),
                            'TNIHCE01': (['station', 'distance', 'time'], var_to3d(amp.amp1)),
                            'TNIHCE02': (['station', 'distance', 'time'], var_to3d(amp.amp2)),
                            'TNIHCE03': (['station', 'distance', 'time'], var_to3d(amp.amp3)),
                            'TNIHCE04': (['station', 'distance', 'time'], var_to3d(amp.amp4)),
                            'CMAGZZ01': (['station', 'distance', 'time'], var_to3d(cor.cor1)),
                            'CMAGZZ02': (['station', 'distance', 'time'], var_to3d(cor.cor2)),
                            'CMAGZZ03': (['station', 'distance', 'time'], var_to3d(cor.cor3)),
                            'CMAGZZ04': (['station', 'distance', 'time'], var_to3d(cor.cor4)),
                            'PCGDAP00': (['station', 'distance', 'time'], var_to3d(pg.pg1)),
                            'PCGDAP02': (['station', 'distance', 'time'], var_to3d(pg.pg2)),
                            'PCGDAP03': (['station', 'distance', 'time'], var_to3d(pg.pg3)),
                            'PCGDAP04': (['station', 'distance', 'time'], var_to3d(pg.pg4)),
                            'PTCHGP01': (['station', 'time'], var_to2d(vel.pitch)),
                            'HEADCM01': (['station', 'time'], var_to2d(vel.heading)),
                            'ROLLGP01': (['station', 'time'], var_to2d(vel.roll)),
                            'TEMPPR01': (['station', 'time'], var_to2d(vel.temperature)),
                            'DISTTRAN': (['station', 'distance'], var_to2d(DISTTRAN)),
                            'PPSAADCP': (['station', 'time'], var_to2d(depth)),
                            'ALATZZ01': (['station'], np.array([float(meta_dict['latitude'])])),
                            'ALONZZ01': (['station'], np.array([float(meta_dict['longitude'])])),
                            'latitude': (['station'], np.array([float(meta_dict['latitude'])])),
                            'longitude': (['station'], np.array([float(meta_dict['longitude'])])),
                            'PRESPR01': (['station', 'time'], var_to2d(pressure)),
                            'SVELCV01': (['station', 'time'], var_to2d(vel.VL['SoundSpeed'])),
                            'DTUT8601': (['time'], time_DTUT8601), #removed nchar dim
                            'filename': (['station'], np.array([outname[:-3]])), #changed dim from nchar to station
                            'instrument_serial_number': (['station'], np.array([meta_dict['serialNumber']])),
                            'instrument_model': (['station'], np.array([meta_dict['instrumentModel']]))})


# Add attributes to each variable

# making lists of variables that need the same attributes could help shorten this part of the script, but how?
# it may also make it harder to rename variables in the future...

fillValue = 1e+15

# Time
var = out.time
var.encoding['units'] = "seconds since 1970-01-01T00:00:00Z"
var.encoding['_FillValue'] = None #omits fill value from time dimension; otherwise would include with value NaN
var.attrs['long_name'] = "time"
var.attrs['cf_role'] = "profile_id"
var.encoding['calendar'] = "gregorian"

# Bin distances
var = out.distance
var.encoding['_FillValue'] = None
var.attrs['units'] = "metres"
#var.attrs['long_name'] = "distance"
var.attrs['long_name'] = "bin_distances_from_ADCP_transducer_along_measurement_axis"

# Station
var = out.station
var.encoding['_FillValue'] = None
var.encoding['dtype'] = 'd' #double type
var.attrs['long_name'] = "station"
var.attrs['cf_role'] = "timeseries_id"
var.attrs['standard_name'] = "platform_name"
var.attrs['longitude'] = float(meta_dict['longitude'])
var.attrs['latitude'] = float(meta_dict['latitude'])

# LCEWAP01: eastward velocity (vel1); all velocities have many overlapping attribute values (but not all)
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
var.attrs['sdn_parameter_name'] = 'Eastward current velocity (Eulerian measurement) in the water body by moored acoustic doppler current profiler (ADCP)'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
var.attrs['sdn_uom_name'] = 'Metres per second'
var.attrs['standard_name'] = 'eastward_sea_water_velocity'
var.attrs['data_max'] = round(np.nanmax(LCEWAP01), 14) #doesn't include leading and trailing ensembles that were set to NAs
var.attrs['data_min'] = round(np.nanmin(LCEWAP01), 14)
var.attrs['valid_max'] = 1000
var.attrs['valid_min'] = -1000

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
var.attrs['sdn_parameter_name'] = 'Northward current velocity (Eulerian measurement) in the water body by moored acoustic doppler current profiler (ADCP)'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
var.attrs['sdn_uom_name'] = 'Metres per second'
var.attrs['standard_name'] = 'northward_sea_water_velocity'
var.attrs['data_max'] = round(np.nanmax(LCNSAP01), 14)
var.attrs['data_min'] = round(np.nanmin(LCNSAP01), 14)
var.attrs['valid_max'] = 1000
var.attrs['valid_min'] = -1000

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
var.attrs['sdn_parameter_name'] = 'Upward current velocity (Eulerian measurement) in the water body by moored acoustic doppler current profiler (ADCP)'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
var.attrs['sdn_uom_name'] = 'Metres per second'
var.attrs['standard_name'] = 'upward_sea_water_velocity'
var.attrs['data_max'] = np.nanmax(vel.vel3)
var.attrs['data_min'] = np.nanmin(vel.vel3)
var.attrs['valid_max'] = 1000
var.attrs['valid_min'] = -1000

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
var.attrs['sdn_parameter_name'] = 'Current velocity error in the water body by moored acoustic doppler current profiler (ADCP)'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
var.attrs['sdn_uom_name'] = 'Metres per second'
var.attrs['data_max'] = np.nanmax(vel.vel4)
var.attrs['data_min'] = np.nanmin(vel.vel4)
var.attrs['valid_max'] = 2000
var.attrs['valid_min'] = -2000

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
var.encoding['units'] = 'seconds since 1970-01-01T00:00:00Z' #was var.attrs
var.attrs['_FillValue'] = fillValue
var.attrs['long_name'] = 'time_02'
var.attrs['legency_GF3_code'] = 'SDN:GF3::N/A'
var.attrs['sdn_parameter_name'] = 'Elapsed time (since 1970-01-01T00:00:00Z)'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UTBB'
var.attrs['sdn_uom_name'] = 'Seconds'
var.attrs['standard_name'] = 'time'

# TNIHCE01: echo intensity beam 1
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
var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current profiler (ADCP) beam 1'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
var.attrs['sdn_uom_name'] = 'Counts'
var.attrs['data_min'] = np.nanmin(amp.amp1)
var.attrs['data_max'] = np.nanmax(amp.amp1)

# TNIHCE02: echo intensity beam 2
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
var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current profiler (ADCP) beam 2'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
var.attrs['sdn_uom_name'] = 'Counts'
var.attrs['data_min'] = np.nanmin(amp.amp2)
var.attrs['data_max'] = np.nanmax(amp.amp2)

# TNIHCE03: echo intensity beam 3
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
var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current profiler (ADCP) beam 3'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
var.attrs['sdn_uom_name'] = 'Counts'
var.attrs['data_min'] = np.nanmin(amp.amp3)
var.attrs['data_max'] = np.nanmax(amp.amp3)

# TNIHCE04: echo intensity beam 4
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
var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current profiler (ADCP) beam 4'
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
var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler current profiler (ADCP) beam 1'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
var.attrs['sdn_uom_name'] = 'Percent'
var.attrs['data_min'] = np.nanmin(pg.pg1)
var.attrs['data_max'] = np.nanmax(pg.pg1)

# PCGDAP02: percent good beam 2
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
var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler current profiler (ADCP) beam 2'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
var.attrs['sdn_uom_name'] = 'Percent'
var.attrs['data_min'] = np.nanmin(pg.pg2)
var.attrs['data_max'] = np.nanmax(pg.pg2)

# PCGDAP03: percent good beam 3
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
var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler current profiler (ADCP) beam 3'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
var.attrs['sdn_uom_name'] = 'Percent'
var.attrs['data_min'] = np.nanmin(pg.pg3)
var.attrs['data_max'] = np.nanmax(pg.pg3)

# PCGDAP03: percent good beam 4
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
var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler current profiler (ADCP) beam 4'
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

# DISTTRAN: height of sea surface
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
var.attrs['data_min'] = np.nanmin(vel.dep)
var.attrs['data_max'] = np.nanmax(vel.dep)

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
var.attrs['bin_size'] = data.CellSize #bin size
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
var.attrs['sdn_parameter_name'] = 'Pressure (spatial co-ordinate) exerted by the water body by profiling pressure sensor and corrected to read zero at sea level'
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
var.attrs['sdn_parameter_name'] = 'Sound velocity in the water body by computation from temperature and salinity by unspecified algorithm'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
var.attrs['sdn_uom_name'] = 'Metres per second'
var.attrs['standard_name'] = 'speed_of_sound_in_sea_water'
var.attrs['data_min'] = np.nanmin(vel.VL['SoundSpeed'])
var.attrs['data_max'] = np.nanmax(vel.VL['SoundSpeed'])

# DTUT8601: time values as ISO8601 string, YY-MM-DD hh:mm:ss #How to make into char dtype?
var = out.DTUT8601
var.encoding['dtype'] = 'U24' #24-character string
var.attrs['note'] = 'time values as ISO8601 string, YY-MM-DD hh:mm:ss'
var.attrs['time_zone'] = 'UTC'
var.attrs['legency_GF3_code'] = 'SDN:GF3::time_string'
var.attrs['sdn_parameter_name'] = 'String corresponding to format \'YYYY-MM-DDThh:mm:ss.sssZ\' or other valid ISO8601 string'
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
var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by moored acoustic doppler current profiler (ADCP) beam 1'
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
var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by moored acoustic doppler current profiler (ADCP) beam 2'
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
var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by moored acoustic doppler current profiler (ADCP) beam 3'
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
var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by moored acoustic doppler current profiler (ADCP) beam 4'
var.attrs['data_min'] = np.nanmin(cor.cor4)
var.attrs['data_max'] = np.nanmax(cor.cor4)
#done variables


# Global attributes

# Attributes not from metadata file:
out.attrs['processing_history'] = processing_history
out.attrs['time_coverage_duration'] = vel.dday[-1]-vel.dday[0]
out.attrs['time_coverage_duration_units'] = "days"
#^calculated from start and end times; in days: add time_coverage_duration_units?
out.attrs['cdm_data_type'] = "station"
out.attrs['number_of_beams'] = data.NBeams #change in python and R to numberOfBeams? from 'janus' -- .adcp files have 'numbeams'
#out.attrs['nprofs'] = data.nprofs #number of ensembles
out.attrs['numberOfCells'] = data.NCells
out.attrs['pings_per_ensemble'] = data.NPings
out.attrs['bin1Distance'] = data.Bin1Dist
#out.attrs['Blank'] = data.Blank #?? blanking distance?
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

# Metadata from the data object
out.attrs['firmware_version'] = str(vel.FL.FWV) + '.' + str(vel.FL.FWR) #firmwareVersion
out.attrs['frequency'] = str(data.sysconfig['kHz'])
out.attrs['beam_pattern'] = beamPattern
out.attrs['beam_angle'] = str(vel.FL.BeamAngle) #beamAngle
out.attrs['systemConfiguration'] = bin(fixed_leader.FL['SysCfg'])[-8:] + '-' + bin(fixed_leader.FL['SysCfg'])[:9].replace('b', '')
out.attrs['sensor_source'] = '{0:08b}'.format(vel.FL['EZ']) #sensorSource
out.attrs['sensors_avail'] = '{0:08b}'.format(vel.FL['SA']) #sensors_avail
out.attrs['three_beam_used'] = str(vel.trans['threebeam']).upper() #netCDF4 file format doesn't support booleans
out.attrs['valid_correlation_range'] = vel.FL['LowCorrThresh'] #lowCorrThresh
out.attrs['minmax_percent_good'] = "100" #hardcoded in oceNc_create(); should be percentGdMinimum?
out.attrs['error_velocity_threshold'] = "2000 m/sec"
#out.attrs['transmit_pulse_length_cm'] = '' #errorVelocityMaximum + m/s
out.attrs['false_target_reject_values'] = 50 #falseTargetThresh
out.attrs['data_type'] = "adcp"
out.attrs['pred_accuracy'] = 1 #where does this come from? velocityResolution * 1000
out.attrs['Conventions'] = "CF-1.7"
out.attrs['creator_type'] = "person"
out.attrs['n_codereps'] = vel.FL.NCodeReps
out.attrs['xmit_lag'] = vel.FL.TransLag
out.attrs['time_coverage_start'] = time_DTUT8601[e1] + ' UTC'
out.attrs['time_coverage_end'] = time_DTUT8601[-e2 - 1] + ' UTC' #-1 is last time entry before cut ones

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
    out.attrs['geospatial_vertical_min'] = sensor_dep - np.nanmax(vel.dep)
    out.attrs['geospatial_vertical_max'] = sensor_dep - np.nanmin(vel.dep)
elif out.attrs['orientation'] == 'down':
    out.attrs['geospatial_vertical_min'] = sensor_dep + np.nanmin(vel.dep)
    out.attrs['geospatial_vertical_max'] = sensor_dep + np.nanmax(vel.dep)

# Add select meta_dict items as global attributes
for k, v in meta_dict.items():
    if k == 'cut_lead_ensembles' or k == 'cut_trail_ensembles' or k == 'processing_level':
        pass
    elif k == 'serialNumber':
        out.attrs['serial_number'] = v
    else:
        out.attrs[k] = v


# Export the 'out' object as a netCDF file
print(outname)
out.to_netcdf(outname, mode='w', format='NETCDF4')
out.close()

# Use exclamation mark to access the underlying shell
#!ls - halt


