"""
author: Hana Hourston
date: Jan. 23, 2020

about: This script is adapted from Jody Klymak's at https://gist.github.com/jklymak/b39172bd0f7d008c81e32bf0a72e2f09 for processing raw ADCP data.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import datetime
from pycurrents.adcp.rdiraw import Multiread

#import inspect
#print(inspect.getsource(Multiread)) #this prints out the Multiread() function code


# Specify raw ADCP file to create nc file from
inFile = "/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/callR_fromPython/20568-A1-56.000" #raw .000 file
file_meta = "/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/ADCP/a1_20160713_20170513_0480m/P01/a1_20160713_20170513_0480m_meta_L1.csv" #csv metadata file

inFile = '/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/ADCP/fortune1_20171007_20180714_0090m/fortune1_20171007_20180714_0090m.pd0'
file_meta = '/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/ADCP/fortune1_20171007_20180714_0090m/P01/fortune1_20171007_20180714_0090m_meta_L1.csv'

# Splice file name to get output netCDF file name
outname = os.path.basename(inFile)[:-3] + 'adcp.L1.nc'; print(outname)

# Get model and timestamp from raw file csv metadata file
# Instrument frequency is not included in the metadata files, so not included in "model". Frequency optional, apparently
meta_dict = {}
model = ""
with open(file_meta) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
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
    

# Multiread(fnames, sonar, yearbase=None, alias=None, gap=None, min_nprofs=1, gbinsonar=None, ibad=None)
# sonar: 'nb', 'bb', 'wh', or 'os' optionally followed by freq and pingtype, or a Sonar instance
# os = "Ocean Surveyor"
m = Multiread(inFile, model)  #m outputs $ <pycurrents.adcp.rdiraw.Multiread at 0x7f05b6eecd10>
data = m.read()  #data outputs the kind of output we're looking for

from pycurrents.adcp.transform import heading_rotate
# import pycurrents.adcp.adcp_nc as adcp_nc
# from pycurrents.adcp.transform import rdi_xyz_enu
import pycurrents.adcp.transform as transform

# Set up dimensions
station = int(meta_dict['station_number']) # Should dimensions be integers or arrays?
nchar = np.array(range(1,100)) #was (1,24) which was the same as R code originally

# Set up variables that need it

# Get timestamp from "data" object just created
# In R, the start time was obtained from the "adp" object created within R
# data.yearbase is an integer of the year that the timeseries starts (e.g., 2016)
data_origin = str(data.yearbase) + '-01-01' #convert to date object; NEED TIME ZONE = UTC

#time = pd.to_datetime(data.dday, unit='D', origin=pd.Timestamp('2016-01-01')) #original code line
time_DTUT8601 = pd.to_datetime(data.dday, unit='D', origin=data_origin, utc=True).strftime('%Y-%m-%d %H:%M:%S %Z')

# convert time variable to seconds since 1970-01-01T00:00:00Z ##########################
time_sec = time_DTUT8601.timestamp() #have to iterate through?

# Overwrite serial number to include the model:
meta_dict['serialNumber'] = model.upper() + meta_dict['serialNumber']

# what is a masked array? bt_depth

# Unpack variables in VL numpy.void object variable
svel = np.zeros(shape=(len(data.VL),))
head = np.zeros(shape=(len(data.VL),))
ptch = np.zeros(shape=(len(data.VL),))
roll = np.zeros(shape=(len(data.VL),))
pres = np.zeros(shape=(len(data.VL),))
sal = np.zeros(shape=(len(data.VL),))
temp = np.zeros(shape=(len(data.VL),))
for i in range(len(data.VL)):
    svel[i] = data.VL[i]['SoundSpeed']
    pres[i] = data.VL[i]['Pressure']
    sal[i] = data.VL[i]['Salinity']

# Make into netCDF file
# unknown items in data.VL: EnsNumMSB (ensemble number ?), BIT, MPT_minutes/seconds/hundredths, ADC0-ADC7, ESW, spare1, spare2, RTCCentury/.../hundredths     

out = xr.Dataset(coords={'distance': data.dep, 'time': time_sec, 'station': station, 'nchar': nchar},
                 data_vars={'time': (['time'], time_sec),
                            'distance': (['distance'], data.dep),
                            'station': (['station'], station),
                            'LCEWAP01': (['station', 'distance', 'time'], data.vel1),
                            'LCNSAP01': (['station', 'distance', 'time'], data.vel2),
                            'LRZAAP01': (['station', 'distance', 'time'], data.vel3),
                            'LERRAP01': (['station', 'distance', 'time'], data.vel4),
                            'ELTMEP01': (['time', 'station'], time_sec),
                            'TNIHCE01': (['station', 'distance', 'time'], data.amp1),
                            'TNIHCE02': (['station', 'distance', 'time'], data.amp2),
                            'TNIHCE03': (['station', 'distance', 'time'], data.amp3),
                            'TNIHCE04': (['station', 'distance', 'time'], data.amp4),
                            'CMAGZZ01': (['station', 'distance', 'time'], data.cor1),
                            'CMAGZZ02': (['station', 'distance', 'time'], data.cor2),
                            'CMAGZZ03': (['station', 'distance', 'time'], data.cor3),
                            'CMAGZZ04': (['station', 'distance', 'time'], data.cor4),
                            'PCGDAP00': (['station', 'distance', 'time'], data.pg1),
                            'PCGDAP02': (['station', 'distance', 'time'], data.pg2),
                            'PCGDAP03': (['station', 'distance', 'time'], data.pg3),
                            'PCGDAP04': (['station', 'distance', 'time'], data.pg4),
                            'PTCHGP01': (['station', 'time'], data.pitch),
                            'HEADCM01': (['station', 'time'], data.heading),
                            'ROLLGP01': (['station', 'time'], data.roll),
                            'TEMPPR01': (['station', 'time'], data.temperature),
                            'DISTTRAN': (['station', 'distance'], data.dep),
                            'PPSAADCP': (['station', 'time'], data.XducerDepth),
                            'ALATZZ01': (['station'], np.array([float(meta_dict['latitude'])])),
                            'ALONZZ01': (['station'], np.array([float(meta_dict['longitude'])])),
                            'latitude': (['station'], np.array([float(meta_dict['latitude'])])),
                            'longitude': (['station'], np.array([float(meta_dict['longitude'])])),
                            'PRESPR01': (['station', 'time'], pres),
                            'SVELCV01': (['station', 'time'], svel),
                            'DTUT8601': (['time', 'nchar'], time_DTUT8601),
                            'filename': (['station', 'nchar'], np.array([outname[-4]])),
                            'instrument_serial_number': (['station', 'nchar'], np.array([meta_dict['serialNumber']])),
                            'instrument_model': (['station', 'nchar'], np.array([model_long+meta_dict['serialNumber']]))})

# Add attributes to each variable:
# making lists of variables that need the same attributes could help shorten this part of the script, but how?
# it may also make it harder to rename variables in the future...
# Time
out.time.attrs['units'] = "seconds since 1970-01-01T00:00:00Z"
out.time.attrs['long_name'] = "time"
out.time.attrs['cf_role'] = "profile_id"
out.time.attrs['calendar'] = "gregorian"

# Bin distances
out.distance.attrs['units'] = "metres"
out.distance.attrs['long_name'] = "distance"

# Station
out.station.attrs['long_name'] = "station"
out.station.attrs['cf_role'] = "timeseries_id"
out.station.attrs['standard_name'] = "platform_name"
out.station.attrs['longitude'] = float(meta_dict['longitude'])
out.station.attrs['latitude'] = float(meta_dict['latitude'])

# LCEWAP01: eastward velocity (vel1); all velocities have many overlapping attribute values (but not all)
# try:
# var = out.LCEWAP01
# var.attrs[''] = '' ... etc.
out.LCEWAP01.attrs['units'] = 'm/sec'
out.LCEWAP01.attrs['_FillValue'] = '1e35'
out.LCEWAP01.attrs['long_name'] = 'eastward_sea_water_velocity'
out.LCEWAP01.attrs['sensor_type'] = 'adcp'
out.LCEWAP01.attrs['sensor_depth'] = '' ############################
out.LCEWAP01.attrs['serial_number'] = meta_dict['serialNumber']
out.LCEWAP01.attrs['generic_name'] = 'u'
out.LCEWAP01.attrs['flag_meanings'] = meta_dict['flag_meaning']
out.LCEWAP01.attrs['flag_values'] = meta_dict['flag_values']
out.LCEWAP01.attrs['References'] = meta_dict['flag_references']
out.LCEWAP01.attrs['legency_GF3_code'] = 'SDN:GF3::EWCT'
out.LCEWAP01.attrs['sdn_parameter_name'] = 'Eastward current velocity (Eulerian measurement) in the water body by moored acoustic doppler current profiler (ADCP)'
out.LCEWAP01.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
out.LCEWAP01.attrs['sdn_uom_name'] = 'Metres per second'
out.LCEWAP01.attrs['standard_name'] = 'eastward_sea_water_velocity'
out.LCEWAP01.attrs['data_max'] = np.nanmax(data.vel1)
out.LCEWAP01.attrs['data_min'] = np.nanmin(data.vel1)
out.LCEWAP01.attrs['valid_max'] = 1000
out.LCEWAP01.attrs['valid_min'] = -1000

# LCNSAP01: northward velocity (vel2)
out.LCNSAP01.attrs['units'] = 'm/sec'
out.LCNSAP01.attrs['_FillValue'] = '1e35'
out.LCNSAP01.attrs['long_name'] = 'northward_sea_water_velocity'
out.LCNSAP01.attrs['sensor_type'] = 'adcp'
out.LCNSAP01.attrs['sensor_depth'] = '' ############################
out.LCNSAP01.attrs['serial_number'] = meta_dict['serialNumber']
out.LCNSAP01.attrs['generic_name'] = 'v'
out.LCNSAP01.attrs['flag_meanings'] = meta_dict['flag_meaning']
out.LCNSAP01.attrs['flag_values'] = meta_dict['flag_values']
out.LCNSAP01.attrs['References'] = meta_dict['flag_references']
out.LCNSAP01.attrs['legency_GF3_code'] = 'SDN:GF3::NSCT'
out.LCNSAP01.attrs['sdn_parameter_name'] = 'Northward current velocity (Eulerian measurement) in the water body by moored acoustic doppler current profiler (ADCP)'
out.LCNSAP01.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
out.LCNSAP01.attrs['sdn_uom_name'] = 'Metres per second'
out.LCNSAP01.attrs['standard_name'] = 'northward_sea_water_velocity'
out.LCNSAP01.attrs['data_max'] = np.nanmax(data.vel2)
out.LCNSAP01.attrs['data_min'] = np.nanmin(data.vel2)
out.LCEWAP01.attrs['valid_max'] = 1000
out.LCEWAP01.attrs['valid_min'] = -1000

# LRZAAP01: vertical velocity (vel3)
out.LRZAAP01.attrs['units'] = 'm/sec'
out.LRZAAP01.attrs['_FillValue'] = '1e35'
out.LRZAAP01.attrs['long_name'] = 'upward_sea_water_velocity'
out.LRZAAP01.attrs['sensor_type'] = 'adcp'
out.LRZAAP01.attrs['sensor_depth'] = '' ############################
out.LRZAAP01.attrs['serial_number'] = meta_dict['serialNumber']
out.LRZAAP01.attrs['generic_name'] = 'w'
out.LRZAAP01.attrs['flag_meanings'] = meta_dict['flag_meaning']
out.LRZAAP01.attrs['flag_values'] = meta_dict['flag_values']
out.LRZAAP01.attrs['References'] = meta_dict['flag_references']
out.LRZAAP01.attrs['legency_GF3_code'] = 'SDN:GF3::VCSP'
out.LRZAAP01.attrs['sdn_parameter_name'] = 'Upward current velocity (Eulerian measurement) in the water body by moored acoustic doppler current profiler (ADCP)'
out.LRZAAP01.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
out.LRZAAP01.attrs['sdn_uom_name'] = 'Metres per second'
out.LRZAAP01.attrs['standard_name'] = 'upward_sea_water_velocity'
out.LRZAAP01.attrs['data_max'] = np.nanmax(data.vel3)
out.LRZAAP01.attrs['data_min'] = np.nanmin(data.vel3)
out.LRZAAP01.attrs['valid_max'] = 1000
out.LRZAAP01.attrs['valid_min'] = -1000

# LERRAP01: error velocity (vel4)
out.LERRAP01.attrs['units'] = 'm/sec'
out.LERRAP01.attrs['_FillValue'] = '1e35'
out.LERRAP01.attrs['long_name'] = 'error_velocity_in_sea_water'
out.LERRAP01.attrs['sensor_type'] = 'adcp'
out.LERRAP01.attrs['sensor_depth'] = '' ############################
out.LERRAP01.attrs['serial_number'] = meta_dict['serialNumber']
out.LERRAP01.attrs['generic_name'] = 'e'
out.LERRAP01.attrs['flag_meanings'] = meta_dict['flag_meaning']
out.LERRAP01.attrs['flag_values'] = meta_dict['flag_values']
out.LERRAP01.attrs['References'] = meta_dict['flag_references']
out.LERRAP01.attrs['legency_GF3_code'] = 'SDN:GF3::ERRV'
out.LERRAP01.attrs['sdn_parameter_name'] = 'Current velocity error in the water body by moored acoustic doppler current profiler (ADCP)'
out.LERRAP01.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
out.LERRAP01.attrs['sdn_uom_name'] = 'Metres per second'
out.LERRAP01.attrs['data_max'] = np.nanmax(data.vel4)
out.LERRAP01.attrs['data_min'] = np.nanmin(data.vel4)
out.LERRAP01.attrs['valid_max'] = 2000
out.LERRAP01.attrs['valid_min'] = -2000

# ELTMEP01: seconds since 1970
out.ELTMEP01.attrs['units'] = 'seconds since 1970-01-01T00:00:00Z'
out.ELTMEP01.attrs['_FillValue'] = '1e35'
out.ELTMEP01.attrs['long_name'] = 'time_02'
out.ELTMEP01.attrs['legency_GF3_code'] = 'SDN:GF3::N/A'
out.ELTMEP01.attrs['sdn_parameter_name'] = 'Elapsed time (since 1970-01-01T00:00:00Z)'
out.ELTMEP01.attrs['sdn_uom_urn'] = 'SDN:P06::UTBB'
out.ELTMEP01.attrs['sdn_uom_name'] = 'Seconds'
out.ELTMEP01.attrs['standard_name'] = 'time'

# TNIHCE01: echo intensity beam 1
out.TNIHCE01.attrs['units'] = 'counts'
out.TNIHCE01.attrs['_FillValue'] = '1e35'
out.TNIHCE01.attrs['long_name'] = 'ADCP_echo_intensity_beam_1'
out.TNIHCE01.attrs['sensor_type'] = 'adcp'
out.TNIHCE01.attrs['sensor_depth'] = '' ############################
out.TNIHCE01.attrs['serial_number'] = meta_dict['serialNumber']
out.TNIHCE01.attrs['generic_name'] = 'AGC'
out.TNIHCE01.attrs['legency_GF3_code'] = 'SDN:GF3::BEAM_01'
out.TNIHCE01.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current profiler (ADCP) beam 1'
out.TNIHCE01.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
out.TNIHCE01.attrs['sdn_uom_name'] = 'Counts'
out.TNIHCE01.attrs['data_min'] = np.nanmin(data.amp1)
out.TNIHCE01.attrs['data_max'] = np.nanmax(data.amp1)

# TNIHCE02: echo intensity beam 2
out.TNIHCE02.attrs['units'] = 'counts'
out.TNIHCE02.attrs['_FillValue'] = '1e35'
out.TNIHCE02.attrs['long_name'] = 'ADCP_echo_intensity_beam_2'
out.TNIHCE02.attrs['sensor_type'] = 'adcp'
out.TNIHCE02.attrs['sensor_depth'] = '' ############################
out.TNIHCE02.attrs['serial_number'] = meta_dict['serialNumber']
out.TNIHCE02.attrs['generic_name'] = 'AGC'
out.TNIHCE02.attrs['legency_GF3_code'] = 'SDN:GF3::BEAM_02'
out.TNIHCE02.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current profiler (ADCP) beam 2'
out.TNIHCE02.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
out.TNIHCE02.attrs['sdn_uom_name'] = 'Counts'
out.TNIHCE02.attrs['data_min'] = np.nanmin(data.amp2)
out.TNIHCE02.attrs['data_max'] = np.nanmax(data.amp2)

# TNIHCE03: echo intensity beam 3
out.TNIHCE03.attrs['units'] = 'counts'
out.TNIHCE03.attrs['_FillValue'] = '1e35'
out.TNIHCE03.attrs['long_name'] = 'ADCP_echo_intensity_beam_3'
out.TNIHCE03.attrs['sensor_type'] = 'adcp'
out.TNIHCE03.attrs['sensor_depth'] = '' ############################
out.TNIHCE03.attrs['serial_number'] = meta_dict['serialNumber']
out.TNIHCE03.attrs['generic_name'] = 'AGC'
out.TNIHCE03.attrs['legency_GF3_code'] = 'SDN:GF3::BEAM_03'
out.TNIHCE03.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current profiler (ADCP) beam 3'
out.TNIHCE03.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
out.TNIHCE03.attrs['sdn_uom_name'] = 'Counts'
out.TNIHCE03.attrs['data_min'] = np.nanmin(data.amp3)
out.TNIHCE03.attrs['data_max'] = np.nanmax(data.amp3)

# TNIHCE04: echo intensity beam 4
out.TNIHCE04.attrs['units'] = 'counts'
out.TNIHCE04.attrs['_FillValue'] = '1e35'
out.TNIHCE04.attrs['long_name'] = 'ADCP_echo_intensity_beam_4'
out.TNIHCE04.attrs['sensor_type'] = 'adcp'
out.TNIHCE04.attrs['sensor_depth'] = '' ############################
out.TNIHCE04.attrs['serial_number'] = meta_dict['serialNumber']
out.TNIHCE04.attrs['generic_name'] = 'AGC'
out.TNIHCE04.attrs['legency_GF3_code'] = 'SDN:GF3::BEAM_04'
out.TNIHCE04.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored acoustic doppler current profiler (ADCP) beam 4'
out.TNIHCE04.attrs['sdn_uom_urn'] = 'SDN:P06::UCNT'
out.TNIHCE04.attrs['sdn_uom_name'] = 'Counts'
out.TNIHCE04.attrs['data_min'] = np.nanmin(data.amp4)
out.TNIHCE04.attrs['data_max'] = np.nanmax(data.amp4)

# PCGDAP00 - 4: percent good beam 1-4
var = out.PCGDAP00
var.attrs['units'] = 'percent'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'percent_good_beam_1'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['generic_name'] = 'PGd'
var.attrs['legency_GF3_code'] = 'SDN:GF3::PGDP_01'
var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler current profiler (ADCP) beam 1'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
var.attrs['sdn_uom_name'] = 'Percent'
var.attrs['data_min'] = np.nanmin(data.Pg1)
var.attrs['data_max'] = np.nanmax(data.Pg1)

# PCGDAP02: percent good beam 2
var = out.PCGDAP02
var.attrs['units'] = 'percent'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'percent_good_beam_2'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['generic_name'] = 'PGd'
var.attrs['legency_GF3_code'] = 'SDN:GF3::PGDP_02'
var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler current profiler (ADCP) beam 2'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
var.attrs['sdn_uom_name'] = 'Percent'
var.attrs['data_min'] = np.nanmin(data.Pg2)
var.attrs['data_max'] = np.nanmax(data.Pg2)

# PCGDAP03: percent good beam 3
var = out.PCGDAP03
var.attrs['units'] = 'percent'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'percent_good_beam_3'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['generic_name'] = 'PGd'
var.attrs['legency_GF3_code'] = 'SDN:GF3::PGDP_03'
var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler current profiler (ADCP) beam 3'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
var.attrs['sdn_uom_name'] = 'Percent'
var.attrs['data_min'] = np.nanmin(data.Pg3)
var.attrs['data_max'] = np.nanmax(data.Pg3)

# PCGDAP03: percent good beam 4
var = out.PCGDAP04
var.attrs['units'] = 'percent'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'percent_good_beam_4'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['generic_name'] = 'PGd'
var.attrs['legency_GF3_code'] = 'SDN:GF3::PGDP_04'
var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler current profiler (ADCP) beam 4'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
var.attrs['sdn_uom_name'] = 'Percent'
var.attrs['data_min'] = np.nanmin(data.Pg4)
var.attrs['data_max'] = np.nanmax(data.Pg4)

# PTCHGP01: pitch
var = out.PTCHGP01
var.attrs['units'] = 'degrees'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'pitch'
var.attrs['sensor_type'] = 'adcp'
var.attrs['legency_GF3_code'] = 'SDN:GF3::PTCH'
var.attrs['sdn_parameter_name'] = 'Orientation (pitch) of measurement platform by inclinometer'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UAAA'
var.attrs['sdn_uom_name'] = 'Degrees'
var.attrs['standard_name'] = 'platform_pitch_angle'
var.attrs['data_min'] = np.nanmin(data.pitch)
var.attrs['data_max'] = np.nanmax(data.pitch)

# ROLLGP01: roll
var = out.ROLLGP01
var.attrs['units'] = 'degrees'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'roll'
var.attrs['sensor_type'] = 'adcp'
var.attrs['legency_GF3_code'] = 'SDN:GF3::ROLL'
var.attrs['sdn_parameter_name'] = 'Orientation (roll angle) of measurement platform by inclinometer (second sensor)'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UAAA'
var.attrs['sdn_uom_name'] = 'Degrees'
var.attrs['standard_name'] = 'platform_roll_angle'
var.attrs['data_min'] = np.nanmin(data.roll)
var.attrs['data_max'] = np.nanmax(data.roll)

# DISTTRAN: height of sea surface
var = out.DISTTRAN
var.attrs['units'] = 'm'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'height of sea surface'
var.attrs['generic_name'] = 'height'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['legency_GF3_code'] = 'SDN:GF3::HGHT'
var.attrs['sdn_uom_urn'] = 'SDN:P06::ULAA'
var.attrs['sdn_uom_name'] = 'Metres'
var.attrs['data_min'] = np.nanmin(data.dep)
var.attrs['data_max'] = np.nanmax(data.dep)

# TEMPPR01: transducer temp
var = out.TEMPPR01
var.attrs['units'] = 'degrees celsius'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'ADCP Transducer Temp.'
var.attrs['generic_name'] = 'temp'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['legency_GF3_code'] = 'SDN:GF3::te90'
var.attrs['sdn_parameter_name'] = 'Temperature of the water body'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UPAA'
var.attrs['sdn_uom_name'] = 'Celsius degree'
var.attrs['data_min'] = np.nanmin(data.temperature)
var.attrs['data_max'] = np.nanmax(data.temperature)

# PPSAADCP: instrument depth (formerly DEPFP01)
var = data.PPSAADCP
var.attrs['units'] = 'm'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'instrument depth'
var.attrs['xducer_offset_from_bottom'] = ''
var.attrs['generic_name'] = 'depth'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['legency_GF3_code'] = 'SDN:GF3::DEPH'
var.attrs['sdn_parameter_name'] = 'Depth below surface of the water body'
var.attrs['sdn_uom_urn'] = 'SDN:P06::ULAA'
var.attrs['sdn_uom_name'] = 'Metres'
var.attrs['standard_name'] = 'depth'
var.attrs['data_min'] = np.nanmin(data.XducerDepth)
var.attrs['data_max'] = np.nanmax(data.XducerDepth)

# ALONZZ01, longitude
for var in [data.ALONZZ01, data.longitude]:
    var.attrs['units'] = 'degrees_east'
    var.attrs['long_name'] = 'longitude'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::lon'
    var.attrs['sdn_parameter_name'] = 'Longitude east'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::DEGE'
    var.attrs['sdn_uom_name'] = 'Degrees east'
    var.attrs['standard_name'] = 'longitude'

# ALATZZ01, latitude
for var in [data.ALATZZ01, data.latitude]:
    var.attrs['units'] = 'degrees_north'
    var.attrs['long_name'] = 'latitude'
    var.attrs['legency_GF3_code'] = 'SDN:GF3::lat'
    var.attrs['sdn_parameter_name'] = 'Latitude north'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::DEGN'
    var.attrs['sdn_uom_name'] = 'Degrees north'
    var.attrs['standard_name'] = 'latitude'

# HEADCM01: heading
var = data.HEADCM01
var.attrs['units'] = 'degrees'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'heading'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['legency_GF3_code'] = 'SDN:GF3::HEAD'
var.attrs['sdn_parameter_name'] = 'Orientation (horizontal relative to true north) of measurement device {heading}'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UAAA'
var.attrs['sdn_uom_name'] = 'Degrees'
var.attrs['data_min'] = np.nanmin(data.heading)
var.attrs['data_max'] = np.nanmax(data.heading)

# PRESPR01: pressure
var = data.PRESPR01
var.attrs['units'] = 'decibars'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'pressure'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['legency_GF3_code'] = 'SDN:GF3::PRES'
var.attrs['sdn_parameter_name'] = 'Pressure (spatial co-ordinate) exerted by the water body by profiling pressure sensor and corrected to read zero at sea level'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UPDB'
var.attrs['sdn_uom_name'] = 'Decibars'
var.attrs['standard_name'] = 'sea_water_pressure
var.attrs['data_min'] = np.nanmin(pres)
var.attrs['data_max'] = np.nanmax(pres)

# SVELCV01: sound velocity
var = data.SVELCV01
var.attrs['units'] = 'm/sec'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'speed of sound'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['legency_GF3_code'] = 'SDN:GF3::SVEL'
var.attrs['sdn_parameter_name'] = 'Sound velocity in the water body by computation from temperature and salinity by unspecified algorithm'
var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
var.attrs['sdn_uom_name'] = 'Metres per second'
var.attrs['standard_name'] = 'speed_of_sound_in_sea_water'
var.attrs['data_min'] = np.nanmin(svel)
var.attrs['data_max'] = np.nanmax(svel)

# DTUT8601: time values as ISO8601 string, YY-MM-DD hh:mm:ss
var = data.SVELCV01
var.attrs['note'] = 'time values as ISO8601 string, YY-MM-DD hh:mm:ss'
var.attrs['time_zone'] = 'UTC'
var.attrs['legency_GF3_code'] = 'SDN:GF3::time_string'
var.attrs['sdn_parameter_name'] = 'String corresponding to format \'YYYY-MM-DDThh:mm:ss.sssZ\' or other valid ISO8601 string'
var.attrs['sdn_uom_urn'] = 'SDN:P06::TISO'
var.attrs['sdn_uom_name'] = 'ISO8601'

# CMAGZZ01-4: correlation magnitude
var = data.CMAGZZ01
var.attrs['units'] = 'counts'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_1'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['generic_name'] = 'CM'
var.attrs['legency_GF3_code'] = 'SDN:GF3::CMAG_01'
var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by moored acoustic doppler current profiler (ADCP) beam 1'
var.attrs['data_min'] = np.nanmin(data.cor1)
var.attrs['data_max'] = np.nanmax(data.cor1)

var = data.CMAGZZ02
var.attrs['units'] = 'counts'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_2'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['generic_name'] = 'CM'
var.attrs['legency_GF3_code'] = 'SDN:GF3::CMAG_02'
var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by moored acoustic doppler current profiler (ADCP) beam 2'
var.attrs['data_min'] = np.nanmin(data.cor2)
var.attrs['data_max'] = np.nanmax(data.cor2)

var = data.CMAGZZ03
var.attrs['units'] = 'counts'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_3'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['generic_name'] = 'CM'
var.attrs['legency_GF3_code'] = 'SDN:GF3::CMAG_03'
var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by moored acoustic doppler current profiler (ADCP) beam 3'
var.attrs['data_min'] = np.nanmin(data.cor3)
var.attrs['data_max'] = np.nanmax(data.cor3)

var = data.CMAGZZ04
var.attrs['units'] = 'counts'
var.attrs['_FillValue'] = '1e35'
var.attrs['long_name'] = 'ADCP_correlation_magnitude_beam_4'
var.attrs['sensor_type'] = 'adcp'
var.attrs['sensor_depth'] = '' ############################
var.attrs['serial_number'] = meta_dict['serialNumber']
var.attrs['generic_name'] = 'CM'
var.attrs['legency_GF3_code'] = 'SDN:GF3::CMAG_04'
var.attrs['sdn_parameter_name'] = 'Correlation magnitude of acoustic signal returns from the water body by moored acoustic doppler current profiler (ADCP) beam 4'
var.attrs['data_min'] = np.nanmin(data.cor4)
var.attrs['data_max'] = np.nanmax(data.cor4)
#done variables


# Global attributes
# system configuration keys added to dataset as Global attributes
for key in data.sysconfig.keys():
    print(int(data.sysconfig[key]))
    print((data.sysconfig[key] is not bool))
    print(type(data.sysconfig[key]) is not bool)
    if type(data.sysconfig[key]) is not bool:
        out.attrs['sysconfig' + key] = data.sysconfig[key]
    else:
        out.attrs['sysconfig' + key] = int(data.sysconfig[key])

# Create more global attributes
# Not from metadata file:
processing_history = "Metadata read in from log sheet and combined with raw data to export as netCDF file."
out.attrs['processing_history'] = processing_history
out.attrs['time_coverage_duration'] = '' #calculated from start and end times
out.attrs['cdm_data_type'] = "station"
out.attrs['number_of_beams'] = data.NBeams #change in python and R to numberOfBeams?
#out.attrs['nprofs'] = data.nprofs #number of ensembles
out.attrs['numberOfCells'] = data.NCells
out.attrs['pings_per_ensemble'] = data.NPings
out.attrs['bin1Distance'] = data.Bin1Dist
out.attrs['Blank'] = data.Blank #?? blanking distance?
out.attrs['cellSize'] = data.CellSize
out.attrs['pingtype'] = data.pingtype
out.attrs['Pulse'] = data.Pulse #??
out.attrs['instrumentType'] = "adcp"
out.attrs['manufacturer'] = manufacturer
out.attrs['source'] = "R code: adcpProcess, github:"
now = datetime.datetime.now(); out.attrs['date_modified'] = now.strftime("%Y-%m-%d %H:%M:%S")
out.attrs['_FillValue'] = "1e35" #is this even needed if we aren't processing?
out.attrs['featureType'] = "timeSeries"

# Metadata from the data object
out.attrs['firmware_version'] = '' #firmwareVersion
out.attrs['frequency'] = ''
out.attrs['beam_pattern'] = "convex" #beamPattern; where is in data
out.attrs['beam_angle'] = '' #beamAngle
out.attrs['sensor_source'] = "" #sensorSource
out.attrs['sensors_avail'] = "" #sensors_avail
out.attrs['three_beam_used'] = "" #threeBeamUsed; boolean
out.attrs['valid_correlation_range'] = '' #lowCorrThresh
out.attrs['minmax_percent_good'] = "100" #hardcoded in oceNc_create(); should be percentGdMinimum? that value is being overwritten
out.attrs['transmit_pulse_length_cm'] = '' #errorVelocityMaximum + m/s
out.attrs['transmit_pulse_length_cm'] = 1710 #xmitPulseLength
out.attrs['false_target_reject_values'] = 50 #falseTargetThresh
out.attrs['data_type'] = "adcp"
out.attrs['pred_accuracy'] = 1 #where does this come from? velocityResolution * 1000
out.attrs['Conventions'] = "CF-1.7"
out.attrs['creater_type'] = "person"
out.attrs['time_coverage_start'] = data.dday[0]
out.attrs['time_coverage_end'] = data.dday[-1]

#geospatial lat, lon, and vertical min/max calculations
out.attrs['geospatial_lat_min'] = meta_dict['latitude']
out.attrs['geospatial_lat_max'] = meta_dict['latitude']
out.attrs['geospatial_lat_units'] = "degrees_north"
out.attrs['geospatial_lon_min'] = meta_dict['longitude']
out.attrs['geospatial_lon_max'] = meta_dict['longitude']
out.attrs['geospatial_lon_units'] = "degrees_east"

# How to extract orientation and sensor depth from data object?
out.attrs['orientation'] = ''  #how to extract from dataset?
sensor_depth = '' #create variable in environment
#sensor_depth is a variable attribute, not a global attribute
if out.attrs['orientation'] == 'up':
    out.attrs['geospatial_vertical_min'] = sensor_depth - np.nanmax(data.dep) #np.nanmax ignores NaNs, like na.rm=T in R
    out.attrs['geospatial_vertical_max'] = sensor_depth - np.nanmin(data.dep)
elif out.attrs['orientation'] == 'down':
    out.attrs['geospatial_vertical_min'] = sensor_depth + np.nanmin(data.dep)
    out.attrs['geospatial_vertical_max'] = sensor_depth + np.nanmax(data.dep)

#Add select meta_dict items as global attributes
for k, v in meta_dict.items():
    if k == 'cut_lead_ensembles' or k == 'cut_trail_ensembles' or k == 'processing_level':
        pass
    elif k == 'serialNumber':
        out.attrs[k] = 'serial_number'
    else:
        out.attrs[k] = v

for key in data.trans.keys():
    print(type(data.trans[key]))
    if type(data.trans[key]) is not bool:
        out.attrs['trans' + key] = data.trans[key]
    else:
        out.attrs['trans' + key] = int(data.trans[key])

# rotate into earth **IF not in enu already.  This just makes the netcdf bigger so...
# Hana added if statement
if data.trans.coordsystem != 'earth':
    trans = transform.Transform(angle=20, geometry='convex')
    xyze = trans.beam_to_xyz(out.vel)
    print(np.shape(xyze))
    enu = transform.rdi_xyz_enu(xyze, out.heading, out.pitch, out['roll'], orientation='up')
    print(np.shape(enu))
    # Apply change in coordinates to velocities
    out['U'] = xr.DataArray(enu[:, :, 0], dims=['time', 'depth'])
    out['V'] = xr.DataArray(enu[:, :, 1], dims=['time', 'depth'])
    out['W'] = xr.DataArray(enu[:, :, 2], dims=['time', 'depth'])
    out['E'] = xr.DataArray(enu[:, :, 3], dims=['time', 'depth'])
    out.attrs['processing_history'] = "The coordinate system was rotated into enu coordinates. " + processing_history

out.attrs['coord_system'] = 'enu'

# Is this a test of the new nc file made?
# out.attrs['rVL'] = data.rVL
print(outname) #oname
print(out.sysconfigkHz)
out.to_netcdf(outname, mode='w', format='NETCDF4')
out.close()
# out.attrs['VL'] = data.VL
# Use exclamation mark to access the underlying shell
#!ls - halt


#def main():