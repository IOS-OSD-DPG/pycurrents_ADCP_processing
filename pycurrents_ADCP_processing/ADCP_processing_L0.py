"""
author: Hana Hourston
purpose: For outputting raw ADCP data in netCDF file format, without any kind of processing
            - Original coordinate system is maintained
            - No corrections for magnetic declination are made
"""

import os
import csv
import numpy as np
import xarray as xr
import datetime
import warnings
from pycurrents.adcp.rdiraw import rawfile
from pycurrents.adcp.rdiraw import SysCfg
from pycurrents_ADCP_processing.ADCP_processing_L1 import mean_orientation, \
    convert_time_var, check_depths
import pycurrents_ADCP_processing.add_var2nc as add_var2nc


def add_attrs_2vars_L0(out_obj, metadata_dict, instrument_depth, fillValue,
                       pres_flag, pg_flag, vb_flag, vb_pg_flag):
    # out_obj: dataset object produced using the xarray package that will be exported as a netCDF file
    # metadata_dict: dictionary object of metadata items
    # instrument_depth: sensor depth recorded by instrument

    uvw_vel_min = -1000
    uvw_vel_max = 1000

    # Time
    var = out_obj.time
    var.encoding['units'] = "seconds since 1970-01-01T00:00:00Z"
    var.encoding['_FillValue'] = None  # omits fill value from time dim; otherwise would be included with value NaN
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

    # VEL_MAGNETIC_EAST: Velocity component towards magnetic east (not corrected for
    # magnetic declination)
    # all velocities have many of the same attribute values, but not all, so each
    # velocity is done separately
    var = out_obj.VEL_MAGNETIC_EAST
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = fillValue
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['data_max'] = np.round(np.nanmax(var.data), decimals=2)
    var.attrs['data_min'] = np.round(np.nanmin(var.data), decimals=2)
    var.attrs['valid_max'] = uvw_vel_max
    var.attrs['valid_min'] = uvw_vel_min

    # VEL_MAGNETIC_NORTH: Velocity component towards magnetic north (uncorrected for
    # magnetic declination)
    var = out_obj.VEL_MAGNETIC_NORTH
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'northward_sea_water_velocity'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['data_max'] = np.round(np.nanmax(var.data), decimals=2)
    var.attrs['data_min'] = np.round(np.nanmin(var.data), decimals=2)
    var.attrs['valid_max'] = uvw_vel_max
    var.attrs['valid_min'] = uvw_vel_min

    # LRZAAP01: vertical velocity (vel3)
    var = out_obj.LRZAAP01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'upward_sea_water_velocity'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'w'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::VCSP'
    var.attrs['sdn_parameter_name'] = 'Upward current velocity (Eulerian measurement) ' \
                                      'in the water body by moored ' \
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
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'e'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::ERRV'
    var.attrs['sdn_parameter_name'] = 'Current velocity error in the water body by ' \
                                      'moored acoustic doppler current ' \
                                      'profiler (ADCP)'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
    var.attrs['sdn_uom_name'] = 'Metres per second'
    var.attrs['standard_name'] = 'indicative_error_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'
    var.attrs['data_max'] = np.nanmax(var.data)
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['valid_max'] = 2 * uvw_vel_max  # To agree with the R package "ADCP"
    var.attrs['valid_min'] = 2 * uvw_vel_min  # To agree with the R package "ADCP"

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
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::BEAM_01'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored ' \
                                      'acoustic doppler current ' \
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
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::BEAM_02'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored ' \
                                      'acoustic doppler current ' \
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
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::BEAM_03'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored ' \
                                      'acoustic doppler current ' \
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
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['generic_name'] = 'AGC'
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::BEAM_04'
    var.attrs['sdn_parameter_name'] = 'Echo intensity from the water body by moored ' \
                                      'acoustic doppler current ' \
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
        var.attrs['serial_number'] = metadata_dict['serialNumber']
        var.attrs['generic_name'] = 'PGd'
        var.attrs['legacy_GF3_code'] = 'SDN:GF3::PGDP_01'
        var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by ' \
                                          'moored acoustic doppler ' \
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
        var.attrs['serial_number'] = metadata_dict['serialNumber']
        var.attrs['generic_name'] = 'PGd'
        var.attrs['legacy_GF3_code'] = 'SDN:GF3::PGDP_02'
        var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by ' \
                                          'moored acoustic doppler ' \
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

    # TEMPPR01: transducer temp
    var = out_obj.TEMPPR01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'degree_C'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'ADCP Transducer Temp.'
    var.attrs['generic_name'] = 'temp'
    var.attrs['sensor_type'] = 'adcp'
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::te90'
    var.attrs['sdn_parameter_name'] = 'Temperature of the water body'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UPAA'
    var.attrs['sdn_uom_name'] = 'Celsius degree'
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
    var.attrs['serial_number'] = metadata_dict['serialNumber']
    var.attrs['legacy_GF3_code'] = 'SDN:GF3::HEAD'
    var.attrs['sdn_parameter_name'] = 'Orientation (horizontal relative to true north) of measurement device {heading}'
    var.attrs['sdn_uom_urn'] = 'SDN:P06::UAAA'
    var.attrs['sdn_uom_name'] = 'Degrees'
    var.attrs['standard_name'] = 'platform_orientation'
    var.attrs['data_min'] = np.nanmin(var.data)
    var.attrs['data_max'] = np.nanmax(var.data)

    # PRESPR01: pressure
    if pres_flag == 0:
        var = out_obj.PRESPR01
        var.encoding['dtype'] = 'float32'
        var.attrs['units'] = 'dbar'
        var.attrs['_FillValue'] = fillValue
        var.attrs['long_name'] = 'pressure'
        var.attrs['sensor_type'] = 'adcp'
        var.attrs['serial_number'] = metadata_dict['serialNumber']
        var.attrs['legacy_GF3_code'] = 'SDN:GF3::PRES'
        var.attrs['sdn_parameter_name'] = 'Pressure (spatial co-ordinate) exerted by the water body by profiling ' \
                                          'pressure sensor and corrected to read zero at sea level'
        var.attrs['sdn_uom_urn'] = 'SDN:P06::UPDB'
        var.attrs['sdn_uom_name'] = 'Decibars'
        var.attrs['standard_name'] = 'sea_water_pressure'
        var.attrs['data_min'] = np.nanmin(var.data)
        var.attrs['data_max'] = np.nanmax(var.data)

    # SVELCV01: sound velocity
    var = out_obj.SVELCV01
    var.encoding['dtype'] = 'float32'
    var.attrs['units'] = 'm s-1'
    var.attrs['_FillValue'] = fillValue
    var.attrs['long_name'] = 'speed of sound'
    var.attrs['sensor_type'] = 'adcp'
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
        var.attrs['ancillary_variables'] = 'VB_VELCTY_QC'
        var.attrs['sensor_type'] = 'adcp'
        var.attrs['serial_number'] = metadata_dict['serialNumber']
        var.attrs['sdn_uom_urn'] = 'SDN:P06::UVAA'
        var.attrs['sdn_uom_name'] = 'Metres per second'
        var.attrs['standard_name'] = 'upward_sea_water_velocity'
        var.attrs['data_max'] = np.round(np.nanmax(var.data), decimals=2)
        var.attrs['data_min'] = np.round(np.nanmin(var.data), decimals=2)
        var.attrs['valid_max'] = uvw_vel_max
        var.attrs['valid_min'] = uvw_vel_min

        var = out_obj.TNIHCE05
        var.encoding['dtype'] = 'float32'
        var.attrs['units'] = 'counts'
        var.attrs['_FillValue'] = fillValue
        var.attrs['long_name'] = 'ADCP_echo_intensity_beam_5'
        var.attrs['sensor_type'] = 'adcp'
        var.attrs['instrument_depth'] = instrument_depth
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
        var.attrs['instrument_depth'] = instrument_depth
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
            var.attrs['instrument_depth'] = instrument_depth
            var.attrs['serial_number'] = metadata_dict['serialNumber']
            var.attrs['generic_name'] = 'PGd'
            var.attrs['sdn_parameter_name'] = 'Acceptable proportion of signal returns by moored acoustic doppler ' \
                                              'current profiler (ADCP) vertical beam'
            var.attrs['standard_name'] = 'proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water'
            var.attrs['sdn_uom_urn'] = 'SDN:P06::UPCT'
            var.attrs['sdn_uom_name'] = 'Percent'
            var.attrs['data_min'] = np.nanmin(var.data)
            var.attrs['data_max'] = np.nanmax(var.data)
    return


def create_meta_dict_L0(f_meta):
    """
        Read in a csv metadata file and output in dictionary format
        Inputs:
            - adcp_meta: csv-format file containing metadata for raw ADCP file
        Outputs:
            - meta_dict: a dictionary containing the metadata from the csv file and additional metadata on conventions
        """
    meta_dict = {}
    with open(f_meta) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        next(csv_reader, None)  # Skip header row
        for row in csv_reader:
            # extract all metadata from csv file into dictionary
            # some items not passed to netCDF file but are extracted anyway
            if row[0] == '' and row[1] == '':
                print('Metadata file contains a blank row; skipping this row !')
            elif row[0] != '' and row[1] == '':
                print('Metadata item in csv file has blank value; skipping this row '
                              'in metadata file !')
            else:
                meta_dict[row[0]] = row[1]

    # Add conventions metadata to meta_dict
    meta_dict['deployment_type'] = 'Sub Surface'
    meta_dict['keywords'] = 'Oceans > Ocean Circulation > Ocean Currents'
    meta_dict['keywords_vocabulary'] = 'GCMD Science Keywords'
    meta_dict['naming_authority'] = 'BODC, MEDS, CF v72'
    meta_dict['variable_code_reference'] = 'BODC P01'
    meta_dict['Conventions'] = "CF-1.8"

    return meta_dict


def nc_create_L0(f_adcp, f_meta, dest_dir, start_year=None, time_file=None):
    """About:
        Perform level 0 processing on a raw ADCP file and export it as a netCDF file
        :param f_adcp: full file name of raw ADCP file
        :param f_meta: full file name of csv metadata file associated with inFile
        :param dest_dir: string type; name of folder in which files will be output
        :param start_year: float type; the year that measurements start for ADCP file; only
                     required for Narrowband ADCP files
        :param time_file: full file name of csv file containing user-generated time data;
                    required if inFile has garbled out-of-range time data
    """
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Define the name for the netCDF file
    out_name = os.path.basename(f_adcp)[:-4] + '.adcp.L0.nc'
    print(out_name)
    if not dest_dir.endswith('/') or not dest_dir.endswith('\\'):
        out_absolute_name = os.path.abspath(dest_dir + '/' + out_name)
    else:
        out_absolute_name = os.path.abspath(dest_dir + out_name)

    # Read information from metadata file into a dictionary, called meta_dict
    meta_dict = create_meta_dict_L0(f_meta)

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

    print('Read in csv metadata file')

    # ------------------------Read in data and start processing--------------------

    # Read in raw ADCP file and model type
    if meta_dict['model'] == 'nb':
        data = rawfile(f_adcp, meta_dict['model'], trim=True, yearbase=start_year)
    else:
        data = rawfile(f_adcp, meta_dict['model'], trim=True)
    print('Read in raw data')

    # Extract multidimensional variables from data object:
    # fixed leader, velocity, amplitude intensity, correlation magnitude, and percent good
    fixed_leader = data.read(varlist=['FixedLeader'])
    vel = data.read(varlist=['Velocity'])
    amp = data.read(varlist=['Intensity'])
    cor = data.read(varlist=['Correlation'])
    pg = data.read(varlist=['PercentGood'])

    # Create flags if pg data or vb data or vb_pg data are missing
    flag_pg = 0
    flag_vb = 0
    flag_vb_pg = 0
    try:
        # Create throwaway test variable to test for variable availability
        test_var = pg.pg1.data[:5]
    except AttributeError:
        flag_pg += 1

    # If model == Sentinel V, read in vertical beam data
    if meta_dict['model'] == 'sv':
        # vb_leader = data.read(varlist=['VBLeader'])
        vb_vel = data.read(varlist=['VBVelocity'])
        vb_amp = data.read(varlist=['VBIntensity'])
        vb_cor = data.read(varlist=['VBCorrelation'])
        vb_pg = data.read(varlist=['VBPercentGood'])

        # Test for missing Sentinel V vertical beam data; if true treat file as regular 4-beam file
        try:
            # Vertical beam velocity data also available from vb_vel.raw.VBVelocity
            # but it's multiplied by 1e3 (to make int type)
            test_var = vb_vel.vbvel.data[:5]
        except AttributeError:
            flag_vb += 1
        # Test for missing vertical beam percent good data
        try:
            test_var = vb_pg.raw.VBPercentGood[:5]
        except AttributeError:
            flag_vb_pg += 1

    # print(flag_pg)
    # print(flag_vb)
    # print(flag_vb_pg)

    # --------------------------Metadata value corrections-------------------------

    # Convert numeric values from string to numeric type
    meta_dict['country_institute_code'] = int(meta_dict['country_institute_code'])

    for key in ['instrument_depth', 'latitude', 'longitude', 'water_depth', 'magnetic_variation']:
        meta_dict[key] = float(meta_dict[key])

    # Serial number corrections

    # Add leading zero to serial numbers that have 3 digits
    if len(str(meta_dict['serialNumber'])) == 3:
        meta_dict['serialNumber'] = '0' + str(meta_dict['serialNumber'])

    if meta_dict['model'].upper() not in meta_dict['serialNumber']:
        # Overwrite serial number to include the model: upper returns uppercase
        meta_dict['serialNumber'] = meta_dict['model'].upper() + meta_dict['serialNumber']

        # Add instrument model variable value
        meta_dict['instrumentModel'] = '{} ADCP {}kHz ({})'.format(model_long, data.sysconfig['kHz'],
                                                                   meta_dict['serialNumber'])

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

    time_s, time_DTUT8601 = convert_time_var(time_var=vel.dday, number_of_profiles=data.nprofs,
                                             metadata_dict=meta_dict,
                                             origin_year=data.yearbase, time_csv=time_file)

    # Distance dimension
    distance = np.round(vel.dep.data, decimals=2)

    # Continue setting up variables

    # Convert SoundSpeed from int16 to float32
    sound_speed = np.float32(vel.VL['SoundSpeed'])

    # Convert decapascal to decibar
    # Do not create pressure variable if no data exists
    flag_no_pres = 0
    try:
        decapascal2decibar = 1/1000
        pressure = np.array(vel.VL['Pressure'] * decapascal2decibar, dtype='float32')
    except ValueError:
        warnings.warn('No pressure data available (no field of name Pressure')
        flag_no_pres += 1

    # ------------------------------Depth-------------------------------

    # Check instrument_depth from metadata csv file: compare with pressure values
    if flag_no_pres == 0:
        check_depths(pressure, distance, meta_dict['instrument_depth'], meta_dict['water_depth'])

    # Adjust velocity data

    # Set velocity values of -32768.0 to nans, since -32768.0 is the automatic fill_value for pycurrents
    vel.vel.data[vel.vel.data == -32768.0] = np.nan

    if meta_dict['model'] == 'sv' and flag_vb == 0:
        vb_vel.vbvel.data[vb_vel.vbvel.data == -32768.0] = np.nan

    # Make into netCDF file

    # Create xarray Dataset object containing all dimensions and variables
    # Sentinel V instruments don't have percent good ('pg') variables
    out = xr.Dataset(coords={'time': time_s, 'distance': distance},
                     data_vars={'VEL_MAGNETIC_EAST': (['distance', 'time'], vel.vel1.data.transpose()),
                                'VEL_MAGNETIC_NORTH': (['distance', 'time'], vel.vel2.data.transpose()),
                                'LRZAAP01': (['distance', 'time'], vel.vel3.data.transpose()),
                                'LERRAP01': (['distance', 'time'], vel.vel4.data.transpose()),
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
                                'XDUCER_DEPTH': (['time'], vel.XducerDepth),
                                'ALATZZ01': ([], meta_dict['latitude']),
                                'ALONZZ01': ([], meta_dict['longitude']),
                                'latitude': ([], meta_dict['latitude']),
                                'longitude': ([], meta_dict['longitude']),
                                'SVELCV01': (['time'], sound_speed),
                                'DTUT8601': (['time'], time_DTUT8601),
                                'filename': ([], out_name[:-3]),
                                'instrument_serial_number': ([], meta_dict['serialNumber']),
                                'instrument_model': ([], meta_dict['instrumentModel'])})

    if flag_no_pres == 0:
        print('Assigning pressure variable')
        out = out.assign(PRESPR01=(('time'), pressure))
    
    if flag_pg == 0:
        print('Assigning percent good variables')
        out = out.assign(PCGDAP00=(('distance', 'time'), pg.pg1.transpose()))
        out = out.assign(PCGDAP02=(('distance', 'time'), pg.pg2.transpose()))
        out = out.assign(PCGDAP03=(('distance', 'time'), pg.pg3.transpose()))
        out = out.assign(PCGDAP04=(('distance', 'time'), pg.pg4.transpose()))

    if meta_dict['model'] == 'sv' and flag_vb == 0:
        print('Assigning Sentinel V vertical beam variables')
        out = out.assign(LRZUVP01=(('distance', 'time'), vb_vel.vbvel.data.transpose()))
        out = out.assign(TNIHCE05=(('distance', 'time'), vb_amp.raw.VBIntensity.transpose()))
        out = out.assign(CMAGZZ05=(('distance', 'time'), vb_cor.raw.VBCorrelation.transpose()))
        if flag_vb_pg == 0:
            print('Assigning Sentinel V vertical beam percent good variable')
            # OR vb_pg.VBPercentGood.transpose() ?
            out = out.assign(PCGDAP05=(('distance', 'time'), vb_pg.raw.VBPercentGood.transpose()))

    # Add attributes to each variable
    fill_value = 1e+15
    add_attrs_2vars_L0(out_obj=out, metadata_dict=meta_dict, instrument_depth=meta_dict['instrument_depth'],
                       fillValue=fill_value, pres_flag=flag_no_pres, pg_flag=flag_pg, vb_flag=flag_vb,
                       vb_pg_flag=flag_vb_pg)

    # Global attributes

    # Add select meta_dict items as global attributes
    pass_dict_keys = ['cut_lead_ensembles', 'cut_trail_ensembles', 'model']
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
    out.attrs['min_percent_good'] = fixed_leader.FL['PGMin']
    out.attrs['blank'] = '{} m'.format(fixed_leader.FL['Blank'] / 100) #convert cm to m
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
    out.attrs['time_coverage_start'] = time_DTUT8601[1] + ' UTC'
    out.attrs['time_coverage_end'] = time_DTUT8601[-1] + ' UTC'

    # geospatial lat, lon, and vertical min/max calculations
    out.attrs['geospatial_lat_min'] = meta_dict['latitude']
    out.attrs['geospatial_lat_max'] = meta_dict['latitude']
    out.attrs['geospatial_lat_units'] = "degrees_north"
    out.attrs['geospatial_lon_min'] = meta_dict['longitude']
    out.attrs['geospatial_lon_max'] = meta_dict['longitude']
    out.attrs['geospatial_lon_units'] = "degrees_east"

    # Export the 'out' object as a netCDF file
    out.to_netcdf(out_absolute_name, mode='w', format='NETCDF4')
    out.close()

    return out_absolute_name


def example_L0_1():
    # 1) raw .000 file
    raw_file = "./sample_data/a1_20050503_20050504_0221m.000"
    # 2) csv metadata file
    raw_file_meta = "./sample_data/a1_20050503_20050504_0221m_meta_L1.csv"
    # 3) destination directory for output files
    dest_dir = 'dest_dir'

    # Create netCDF file
    nc_name = nc_create_L0(raw_file, raw_file_meta, dest_dir, start_year=None, time_file=None)
    # Produce new netCDF file that includes a geographic_area variable
    geo_name = add_var2nc.add_geo(nc_name, dest_dir)

    return nc_name, geo_name


def example_L0_2():
    # Specify raw ADCP file to create nc file from, along with associated csv metadata file
    # AND time file

    # raw .000 file
    raw_file = "./sample_data/scott2_20160711_20170707_0040m.pd0"
    # csv metadata file
    raw_file_meta = "./sample_data/scott2_20160711_20170707_0040m_meta_L1.csv"
    # csv time file
    scott_time = './sample_data/scott2_20160711_20170707_0040m_time.csv'

    dest_dir = 'dest_dir'

    # Create netCDF file
    nc_name = nc_create_L0(f_adcp=raw_file, f_meta=raw_file_meta, dest_dir=dest_dir, time_file=scott_time)

    # Produce new netCDF file that includes a geographic_area variable
    geo_name = add_var2nc.add_geo(nc_name, dest_dir)

    return nc_name, geo_name
