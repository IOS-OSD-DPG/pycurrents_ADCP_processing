# author: Lu Guan

import xarray as xr
# import numpy as np
# from matplotlib import pyplot as plt
# import netCDF4
import os
# import glob
import pandas as pd
import sys
from datetime import datetime
# from datetime import timedelta
from decimal import Decimal

## File Section+_updated


def convert_timedelta(duration):
    # define function to find out time increment
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    m_seconds = int(seconds * 0.001)
    return str(days) + " " + str(hours) + " " + str(minutes) + " " + str(seconds) + " " + str(m_seconds)


def unit(duration):
    # define function to find out time units
    days, seconds = duration.days, duration.seconds
    if days != 0:
        time_units = "Days"
    else:
        if (days * 24 + seconds // 3600) != 0:
            time_units = "Hours"
        else:
            if ((seconds % 3600) // 60) != 0:
                time_units = "Minutes"
            else:
                if (seconds % 60) != 0:
                    time_units = "Seconds"
    return time_units


def write_file(nc):
    # define function to write file section

    # Test if the PCGPAP00-4 variables exist
    flag_pg = 0
    try:
        x = nc.PCGDAP00.data_max
    except AttributeError:
        flag_pg = 1
    # For Sentinel V files, test if the vertical beam variables exist
    flag_vb = 0
    if nc.instrumentSubtype == 'Sentinel V':
        try:
            x = nc.LRZUVP01.data_max
        except AttributeError:
            flag_vb += 1

    start_time = pd.to_datetime(nc.coords["time"].values[1]).strftime("%Y/%m/%d %H:%M:%S.%f")[0:-4]
    end_time = pd.to_datetime(nc.coords["time"].values[-1]).strftime("%Y/%m/%d %H:%M:%S.%f")[0:-4]
    time_increment_2 = nc.coords["time"].values[2].astype('M8[ms]').astype('O')
    time_increment_1 = nc.coords["time"].values[0].astype('M8[ms]').astype('O')
    time_increment = (time_increment_2 - time_increment_1) / 2
    time_increment_string = convert_timedelta(time_increment)  # call convert_timedelta function
    time_units_string = unit(time_increment)  # call unit function
    number_of_records = str(nc.coords["time"].size)  # number of ensumbles
    data_description = nc.attrs["instrumentType"]
    # if nc.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
    #     if flag_pg == 1:
    #         number_of_channels = "30"
    #     else:
    #         number_of_channels = "35"
    # else:
    #     number_of_channels = "30"
    # nc.PTCHGP01.attrs["units"]
    nan = -99

    # TODO make a dictionary to make writing channels easier to follow
    channel_dict = {}
    channels_to_use = ['DAY', 'TIME',
                       'LCEWAP01', 'LCNSAP01', 'LRZAAP01', 'LERRAP01', 'LRZUVP01',
                       'LCEWAP01_QC', 'LCNSAP01_QC', 'LRZAAP01_QC', 'LRZUVP01_QC',
                       'TNIHCE01', 'TNIHCE02', 'TNIHCE03', 'TNIHCE04', 'TNIHCE05',
                       'CMAGZZ01', 'CMAGZZ02', 'CMAGZZ03', 'CMAGZZ04', 'CMAGZZ05',
                       'PCGDAP00', 'PCGDAP02', 'PCGDAP03', 'PCGDAP04', 'PCGDAP05',
                       'PTCHGP01', 'HEADCM01', 'ROLLGP01', 'TEMPPR01', 'DISTTRAN',
                       'PPSAADCP', 'PRESPR01', 'PRESPR01_QC', 'SVELCV01', 'PREXMCAT']

    channel_num = 1
    for channel in channels_to_use:
        # Populate the dictionary with format {channel: (name_to_use, unit, min, max)
        if channel == 'DAY':
            channel_dict[channel] = {
                'channel_num': str(channel_num),
                'name_to_use': nc.DTUT8601.time_zone + " " + "Date",
                'unit': "YYYY-MM-DD", 'data_min': "n/a", 'data_max': "n/a",
                'pad': "' '", 'width': "' '", 'format': 'YYYY-MM-DD', 'type': 'D',
                'decimal_places': "' '"}
            channel_num += 1
        elif channel == 'TIME':
            channel_dict[channel] = {
                'channel_num': str(channel_num),
                'name_to_use': nc.DTUT8601.time_zone + " " + "Time",
                'unit': "HH:MM:SS", 'data_min': "n/a", 'data_max': "n/a",
                'pad': "' '", 'width': "' '", 'format': 'HH:MM:SS', 'type': 'T',
                'decimal_places': "' '"}
            channel_num += 1
        else:
            if hasattr(nc, channel):
                if channel == 'PREXMCAT':
                    channel_dict[channel] = {
                        'channel_num': str(channel_num),
                        'name_to_use': 'Pressure_from_ctd',
                        'unit': nc[channel].attrs['units'],
                        'data_min': float(nc[channel].attrs["data_min"]),
                        'data_max': float(nc[channel].attrs["data_max"]),
                        'pad': '%.6E' % nan, 'width': '14', 'format': 'E',
                        'type': 'R4',
                        'decimal_places': '6'
                    }
                else:
                    channel_dict[channel] = {
                        'channel_num': str(channel_num),
                        'name_to_use': nc[channel].long_name.title(),
                        'unit': nc[channel].attrs['units'] if hasattr(nc[channel], 'units') else '',
                        'data_min': float(nc[channel].attrs["data_min"]),
                        'data_max': float(nc[channel].attrs["data_max"]),
                        'pad': '%.6E' % nan, 'width': '14', 'format': 'E',
                        'type': 'I' if 'QC' in channel else "R4",
                        'decimal_places': '6'}
                channel_num += 1
    number_of_channels = str(channel_num - 1)

    print("*FILE")
    print("    " + '{:20}'.format('START TIME') + ": UTC " + start_time)
    print("    " + '{:20}'.format('END TIME') + ": UTC " + end_time)
    print("    " + '{:20}'.format('TIME INCREMENT') + ": " + time_increment_string +
          "  ! (day hr min sec ms)")
    print("    " + '{:20}'.format('TIME UNITS') + ": " + time_units_string)
    print("    " + '{:20}'.format('NUMBER OF RECORDS') + ": " + number_of_records)
    print("    " + '{:20}'.format('DATA DESCRIPTION') + ": " + data_description)
    print("    " + '{:20}'.format('NUMBER OF CHANNELS') + ": " + number_of_channels)
    print()
    print('{:>20}'.format('$TABLE: CHANNELS'))
    print('    ' + '! No Name                               Units            Minimum          Maximum')
    print('    ' + '!--- ---------------------------------  ---------------  ---------------  ---------------')
    # print('{:>8}'.format('1') + " " + '{:25}'.format('Record_Number') + '{:13}'.format('n/a') + '{:16}'.format('1') + '{:12}'.format(number_of_records))
    # print('{:>8}'.format('1') + " " + '{:35}'.format(nc.ELTMEP01.standard_name.title()) + '{:20}'.format("YYYY-MM-DDThh:mm:ssZ") + '{:>22}'.format(str(nc.ELTMEP01.values[0,0])[:-10]) + '{:>22}'.format(str(nc.ELTMEP01.values[-1,-1])[:-10]))
    # print('{:>8}'.format('1') + " " + '{:35}'.format(nc.DTUT8601.time_zone) + '{:20}'.format("YYYY-MM-DD hh:mm:ss") + '{:>22}'.format("n/a") + '{:>22}'.format("n/a"))

    # USE THE DICTIONARY
    for channel in channel_dict.keys():
        if channel in ['DATE', 'TIME']:
            print('{:>8}'.format(channel_dict[channel]['channel_num']) +
                  " " + '{:35}'.format(channel_dict[channel]['name_to_use']) +
                  '{:15}'.format(channel_dict[channel]['unit']) +
                  '{:>17}'.format(channel_dict[channel]['data_min']) +
                  '{:>17}'.format(channel_dict[channel]['data_max']))
        else:
            print('{:>8}'.format(channel_dict[channel]['channel_num']) +
                  " " + '{:35}'.format(channel_dict[channel]['name_to_use']) +
                  '{:15}'.format(channel_dict[channel]['unit']) +
                  '{:>17}'.format('%.6E' % channel_dict[channel]['data_min']) +
                  '{:>17}'.format('%.6E' % channel_dict[channel]['data_max']))

    # if nc.instrumentSubtype == 'Sentinel V' and flag_vb == 0:
    #     print('{:>8}'.format('1') + " " + '{:35}'.format(nc.DTUT8601.time_zone + " " + "Date") + '{:15}'.format(
    #         "YYYY-MM-DD") + '{:>17}'.format("n/a") + '{:>17}'.format("n/a"))
    #     print('{:>8}'.format('2') + " " + '{:35}'.format(nc.DTUT8601.time_zone + " " + "Time") + '{:15}'.format(
    #         "HH:MM:SS") + '{:>17}'.format("n/a") + '{:>17}'.format("n/a"))
    #     print('{:>8}'.format('3') + " " + '{:35}'.format(nc.LCEWAP01.long_name.title()) + '{:15}'.format(
    #         nc.LCEWAP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.LCEWAP01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LCEWAP01.attrs["data_max"]))
    #     print('{:>8}'.format('4') + " " + '{:35}'.format(nc.LCNSAP01.long_name.title()) + '{:15}'.format(
    #         nc.LCNSAP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.LCNSAP01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LCNSAP01.attrs["data_max"]))
    #     print('{:>8}'.format('5') + " " + '{:35}'.format(nc.LRZAAP01.long_name.title()) + '{:15}'.format(
    #         nc.LRZAAP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.LRZAAP01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LRZAAP01.attrs["data_max"]))
    #     print('{:>8}'.format('6') + " " + '{:35}'.format(nc.LERRAP01.long_name.title()) + '{:15}'.format(
    #         nc.LERRAP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.LERRAP01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LERRAP01.attrs["data_max"]))
    #     # print('{:>8}'.format('7') + " " + '{:35}'.format(nc.LRZUVP01.long_name.title()) + '{:15}'.format(
    #     #     nc.LRZUVP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.LRZUVP01.attrs["data_min"]) + '{:>17}'.format(
    #     #     '%.6E' % nc.LRZUVP01.attrs["data_max"]))
    #     print('{:>8}'.format('8') + " " + '{:35}'.format("Upward...Velocity_By_Vertical_Beam") + '{:15}'.format(
    #         nc.LRZUVP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.LRZUVP01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LRZUVP01.attrs["data_max"]))
    #     print('{:>8}'.format('8') + " " + '{:35}'.format(nc.LCEWAP01_QC.long_name.title()) + '{:15}'.format(
    #         ' ') + '{:>17}'.format('%.6E' % nc.LCEWAP01_QC.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LCEWAP01_QC.attrs["data_max"]))
    #     print('{:>8}'.format('9') + " " + '{:35}'.format(nc.LCNSAP01_QC.long_name.title()) + '{:15}'.format(
    #         ' ') + '{:>17}'.format('%.6E' % nc.LCNSAP01_QC.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LCNSAP01_QC.attrs["data_max"]))
    #     print('{:>8}'.format('10') + " " + '{:35}'.format(nc.LRZAAP01_QC.long_name.title()) + '{:15}'.format(
    #         ' ') + '{:>17}'.format('%.6E' % nc.LRZAAP01_QC.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LRZAAP01_QC.attrs["data_max"]))
    #     print('{:>8}'.format('11') + " " + '{:35}'.format(nc.LRZUVP01_QC.long_name.title()) + '{:15}'.format(
    #         ' ') + '{:>17}'.format('%.6E' % nc.LRZUVP01_QC.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LRZUVP01_QC.attrs["data_max"]))
    #     print('{:>8}'.format('12') + " " + '{:35}'.format(nc.TNIHCE01.long_name.title()) + '{:15}'.format(
    #         nc.TNIHCE01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TNIHCE01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.TNIHCE01.attrs["data_max"]))
    #     print('{:>8}'.format('13') + " " + '{:35}'.format(nc.TNIHCE02.long_name.title()) + '{:15}'.format(
    #         nc.TNIHCE02.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TNIHCE02.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.TNIHCE02.attrs["data_max"]))
    #     print('{:>8}'.format('14') + " " + '{:35}'.format(nc.TNIHCE03.long_name.title()) + '{:15}'.format(
    #         nc.TNIHCE03.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TNIHCE03.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.TNIHCE03.attrs["data_max"]))
    #     print('{:>8}'.format('15') + " " + '{:35}'.format(nc.TNIHCE04.long_name.title()) + '{:15}'.format(
    #         nc.TNIHCE04.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TNIHCE04.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.TNIHCE04.attrs["data_max"]))
    #     print('{:>8}'.format('16') + " " + '{:35}'.format(nc.TNIHCE05.long_name.title()) + '{:15}'.format(
    #         nc.TNIHCE05.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TNIHCE05.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.TNIHCE05.attrs["data_max"]))
    #     print('{:>8}'.format('17') + " " + '{:35}'.format(nc.CMAGZZ01.long_name.title()) + '{:15}'.format(
    #         nc.CMAGZZ01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.CMAGZZ01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.CMAGZZ01.attrs["data_max"]))
    #     print('{:>8}'.format('18') + " " + '{:35}'.format(nc.CMAGZZ02.long_name.title()) + '{:15}'.format(
    #         nc.CMAGZZ02.attrs["units"]) + '{:>17}'.format('%.6E' % nc.CMAGZZ02.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.CMAGZZ02.attrs["data_max"]))
    #     print('{:>8}'.format('19') + " " + '{:35}'.format(nc.CMAGZZ03.long_name.title()) + '{:15}'.format(
    #         nc.CMAGZZ03.attrs["units"]) + '{:>17}'.format('%.6E' % nc.CMAGZZ03.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.CMAGZZ03.attrs["data_max"]))
    #     print('{:>8}'.format('20') + " " + '{:35}'.format(nc.CMAGZZ04.long_name.title()) + '{:15}'.format(
    #         nc.CMAGZZ04.attrs["units"]) + '{:>17}'.format('%.6E' % nc.CMAGZZ04.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.CMAGZZ04.attrs["data_max"]))
    #     print('{:>8}'.format('21') + " " + '{:35}'.format(nc.CMAGZZ05.long_name.title()) + '{:15}'.format(
    #         nc.CMAGZZ05.attrs["units"]) + '{:>17}'.format('%.6E' % nc.CMAGZZ05.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.CMAGZZ05.attrs["data_max"]))
    #     if flag_pg == 1:
    #         print('{:>8}'.format('22') + " " + '{:35}'.format(nc.PTCHGP01.long_name.title()) + '{:15}'.format(
    #             nc.PTCHGP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PTCHGP01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PTCHGP01.attrs["data_max"]))
    #         print('{:>8}'.format('23') + " " + '{:35}'.format(nc.HEADCM01.long_name.title()) + '{:15}'.format(
    #             nc.HEADCM01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.HEADCM01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.HEADCM01.attrs["data_max"]))
    #         print('{:>8}'.format('24') + " " + '{:35}'.format(nc.ROLLGP01.long_name.title()) + '{:15}'.format(
    #             nc.ROLLGP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.ROLLGP01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.ROLLGP01.attrs["data_max"]))
    #         print('{:>8}'.format('25') + " " + '{:35}'.format(nc.TEMPPR01.long_name.title()) + '{:15}'.format(
    #             nc.TEMPPR01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TEMPPR01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.TEMPPR01.attrs["data_max"]))
    #         print('{:>8}'.format('26') + " " + '{:35}'.format(nc.DISTTRAN.long_name.title()) + '{:15}'.format(
    #             nc.DISTTRAN.attrs["units"]) + '{:>17}'.format('%.6E' % nc.DISTTRAN.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.DISTTRAN.attrs["data_max"]))
    #         print('{:>8}'.format('27') + " " + '{:35}'.format(nc.PPSAADCP.long_name.title()) + '{:15}'.format(
    #             nc.PPSAADCP.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PPSAADCP.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PPSAADCP.attrs["data_max"]))
    #         # print('{:>8}'.format('27') + " " + '{:35}'.format(nc.DEPFP01.long_name.title()) + '{:15}'.format(nc.DEPFP01.attrs["units"]) + '{:>17}'.format('%.6E'% nc.DEPFP01.attrs["data_min"]) + '{:>17}'.format('%.6E'% nc.DEPFP01.attrs["data_max"]))
    #         print('{:>8}'.format('28') + " " + '{:35}'.format(nc.PRESPR01.long_name.title()) + '{:15}'.format(
    #             nc.PRESPR01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PRESPR01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PRESPR01.attrs["data_max"]))
    #         print('{:>8}'.format('29') + " " + '{:35}'.format(nc.PRESPR01_QC.long_name.title()) + '{:15}'.format(
    #             ' ') + '{:>17}'.format('%.6E' % nc.PRESPR01_QC.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PRESPR01_QC.attrs["data_max"]))
    #         print('{:>8}'.format('30') + " " + '{:35}'.format(nc.SVELCV01.long_name.title()) + '{:15}'.format(
    #             nc.SVELCV01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.SVELCV01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.SVELCV01.attrs["data_max"]))
    #     else:
    #         # flag_pg == 0
    #         print('{:>8}'.format('22') + " " + '{:35}'.format(nc.PCGDAP00.long_name.title()) + '{:15}'.format(
    #             nc.PCGDAP00.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PCGDAP00.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PCGDAP00.attrs["data_max"]))
    #         print('{:>8}'.format('23') + " " + '{:35}'.format(nc.PCGDAP02.long_name.title()) + '{:15}'.format(
    #             nc.PCGDAP02.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PCGDAP02.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PCGDAP02.attrs["data_max"]))
    #         print('{:>8}'.format('24') + " " + '{:35}'.format(nc.PCGDAP03.long_name.title()) + '{:15}'.format(
    #             nc.PCGDAP03.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PCGDAP03.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PCGDAP03.attrs["data_max"]))
    #         print('{:>8}'.format('25') + " " + '{:35}'.format(nc.PCGDAP04.long_name.title()) + '{:15}'.format(
    #             nc.PCGDAP04.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PCGDAP04.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PCGDAP04.attrs["data_max"]))
    #         print('{:>8}'.format('26') + " " + '{:35}'.format(nc.PCGDAP05.long_name.title()) + '{:15}'.format(
    #             nc.PCGDAP05.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PCGDAP05.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PCGDAP05.attrs["data_max"]))
    #         print('{:>8}'.format('27') + " " + '{:35}'.format(nc.PTCHGP01.long_name.title()) + '{:15}'.format(
    #             nc.PTCHGP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PTCHGP01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PTCHGP01.attrs["data_max"]))
    #         print('{:>8}'.format('28') + " " + '{:35}'.format(nc.HEADCM01.long_name.title()) + '{:15}'.format(
    #             nc.HEADCM01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.HEADCM01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.HEADCM01.attrs["data_max"]))
    #         print('{:>8}'.format('29') + " " + '{:35}'.format(nc.ROLLGP01.long_name.title()) + '{:15}'.format(
    #             nc.ROLLGP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.ROLLGP01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.ROLLGP01.attrs["data_max"]))
    #         print('{:>8}'.format('30') + " " + '{:35}'.format(nc.TEMPPR01.long_name.title()) + '{:15}'.format(
    #             nc.TEMPPR01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TEMPPR01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.TEMPPR01.attrs["data_max"]))
    #         print('{:>8}'.format('31') + " " + '{:35}'.format(nc.DISTTRAN.long_name.title()) + '{:15}'.format(
    #             nc.DISTTRAN.attrs["units"]) + '{:>17}'.format('%.6E' % nc.DISTTRAN.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.DISTTRAN.attrs["data_max"]))
    #         print('{:>8}'.format('32') + " " + '{:35}'.format(nc.PPSAADCP.long_name.title()) + '{:15}'.format(
    #             nc.PPSAADCP.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PPSAADCP.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PPSAADCP.attrs["data_max"]))
    #         # print('{:>8}'.format('27') + " " + '{:35}'.format(nc.DEPFP01.long_name.title()) + '{:15}'.format(nc.DEPFP01.attrs["units"]) + '{:>17}'.format('%.6E'% nc.DEPFP01.attrs["data_min"]) + '{:>17}'.format('%.6E'% nc.DEPFP01.attrs["data_max"]))
    #         print('{:>8}'.format('33') + " " + '{:35}'.format(nc.PRESPR01.long_name.title()) + '{:15}'.format(
    #             nc.PRESPR01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PRESPR01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PRESPR01.attrs["data_max"]))
    #         print('{:>8}'.format('34') + " " + '{:35}'.format(nc.PRESPR01_QC.long_name.title()) + '{:15}'.format(
    #             ' ') + '{:>17}'.format('%.6E' % nc.PRESPR01_QC.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.PRESPR01_QC.attrs["data_max"]))
    #         print('{:>8}'.format('35') + " " + '{:35}'.format(nc.SVELCV01.long_name.title()) + '{:15}'.format(
    #             nc.SVELCV01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.SVELCV01.attrs["data_min"]) + '{:>17}'.format(
    #             '%.6E' % nc.SVELCV01.attrs["data_max"]))
    # else:
    #     # instrumentSubtype == Workhorse, Broadband, or Narrowband
    #     print('{:>8}'.format('1') + " " + '{:35}'.format(nc.DTUT8601.time_zone + " " + "Date") + '{:15}'.format(
    #         "YYYY-MM-DD") + '{:>17}'.format("n/a") + '{:>17}'.format("n/a"))
    #     print('{:>8}'.format('2') + " " + '{:35}'.format(nc.DTUT8601.time_zone + " " + "Time") + '{:15}'.format(
    #         "HH:MM:SS") + '{:>17}'.format("n/a") + '{:>17}'.format("n/a"))
    #     print('{:>8}'.format('3') + " " + '{:35}'.format(nc.LCEWAP01.long_name.title()) + '{:15}'.format(
    #         nc.LCEWAP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.LCEWAP01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LCEWAP01.attrs["data_max"]))
    #     print('{:>8}'.format('4') + " " + '{:35}'.format(nc.LCNSAP01.long_name.title()) + '{:15}'.format(
    #         nc.LCNSAP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.LCNSAP01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LCNSAP01.attrs["data_max"]))
    #     print('{:>8}'.format('5') + " " + '{:35}'.format(nc.LRZAAP01.long_name.title()) + '{:15}'.format(
    #         nc.LRZAAP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.LRZAAP01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LRZAAP01.attrs["data_max"]))
    #     print('{:>8}'.format('6') + " " + '{:35}'.format(nc.LERRAP01.long_name.title()) + '{:15}'.format(
    #         nc.LERRAP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.LERRAP01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LERRAP01.attrs["data_max"]))
    #     print('{:>8}'.format('7') + " " + '{:35}'.format(nc.LCEWAP01_QC.long_name.title()) + '{:15}'.format(
    #         ' ') + '{:>17}'.format('%.6E' % nc.LCEWAP01_QC.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LCEWAP01_QC.attrs["data_max"]))
    #     print('{:>8}'.format('8') + " " + '{:35}'.format(nc.LCNSAP01_QC.long_name.title()) + '{:15}'.format(
    #         ' ') + '{:>17}'.format('%.6E' % nc.LCNSAP01_QC.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LCNSAP01_QC.attrs["data_max"]))
    #     print('{:>8}'.format('9') + " " + '{:35}'.format(nc.LRZAAP01_QC.long_name.title()) + '{:15}'.format(
    #         ' ') + '{:>17}'.format('%.6E' % nc.LRZAAP01_QC.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.LRZAAP01_QC.attrs["data_max"]))
    #     print('{:>8}'.format('10') + " " + '{:35}'.format(nc.TNIHCE01.long_name.title()) + '{:15}'.format(
    #         nc.TNIHCE01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TNIHCE01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.TNIHCE01.attrs["data_max"]))
    #     print('{:>8}'.format('11') + " " + '{:35}'.format(nc.TNIHCE02.long_name.title()) + '{:15}'.format(
    #         nc.TNIHCE02.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TNIHCE02.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.TNIHCE02.attrs["data_max"]))
    #     print('{:>8}'.format('12') + " " + '{:35}'.format(nc.TNIHCE03.long_name.title()) + '{:15}'.format(
    #         nc.TNIHCE03.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TNIHCE03.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.TNIHCE03.attrs["data_max"]))
    #     print('{:>8}'.format('13') + " " + '{:35}'.format(nc.TNIHCE04.long_name.title()) + '{:15}'.format(
    #         nc.TNIHCE04.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TNIHCE04.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.TNIHCE04.attrs["data_max"]))
    #     print('{:>8}'.format('14') + " " + '{:35}'.format(nc.CMAGZZ01.long_name.title()) + '{:15}'.format(
    #         nc.CMAGZZ01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.CMAGZZ01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.CMAGZZ01.attrs["data_max"]))
    #     print('{:>8}'.format('15') + " " + '{:35}'.format(nc.CMAGZZ02.long_name.title()) + '{:15}'.format(
    #         nc.CMAGZZ02.attrs["units"]) + '{:>17}'.format('%.6E' % nc.CMAGZZ02.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.CMAGZZ02.attrs["data_max"]))
    #     print('{:>8}'.format('16') + " " + '{:35}'.format(nc.CMAGZZ03.long_name.title()) + '{:15}'.format(
    #         nc.CMAGZZ03.attrs["units"]) + '{:>17}'.format('%.6E' % nc.CMAGZZ03.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.CMAGZZ03.attrs["data_max"]))
    #     print('{:>8}'.format('17') + " " + '{:35}'.format(nc.CMAGZZ04.long_name.title()) + '{:15}'.format(
    #         nc.CMAGZZ04.attrs["units"]) + '{:>17}'.format('%.6E' % nc.CMAGZZ04.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.CMAGZZ04.attrs["data_max"]))
    #     print('{:>8}'.format('18') + " " + '{:35}'.format(nc.PCGDAP00.long_name.title()) + '{:15}'.format(
    #         nc.PCGDAP00.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PCGDAP00.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.PCGDAP00.attrs["data_max"]))
    #     print('{:>8}'.format('19') + " " + '{:35}'.format(nc.PCGDAP02.long_name.title()) + '{:15}'.format(
    #         nc.PCGDAP02.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PCGDAP02.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.PCGDAP02.attrs["data_max"]))
    #     print('{:>8}'.format('20') + " " + '{:35}'.format(nc.PCGDAP03.long_name.title()) + '{:15}'.format(
    #         nc.PCGDAP03.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PCGDAP03.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.PCGDAP03.attrs["data_max"]))
    #     print('{:>8}'.format('21') + " " + '{:35}'.format(nc.PCGDAP04.long_name.title()) + '{:15}'.format(
    #         nc.PCGDAP04.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PCGDAP04.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.PCGDAP04.attrs["data_max"]))
    #     print('{:>8}'.format('22') + " " + '{:35}'.format(nc.PTCHGP01.long_name.title()) + '{:15}'.format(
    #         nc.PTCHGP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PTCHGP01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.PTCHGP01.attrs["data_max"]))
    #     print('{:>8}'.format('23') + " " + '{:35}'.format(nc.HEADCM01.long_name.title()) + '{:15}'.format(
    #         nc.HEADCM01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.HEADCM01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.HEADCM01.attrs["data_max"]))
    #     print('{:>8}'.format('24') + " " + '{:35}'.format(nc.ROLLGP01.long_name.title()) + '{:15}'.format(
    #         nc.ROLLGP01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.ROLLGP01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.ROLLGP01.attrs["data_max"]))
    #     print('{:>8}'.format('25') + " " + '{:35}'.format(nc.TEMPPR01.long_name.title()) + '{:15}'.format(
    #         nc.TEMPPR01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.TEMPPR01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.TEMPPR01.attrs["data_max"]))
    #     print('{:>8}'.format('26') + " " + '{:35}'.format(nc.DISTTRAN.long_name.title()) + '{:15}'.format(
    #         nc.DISTTRAN.attrs["units"]) + '{:>17}'.format('%.6E' % nc.DISTTRAN.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.DISTTRAN.attrs["data_max"]))
    #     print('{:>8}'.format('27') + " " + '{:35}'.format(nc.PPSAADCP.long_name.title()) + '{:15}'.format(
    #         nc.PPSAADCP.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PPSAADCP.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.PPSAADCP.attrs["data_max"]))
    #     print('{:>8}'.format('28') + " " + '{:35}'.format(nc.PRESPR01.long_name.title()) + '{:15}'.format(
    #         nc.PRESPR01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.PRESPR01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.PRESPR01.attrs["data_max"]))
    #     print('{:>8}'.format('29') + " " + '{:35}'.format(nc.PRESPR01_QC.long_name.title()) + '{:15}'.format(
    #         ' ') + '{:>17}'.format('%.6E' % nc.PRESPR01_QC.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.PRESPR01_QC.attrs["data_max"]))
    #     print('{:>8}'.format('30') + " " + '{:35}'.format(nc.SVELCV01.long_name.title()) + '{:15}'.format(
    #         nc.SVELCV01.attrs["units"]) + '{:>17}'.format('%.6E' % nc.SVELCV01.attrs["data_min"]) + '{:>17}'.format(
    #         '%.6E' % nc.SVELCV01.attrs["data_max"]))

    # Add in table of Channel summary
    print('{:>8}'.format('$END'))
    print()
    print('{:>26}'.format('$TABLE: CHANNEL DETAIL'))
    print('    ' + '! No  Pad            Start  Width  Format      Type  Decimal_Places')
    print('    ' + '!---  -------------  -----  -----  ----------  ----  --------------')
    # print('{:>8}'.format('1') + "  " + '{:15}'.format("' '") + '{:7}'.format(' ') + '{:7}'.format("' '") + '{:22}'.format('YYYY-MM-DDThh:mm:ssZ') + '{:6}'.format('D, T') + '{:14}'.format("' '"))

    # USE DICTIONARY
    for channel in channel_dict.keys():
        print('{:>8}'.format(channel_dict[channel]['channel_num']) +
              "  " + '{:15}'.format('%.6E' % channel_dict[channel]['pad']) +
              '{:7}'.format(' ') + '{:7}'.format(channel_dict[channel]['width']) +
              '{:12}'.format(channel_dict[channel]['format']) +
              '{:6}'.format(channel_dict[channel]['type']) +
              '{:14}'.format(channel_dict[channel]['decimal_places']))

    # print(
    #     '{:>8}'.format('1') + "  " + '{:15}'.format("' '") + '{:7}'.format(' ') + '{:7}'.format("' '") + '{:12}'.format(
    #         'YYYY-MM-DD') + '{:6}'.format('D') + '{:14}'.format("' '"))
    # print(
    #     '{:>8}'.format('2') + "  " + '{:15}'.format("' '") + '{:7}'.format(' ') + '{:7}'.format("' '") + '{:12}'.format(
    #         'HH:MM:SS') + '{:6}'.format('T') + '{:14}'.format("' '"))
    # print('{:>8}'.format('3') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('4') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('5') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('6') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('7') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('I') + '{:14}'.format('6'))
    # print('{:>8}'.format('8') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('I') + '{:14}'.format('6'))
    # print('{:>8}'.format('9') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('I') + '{:14}'.format('6'))
    # print('{:>8}'.format('10') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('11') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('12') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('13') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('14') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('15') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('16') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('17') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('18') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('19') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('20') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('21') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('22') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('23') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('24') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('25') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('26') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('27') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('28') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))
    # print('{:>8}'.format('29') + "  " + '{:15}'.format('%.6E' % nan) + '{:7}'.format(' ') + '{:7}'.format(
    #     '14') + '{:12}'.format('E') + '{:6}'.format('R4') + '{:14}'.format('6'))

    # Add in table of Channel detail summary
    print('{:>8}'.format('$END'))
    print()


def write_admin(nc):
    # define function to write administration section
    agency = nc.attrs["agency"]
    country = nc.attrs["country"]
    project = nc.attrs["project"]
    scientist = nc.attrs["scientist"]
    platform = nc.attrs["platform"]
    print("*ADMINISTRATION")
    print("    " + '{:20}'.format('AGENCY') + ": " + agency)
    print("    " + '{:20}'.format('COUNTRY') + ": " + country)
    print("    " + '{:20}'.format('PROJECT') + ": " + project)
    print("    " + '{:20}'.format('SCIENTIST') + ": " + scientist)
    print("    " + '{:20}'.format('PLATFORM ') + ": " + platform)
    print()


def decimalDegrees2DMS(value, lon_or_lat):
    # Function to convert decimal degree to degree & minutes
    degrees = int(value)
    submin = abs((Decimal(str(value)) - int(Decimal(str(value)))) * 60)
    minutes = str(submin)
    direction = ""
    if lon_or_lat == "Longitude":
        if degrees < 0:
            direction = "W"
        elif degrees > 0:
            direction = "E"
        else:
            direction = ""
    elif lon_or_lat == "Latitude":
        if degrees < 0:
            direction = "S"
        elif degrees > 0:
            direction = "N"
        else:
            direction = ""
    if abs(degrees) >= 100:
        notation = str(abs(degrees)) + "  " + str(minutes) + "0" * (8 - len(minutes)) + " " + \
                   "" + direction
    elif abs(degrees) < 100:
        notation = " " + str(abs(degrees)) + "  " + str(minutes) + "0" * (8 - len(minutes)) + " " + \
                   "" + direction
    else:
        notation = ""

    return notation


def write_location(nc):
    # Function to define geographic location
    geo_area = str(nc.geographic_area.values)
    station = nc.attrs["station"]
    lat = nc.attrs["latitude"]
    lat_string = decimalDegrees2DMS(lat, "Latitude")  # call decimal degree conversion function
    lon = nc.attrs["longitude"]
    lon_string = decimalDegrees2DMS(lon, "Longitude")  # call decimal degree conversion function
    water_depth = nc.attrs["water_depth"]
    if hasattr(nc, 'magnetic_declination'):
        mag_declination = nc.attrs["magnetic_declination"]
    if hasattr(nc, 'magnetic_variation'):
        mag_declination = nc.attrs["magnetic_variation"]

    print("*LOCATION")
    print("    " + '{:20}'.format('GEOGRAPHIC AREA') + ": " + geo_area)
    print("    " + '{:20}'.format('STATION') + ": " + station)
    print("    " + '{:20}'.format('LATITUDE') + ": " + lat_string + "  ! (deg min)")
    print("    " + '{:20}'.format('LONGITUDE') + ": " + lon_string + "  ! (deg min)")
    print("    " + '{:20}'.format('WATER DEPTH') + ": " + str(water_depth))
    print("    " + '{:20}'.format('MAGNETIC DECLINATION') + ": " + str(mag_declination))
    print()


def write_deployment_recovery(nc):
    # define function to write deployment & recovery info
    mission_deployment = nc.attrs["deployment_cruise_number"]
    deployment_type = nc.attrs["deployment_type"]
    if type(nc.attrs["anchor_drop_time"]) == str:
        anchor_drop_time = nc.attrs["anchor_drop_time"]
    else:
        anchor_drop_time = nc.attrs["anchor_drop_time"][:-4]
    remark = nc.attrs["anchor_type"]
    mission_recovery = nc.attrs["return_cruise_number"]
    if type(nc.attrs["anchor_release_time"]) == str:
        anchor_release_time = nc.attrs["anchor_release_time"]
    else:
        anchor_release_time = nc.attrs["anchor_release_time"][:-4]

    print("*DEPLOYMENT")
    print("    " + '{:20}'.format('MISSION') + ": " + mission_deployment)
    print("    " + '{:20}'.format('TYPE') + ": " + deployment_type)
    if type(nc.attrs["anchor_drop_time"]) == str:
        print("    " + '{:20}'.format('TIME ANCHOR DROPPED') + ": " + anchor_drop_time)
    else:
        print("    " + '{:20}'.format('TIME ANCHOR DROPPED') + ": UTC " + anchor_drop_time + ".000")
    print("    $REMARKS")
    print("        " + remark)
    print("    $END")
    print()
    print("*RECOVERY")
    print("    " + '{:20}'.format('MISSION') + ": " + mission_recovery)
    if type(nc.attrs["anchor_release_time"]) == str:
        print("    " + '{:20}'.format('TIME ANCHOR RELEASED') + ": " + anchor_release_time)
    else:
        print("    " + '{:20}'.format('TIME ANCHOR RELEASED') + ": UTC " + anchor_release_time + ".000")
    print()


def write_instrument(nc):
    # define function to write instrument info
    data_type = nc.attrs["data_type"].upper()
    model = nc.attrs["instrumentSubtype"] + "-" + nc.attrs["instrumentType"]
    if hasattr(nc, 'serial_number'):
        serial_number = nc.attrs["serial_number"]
    elif hasattr(nc, 'instrument_serial_number'):
        serial_number = nc.attrs["instrument_serial_number"]
    # serial_number = nc.attrs["serial_number"]  nc.attrs["instrument_serial_number"]
    depth = str(int(nc.attrs["instrument_depth"]))
    orientation = nc.attrs["orientation"]

    print("*INSTRUMENT")
    print("    TYPE                : " + model)
    print("    SERIAL NUMBER       : " + serial_number)
    print("    DEPTH               : " + depth)
    print("    ORIENTATION         : " + orientation)
    print()
    print("    $ARRAY: BIN DEPTHS (M)")
    n = nc.LCEWAP01["distance"].values.size
    for i in range(0, n):
        Bin_depth = str(round(nc.LCEWAP01["distance"].values[i], 2))
        num_space = 13 - len(str(Bin_depth))
        print(" " * num_space + Bin_depth)
    print("    $END")
    print("    $REMARKS")
    print("        Instrument depth in meters.")
    print("    $END")
    print()


def convert_timedelta(duration):
    # define function to find out time increment
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    m_seconds = int(seconds * 0.001)
    return str(days) + " " + str(hours) + " " + str(minutes) + " " + str(seconds) + " " + str(m_seconds)


def write_raw(nc):
    # define function to write raw info
    time_start = str(nc.coords["time"].values[0].astype('M8[s]')).replace("T", " ")
    time_first_good = nc.attrs["time_coverage_start"][:-4]
    time_end = nc.attrs["time_coverage_end"][:-4]
    time_increment_2 = nc.coords["time"].values[2].astype('M8[s]').astype('O')
    time_increment_1 = nc.coords["time"].values[0].astype('M8[s]').astype('O')
    time_increment = (time_increment_2 - time_increment_1) / 2
    time_increment_string = convert_timedelta(time_increment)  # call convert_timedelta function
    number_records = str(nc.time.shape[0])

    name = nc.attrs["instrumentSubtype"] + "-" + nc.attrs["instrumentType"]
    sourceprog = "instrument"
    prog_ver = nc.attrs["firmware_version"]
    config = "NA"  # nc.attrs["systemConfiguration"]
    beam_angle = str(nc.attrs["beam_angle"])
    numbeams = str(nc.attrs["number_of_beams"])  # numbeams = str(nc.attrs["janus"]), janus was used in R
    beam_freq = str(nc.attrs["frequency"])
    beam_pattern = nc.attrs["beam_pattern"]
    orientation = nc.attrs["orientation"]
    # simflag =  "??" # ?????????????????????????
    n_beams = str(nc.attrs["number_of_beams"])  # n_beams = str(nc.attrs["janus"])
    n_cells = str(nc.attrs["numberOfCells"])
    pings_per_ensemble = str(nc.attrs["pings_per_ensemble"])
    cell_size = str(nc.attrs["cellSize"])
    blank = str(nc.attrs['blank'])
    # prof_mode = "??" # ??????????????????????
    corr_threshold = str(nc.attrs["valid_correlation_range"])
    n_codereps = str(nc.attrs["n_codereps"])  # n_codereps = "NA" #
    min_pgood = str(nc.attrs["min_percent_good"])
    evel_threshold = nc.attrs["error_velocity_threshold"]
    time_between_ping_groups = str(nc.attrs['time_ping'])
    coord = "00011111"  # need check and confirm?
    coord_sys = nc.attrs["coord_system"]
    use_pitchroll = "yes"  # need check and confirm?
    use_3beam = "yes"  # need check and confirm?
    bin_mapping = "yes"  # need check and confirm?
    xducer_misalign = "0"  # need check and confirm?
    if hasattr(nc, 'magnetic_declination'):
        magnetic_var = str(nc.attrs["magnetic_declination"])
    if hasattr(nc, 'magnetic_variation'):
        magnetic_var = str(nc.attrs["magnetic_variation"])

    sensors_src = str(nc.attrs["sensor_source"])
    sensors_avail = str(nc.attrs["sensors_avail"])
    bin1_dist = "%.2f" % round(nc.attrs["bin1Distance"], 2)
    # xmit_pulse = "%.2f" % round(nc.attrs["transmit_pulse_length_cm"] / 100, 2)
    xmit_length = str(nc.attrs['xmit_length'])
    fls_target_threshold = nc.attrs["false_target_reject_values"]
    xmit_lag = str(nc.attrs["xmit_lag"])  # xmit_lag = "NA"
    # syspower = "??" # ??????????????????????
    # navigator_basefreqindex = "??" # ????????????????????
    h_adcp_beam_angle = str(nc.attrs["beam_angle"])
    # water_ref_cells = "??" #???????????????????
    # serialnum = "??" #??????????????????
    # sysbandwidth = "??" #?????????????????????
    # remus_serialnum = "??" #??????????????????

    print("*RAW")
    print("    " + '{:20}'.format('START TIME') + ": UTC " + time_start + ".000")
    print("    " + '{:20}'.format('FIRST GOOD REC TIME') + ": UTC " + time_first_good + ".000")
    print("    " + '{:20}'.format('END TIME') + ": UTC " + time_end + ".000")
    print("    " + '{:20}'.format('TIME INCREMENT') + ": " + time_increment_string + "  ! (day hr min sec ms)")
    print("    " + '{:20}'.format('NUMBER OF RECORDS') + ": " + number_records)
    print("    $REMARKS")
    print("        " + "The data and following metadata were extracted from the raw ADCP binary file using")
    #print("        " + "R script provided by Emily Chisholm to perform the processing and netCDF file output")
    print("        " + "a Python script adapted from Jody Klymak to perform the processing and netCDF file output")
    print()
    print("        " + '{:29}'.format('name:') + name)
    print("        " + '{:29}'.format('sourceprog:') + sourceprog)
    print("        " + '{:29}'.format('prog_ver:') + prog_ver)
    print("        " + '{:29}'.format('config:') + config)
    print("        " + '{:29}'.format('beam_angle:') + beam_angle)
    print("        " + '{:29}'.format('numbeams:') + numbeams)
    print("        " + '{:29}'.format('beam_freq:') + beam_freq)
    print("        " + '{:29}'.format('beam_pattern:') + beam_pattern)
    print("        " + '{:29}'.format('orientation:') + orientation)
    # print("        " + '{:29}'.format('simflag:') + simflag) # ???????????????????????????
    print("        " + '{:29}'.format('n_beams:') + n_beams)
    print("        " + '{:29}'.format('n_cells: ') + n_cells)
    print("        " + '{:29}'.format('pings_per_ensemble:') + pings_per_ensemble)
    print("        " + '{:29}'.format('cell_size:') + cell_size)
    print("        " + '{:29}'.format('blank:') + blank)  # ???????????????????????????
    # print("        " + '{:29}'.format('prof_mode:') + prof_mode) # ???????????????????????
    print("        " + '{:29}'.format('corr_threshold:') + corr_threshold)
    print("        " + '{:29}'.format('n_codereps:') + n_codereps)
    print("        " + '{:29}'.format('min_pgood:') + min_pgood)
    print("        " + '{:29}'.format('evel_threshold:') + evel_threshold)
    print("        " + '{:29}'.format('time_between_ping_groups:') + time_between_ping_groups)
    print("        " + '{:29}'.format('coord:  ') + coord)
    print("        " + '{:29}'.format('coord_sys:') + coord_sys)
    print("        " + '{:29}'.format('use_pitchroll:') + use_pitchroll)
    print("        " + '{:29}'.format('use_3beam: ') + use_3beam)
    print("        " + '{:29}'.format('bin_mapping:') + bin_mapping)
    print("        " + '{:29}'.format('xducer_misalign:') + xducer_misalign)
    print("        " + '{:29}'.format('magnetic_var:') + magnetic_var)
    print("        " + '{:29}'.format('sensors_src: ') + sensors_src)
    print("        " + '{:29}'.format('sensors_avail:') + sensors_avail)
    print("        " + '{:29}'.format('bin1_dist: ') + bin1_dist)
    print("        " + '{:29}'.format('xmit_length:') + xmit_length)
    print("        " + '{:29}'.format('fls_target_threshold:') + fls_target_threshold)
    print("        " + '{:29}'.format('xmit_lag:') + xmit_lag)
    # print("        " + '{:29}'.format('syspower:') + syspower)
    # print("        " + '{:29}'.format('navigator_basefreqindex:') + navigator_basefreqindex)
    print("        " + '{:29}'.format('h_adcp_beam_angle:') + h_adcp_beam_angle)
    # print("        " + '{:29}'.format('water_ref_cells:'))
    # print("        " + water_ref_cells)
    # print("        " + '{:29}'.format('serialnum:'))
    # print("        " + serialnum)
    # print("        " + '{:29}'.format('sysbandwidth:'))
    # print("        " + sysbandwidth)
    # print("        " + '{:29}'.format('remus_serialnum:'))
    # print("        " + remus_serialnum)

    print("        " + '{:29}'.format('ranges:'))
    n = nc.LCEWAP01["distance"].values.size
    for i in range(0, n):
        Bin_depth = str(round(nc.LCEWAP01["distance"].values[i], 2))
        num_space = 13 - len(str(Bin_depth))
        print('{:>14}'.format(Bin_depth))

    print("    $END")
    print()


def write_history(nc, f_name, ds_is_segment=False, ctd_pressure_file=None):
    # define function to write raw info
    process_1 = "ADCP2NC "
    process_1_ver = '1'  # str(nc.attrs["pred_accuracy"])
    date_time_1 = nc.attrs["date_modified"]
    digits_in_process_hist = [
        int(s) for s in nc.attrs['processing_history'].split() if s.isdigit()
    ] #H.Hourston May 27, 2020
    Recs_in_1 = str(digits_in_process_hist[-2] + digits_in_process_hist[-1] + nc.coords["time"].size)
    Recs_out_1 = str(nc.coords["time"].size)
    process_2 = "NC2IOS "
    process_2_ver = "1.0"
    now = datetime.now()  # dd/mm/YY H:M:S
    date_time_2 = now.strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7]
    Recs_in_2 = Recs_out_1
    Recs_out_2 = Recs_out_1

    # Add note about L2 processing steps if applicable
    if ds_is_segment:
        nc.attrs['history'] += ' The dataset was split into segments where water depth changed from mooring strike(s).'

    if ctd_pressure_file is not None:
        nc.attrs['history'] += f' The ADCP dataset was missing (good quality) pressure sensor data,' \
                               f' so pressure sensor data from {ctd_pressure_file} was merged with the ADCP dataset.'

    n = len(nc.history.split(". "))  # n = len(nc.processing_history.split(". "))

    print("*HISTORY")
    print()
    print("    !   Name     Vers  Date       Time     Recs In   Recs Out")
    print("    !   -------- ----  ---------- -------- --------- ---------")
    print("        " + '{:9}'.format(process_1) + '{:6}'.format(process_1_ver) + '{:9}'.format(
        date_time_1) + '{:>9}'.format(Recs_in_1) + '{:>10}'.format(Recs_out_1))
    print("        " + '{:9}'.format(process_2) + '{:6}'.format(process_2_ver) + '{:9}'.format(
        date_time_2) + '{:>9}'.format(Recs_in_2) + '{:>10}'.format(Recs_out_2))
    print("    $END")
    print("    $REMARKS")
    print("        -" + process_1 + " processing: " + date_time_1)
    for i in range(0, n):
        print("         " + nc.history.split(". ")[i] + ".")
        # print("         " + nc.processing_history.split(". ")[i] + ".")
    # print("         " + '{:100}'.format(nc.history.split(". ")[i]))
    # adding more processing content, check with Hana
    print("        -" + process_2 + " processing: " + date_time_2)
    print("         " + "NetCDF file converted to IOSShell format.")

    print("    $END")
    print()
    print("*COMMENTS")
    print("    To get the actual data, please see " + f_name)
    print()
    print("*END OF HEADER")
    return


def main_header(f, dest_dir, ds_is_segment=False, ctd_pressure_file=None):
    #Start
    in_f_name = f.split("/")[-1]
    # Create subdir for new netCDF file if one doesn't exist yet
    newnc_dir = './{}/newnc/'.format(dest_dir)
    if not os.path.exists(newnc_dir):
        os.makedirs(newnc_dir)
    f_output = newnc_dir + in_f_name.split(".")[0] + ".adcp"
    # print(f_output) prints to previously opened f_output in line 730
    nc_file = xr.open_dataset(f)

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S.%f")[0:-4]
    IOS_string = '*IOS HEADER VERSION 2.0      2020/03/01 2020/04/15 PYTHON' # ?? check with Germaine on the dates

    orig_stdout = sys.stdout
    file_handle = open(f_output, 'wt')
    try:
        sys.stdout = file_handle
        print("*" + dt_string)
        print(IOS_string)
        print() # print("\n") pring("\n" * 40)
        write_file(nc=nc_file)
        write_admin(nc=nc_file)
        write_location(nc=nc_file)
        write_deployment_recovery(nc=nc_file)
        write_instrument(nc=nc_file)
        write_raw(nc=nc_file)
        write_history(nc=nc_file, f_name=in_f_name, ds_is_segment=ds_is_segment,
                      ctd_pressure_file=ctd_pressure_file)
        sys.stdout.flush() #Recommended by Tom
    finally:
        sys.stdout = orig_stdout
    return os.path.abspath(f_output)


def example_usage_header():
    # Input
    in_file = './newnc/a1_20050503_20050504_0221m.adcp.L1.nc'
    dest_dir = 'dest_dir'
    header_name = main_header(f=in_file, dest_dir=dest_dir)
    return header_name
