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
from pycurrents.adcp.rdiraw import Multiread


# Specify raw ADCP file to create nc file from
inFile = "/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/20568-A1-56.000" #raw .000 file
file_meta = "/home/hourstonh/Documents/Hana_D_drive/ADCP_processing/ADCP/a1_20160713_20170513_0480m/P01/a1_20160713_20170513_0480m_meta_L1.csv" #csv metadata file

# Splice file name to get output netCDF file name
outname = os.path.basename(inFile)[:-3] + 'adcp.L1.nc'; print(outname)

# Get model and timestamp from raw file csv metadata file
# Instrument frequency is not included in the metadata files, so not included in "model". Frequency optional, apparently
model = ""
timestamp = ""
with open(file_meta) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    line_count = 0
    for row in csv_reader:
        if row[0] == "instrumentSubtype":
            if row[1] == "Workhorse":
                model = "wh"
            elif row[1] == "Broadband":
                model = "bb"
            elif row[1] == "Narrowband":
                model = "nb"
            elif row[1] == "Sentinel V": #not sure if Multiread() function supports Sentinel V or MC DCP files
                model = ""
            elif row[1] == "Multi-Cell Doppler Current Profiler":
                model = ""
            else:
                continue
        else:
            continue

    if model == "":
        print("No valid instrumentSubtype detected")


#model = 'wh75' #A1-56 file

# Not sure what multiread does
# Multiread(fnames, sonar, yearbase=None, alias=None, gap=None, min_nprofs=1, gbinsonar=None, ibad=None)
# sonar: 'nb', 'bb', 'wh', or 'os' optionally followed by freq and pingtype, or a Sonar instance
m = Multiread(inFile, model) #m generates $ <pycurrents.adcp.rdiraw.Multiread at 0x7f05b6eecd10>
data = m.read() #data generates the kind of output we're looking for

from pycurrents.adcp.transform import heading_rotate
# import pycurrents.adcp.adcp_nc as adcp_nc
# from pycurrents.adcp.transform import rdi_xyz_enu
import pycurrents.adcp.transform as transform

# Get timestamp from "data" object just created
# In R, the start time was obtained from the "adp" object created within R
data_origin = str(data.yearbase) + '-01-01'

#time = pd.to_datetime(data.dday, unit='D', origin=pd.Timestamp('2016-01-01')) #original code line
time = pd.to_datetime(data.dday, unit='D', origin=data_origin) #pandas; use 1970-01-01 ? 'unix' = '1970-01-01; default


# Make into netCDF file
out = xr.Dataset(coords={'depth': data.dep, 'time': time, 'beam': [0, 1, 2, 3]},
                 data_vars={'vel':(['time', 'depth', 'beam'], data.vel),
                            'amp':(['time', 'depth', 'beam'], data.amp),
                            'cor':(['time', 'depth', 'beam'], data.cor),
                            'ens_num': (['time'], data.ens_num),
                            'heading': (['time'], data.heading),
                            'pitch': (['time'], data.pitch),
                            'roll': (['time'], data.roll),
                            'temperature': (['time'], data.temperature),
                            'xducerdepth': (['time'], data.XducerDepth),
                            'yday': (['time'], data.dday)})

# to here, things have run without error

# system configuration keys added to dataset
for key in data.sysconfig.keys():
    print(int(data.sysconfig[key]))
    print((data.sysconfig[key] is not bool))
    print(type(data.sysconfig[key]) is not bool)
    if type(data.sysconfig[key]) is not bool:
        out.attrs['sysconfig' + key] = data.sysconfig[key]
    else:
        out.attrs['sysconfig' + key] = int(data.sysconfig[key])

# Write data to variables
out.attrs['NBeams'] = data.NBeams #need to make pycurrents work in pycharm in order to see if variable names in 'data' are the same as Jody's
out.attrs['nprofs'] = data.nprofs
out.attrs['NCells'] = data.NCells
out.attrs['NPings'] = data.NPings
out.attrs['Bin1Dist'] = data.Bin1Dist
out.attrs['Blank'] = data.Blank

for key in data.trans.keys():
    print(type(data.trans[key]))
    if type(data.trans[key]) is not bool:
        out.attrs['trans' + key] = data.trans[key]
    else:
        out.attrs['trans' + key] = int(data.trans[key])

out.attrs['CellSize'] = data.CellSize
out.attrs['pingtype'] = data.pingtype
out.attrs['Pulse'] = data.Pulse

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
#    return

# Plot transducer depth: from jody's code
fig, ax = plt.subplots()
ax.plot(data.XducerDepth)
ax.title.set_text('Transducer depth')
fig.savefig('transducerdepth.jpeg')


