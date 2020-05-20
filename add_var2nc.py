# Author: Di Wan
# Adding geographic area for all .adcp.nc files
import xarray as xr
import glob as glob
import sys
import os
import utils as utils
# Library detail: https://github.com/cioos-siooc/cioos-siooc_data_transform/tree/master/cioos_data_transform/ios_data_transform
# Credit: CIOOS Pacific
from shapely.geometry import Point


def add_geo(ncfile):
   
    data_xr = xr.open_dataset(ncfile)
    lon = data_xr.ALONZZ01
    lat = data_xr.ALATZZ01

    data_xr.attrs['_FillValue'] = 1e35
    # Geojson definitions for IOS
    
    json_file = './pyutils/ios_polygons.geojson'
    polygons_dict = utils.read_geojson(json_file)
    data_xr['geographic_area'] = utils.find_geographic_area(polygons_dict, Point(lon, lat))
    print(utils.find_geographic_area(polygons_dict, Point(lon, lat)))
    print(ncfile)
    print('New file is located at: ', './newnc/' + ncfile[9::])
    data_xr.to_netcdf('./newnc/' + ncfile[9::])


def get_files(archive_dir):

    print('Current Working Directory: ', os.getcwd())
    list_of_profiles = glob.glob(archive_dir + '*.nc', recursive=True)
    print(archive_dir + '*.nc')
    print(list_of_profiles)

    for profile in list_of_profiles:
        print(profile)
        add_geo(profile)

# directory for all the netcdf ADCP files
archive_dir_test = './ncdata/'

# Add geographic areas to the ncfile
get_files(archive_dir_test)

