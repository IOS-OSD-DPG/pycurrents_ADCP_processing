# Author: Di Wan
# Adding geographic area for all .adcp.nc files
import xarray as xr
import glob as glob
import sys
import os
from pycurrents_ADCP_processing import utils
# Library detail: https://github.com/cioos-siooc/cioos-siooc_data_transform/tree/master/cioos_data_transform/ios_data_transform
# Credit: Pramod Thupaki
from shapely.geometry import Point


def add_geo(ncfile, dest_dir):
    # Function returns full path of the netCDF file it creates
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
    # Create subdir for new netCDF file if one doesn't exist yet
    if not os.path.exists('./{}/newnc/'.format(dest_dir)):
        os.makedirs('./{}/newnc/'.format(dest_dir))
    new_name = './{}/newnc/{}'.format(dest_dir, os.path.basename(ncfile))
    print('New file is located at: ', new_name)
    data_xr.to_netcdf(new_name)
    return os.path.abspath(new_name)


def get_files(archive_dir):

    print('Current Working Directory: ', os.getcwd())
    list_of_profiles = glob.glob(archive_dir + '*.nc', recursive=True)
    print(archive_dir + '*.nc')
    print(list_of_profiles)

    for profile in list_of_profiles:
        print(profile)
        abs_path = add_geo(profile)


def example_usage_geo():
    # directory for all the netcdf ADCP files
    archive_dir_test = './ncdata/'

    # Add geographic areas to the ncfile
    get_files(archive_dir_test)

