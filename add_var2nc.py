# Author: Di Wan
# Adding geographic area for all .adcp files
import xarray as xr
import glob as glob
import sys
import os
sys.path.insert(0, os.getcwd()+'/../')
import ios_data_transform as iod
from shapely.geometry import Point


def add_geo(ncfile):
    # print(ncfile)
    data_xr = xr.open_dataset(ncfile)
    lon = data_xr.ALONZZ01
    lat = data_xr.ALATZZ01
    # print(data_xr.attrs['_FillValue'])
    data_xr.attrs['_FillValue'] = 1e35

    json_file = './tests/test_files/ios_polygons.geojson'
    polygons_dict = iod.utils.read_geojson(json_file)
    data_xr['geographic_area'] = iod.utils.find_geographic_area(polygons_dict, Point(lon, lat))
    print(iod.utils.find_geographic_area(polygons_dict, Point(lon, lat)))
    print('./newdata/' + ncfile[113::])
    data_xr.to_netcdf('./newdata/' + ncfile[113::])


def get_files(archive_dir):
    # list_of_profiles = glob.glob(archive_dir + 'netCDF_Data/ADCP/*.nc', recursive=True)
    list_of_profiles = glob.glob(archive_dir + '*.nc', recursive=True)

    for profile in list_of_profiles:
        print(profile)
        add_geo(profile)

# directory for all the netcdf ADCP files
archive_dir_test = '/run/user/1000/gvfs/smb-share:server=sid01hnas01b,share=osd_data/OSD_DataArchive/osd_data_final/netCDF_Data/ADCP/'
# archive_dir_test = './data/'
json_file = './tests/test_files/ios_polygons.geojson'
get_files(archive_dir_test)
