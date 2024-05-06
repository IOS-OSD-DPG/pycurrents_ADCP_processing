from shapely.geometry import Polygon, Point
import json
import numpy as np
import xarray as xr
import os


# general utility functions common to multiple classes

def is_in(keywords, string):
    # simple function to check if any keyword is in string
    # convert string and keywords to upper case before checking
    return any([string.upper().find(z.upper()) >= 0 for z in keywords])


def import_env_variables(filename='./.env'):
    # import information in file to a dictionary
    # this file makes the implementation independent of local folder structure
    # data in file should be key:value pairs. Key should be unique
    info = {}
    with open(filename, 'r') as fid:
        lines = fid.readlines()
        for line in lines:
            if line.strip() == '':
                break
            elif line.strip()[0] == '#':
                continue
            info[line.split(':')[0].strip()] = line.split(':')[1].strip()
    return info


def file_mod_time(filename):
    # returns how old the file is based on timestamp
    # returns the time in hours
    import time
    import os
    dthrs = (os.path.getmtime(filename) - time.time()) / 3600.
    return dthrs


def release_memory(outfile):
    # release memory from file and variable class created.
    for c in outfile.varlist:
        del c
    del outfile


def read_geojson(filename):
    # read shapefile in geojson format into Polygon object
    # input geojson file
    # output: Polygon object
    with open(filename) as f:
        data = json.load(f)
    poly_dict = {}

    for feature in data['features']:
        if feature['geometry']['type'] == 'Polygon':
            # print(feature['geometry']['coordinates'][0])
            p = Polygon(feature['geometry']['coordinates'][0])
            name = feature['properties']['name']
            poly_dict[name] = p
    return poly_dict


def is_in_polygon(polygon, point):
    # identify if point is inside polygon
    return polygon.contains(point)


def find_geographic_area(poly_dict, point):
    name_str = ''
    for key in poly_dict:
        if is_in_polygon(poly_dict[key], point):
            name_str = '{}{} '.format(name_str, key.replace(' ', '-'))
            # print(name_str)
    return name_str


def find_geographic_area_attr(lon: float, lat: float):
    # Geojson definitions for IOS
    json_file = 'ios_polygons.geojson'
    json_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), json_file)
    # json_file = os.path.realpath(json_file)
    polygons_dict = read_geojson(json_file)
    return find_geographic_area(polygons_dict, Point(lon, lat))


def parse_processing_history(processing_history: str):
    """
    Get number of leading and trailing ensembles cut from an ADCP dataset
    """
    history_parts = processing_history.split('. ')  # Break into list of sentences

    leading_ens_cut = 0
    trailing_ens_cut = 0

    for part in history_parts:
        if part.startswith('Leading') and part.endswith('ensembles from before deployment discarded'):
            # Assume part has format: "Leading 16 ensembles from before deployment discarded"
            leading_ens_cut = int(part.split()[1])  # Split by space " "
        elif part.startswith('Trailing') and part.endswith('ensembles from after recovery discarded'):
            # Assume part has format: "Trailing 16 ensembles from after recovery discarded"
            trailing_ens_cut = int(part.split()[1])  # Split by space " "
        elif part.startswith('From the original time series with length'):
            # dsout.attrs['processing_history'] += (f" From the original time series with length {len(ds.time.data)},"
            #                                       f" the segment start index = {segment_start_idx} and the"
            #                                       f" segment end index = {segment_end_idx}.")
            len_original_time_series = int(part.split()[7].replace(',', ''))
            leading_ens_cut = int(part.split()[13]) - 1  # Convert start index to num leading ensembles cut
            trailing_ens_cut = len_original_time_series - int(part.split()[-1].replace('.', ''))

    return leading_ens_cut, trailing_ens_cut


def numpy_datetime_to_str_utc(t: np.datetime64):
    # Use split to remove decimal seconds from the time
    return t.astype(str).replace('T', ' ').split('.')[0] + ' UTC'


def geospatial_vertical_extrema(orientation: str, sensor_depth: float, distance: np.ndarray):
    """
    :param orientation: 'up' or 'down', from xr.Dataset.attrs['orientation']
    :param sensor_depth: mean of PPSAADCP
    :param distance: distance of each bin from the ADCP; dimension of the netCDF file
    """
    # print(orientation, sensor_depth, distance, sep='\n')
    if orientation == 'up':
        geospatial_vertical_min = float(sensor_depth) - np.nanmax(distance)
        geospatial_vertical_max = float(sensor_depth) - np.nanmin(distance)
    elif orientation == 'down':
        geospatial_vertical_min = float(sensor_depth) + np.nanmin(distance)
        geospatial_vertical_max = float(sensor_depth) + np.nanmax(distance)
    else:
        ValueError(f'Orientation value {orientation} invalid')

    # adcp float instrument_depth.data tends to acquire many decimal places for some reason
    num_decimals = len(str(distance[0]).split('.')[1])

    return np.round(geospatial_vertical_min, num_decimals), np.round(geospatial_vertical_max, num_decimals)


def round_to_int(x: float):
    """
    Use instead of numpy.round(), which uses "banker's rounding" which rounds 1.5 and 2.5 to 2 !!
    """
    if not np.isnan(x):
        return int(np.ceil(x)) if x % 1 >= .5 else int(np.floor(x))
    else:
        return np.nan


def vb_flag(dataset: xr.Dataset):
    """
    Create flag for missing vertical beam data in files from Sentinel V ADCPs
    flag = 0 if Sentinel V file has vertical beam data, or file not from Sentinel V
    flag = 1 if Sentinel V file does not have vertical beam data
    Inputs:
        - dataset: dataset-type object created by reading in a netCDF ADCP file
                   with the xarray package
    Outputs:
        - value of flag
    """
    try:
        x = dataset.TNIHCE05.data
        return 0
    except AttributeError:
        if dataset.instrument_subtype == 'Sentinel V':
            return 1
        else:
            return 0


def calculate_depths(dataset: xr.Dataset):
    """
    Calculate ADCP bin depths in the water column
    Inputs:
        - dataset: dataset-type object created by reading in a netCDF ADCP
                   file with the xarray package
    Outputs:
        - numpy array of ADCP bin depths
    """
    # depths = np.mean(ncdata.PRESPR01[0,:]) - ncdata.distance  #What Di used
    if dataset.orientation == 'up':
        return dataset.instrument_depth.data - dataset.distance.data
    else:
        return dataset.instrument_depth.data + dataset.distance.data


def mean_orientation(o: list):
    # orientation, o, is a list of bools with True=up and False=down
    if sum(o) > len(o) / 2:
        return 'up'
    elif sum(o) < len(o) / 2:
        return 'down'
    else:
        ValueError('Number of \"up\" orientations equals number of \"down\" '
                   'orientations in data subset')