from shapely.geometry import Polygon, Point
import json


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
