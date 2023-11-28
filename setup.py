import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycurrents_ADCP_processing",
    version="0.0.2",
    author="Hana Hourston, Di Wan, Lu Guan, Maxim Krassovski",
    author_email="hana.hourston@dfo-mpo.gc.ca",
    description="A Python package for processing Acoustic Doppler Current Profiler (ADCP) data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IOS-OSD-DPG/pycurrents_ADCP_processing",
    install_requires=['numpy', 'xarray', 'pandas', 'datetime', 'gsw', 'netCDF4', 'scipy', 'matplotlib', 'shapely',
                      'ruamel.yaml'],
    packages=setuptools.find_packages(),
    package_data={'pycurrents_ADCP_processing': ['*.geojson', '*.yml']},
    classifiers=["Programming Language :: Python :: 3"],
    python_requires='~=3.7',
)
