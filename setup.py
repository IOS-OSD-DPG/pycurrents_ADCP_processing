import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycurrents_ADCP_processing",
    version="0.0.1",
    author="Hana Hourston, Di Wan, Lu Guan",
    author_email="hana.hourston@dfo-mpo.gc.ca",
    description="A Python package for processing Acoustic Doppler Current Profiler (ADCP) data and exporting it in netCDF file format.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hhourston/pycurrents_ADCP_processing",
    install_requires=['numpy', 'xarray', 'pandas', 'datetime', 'gsw', 'netCDF4', 'scipy', 'matplotlib', 'shapely'],
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3"],
    python_requires='~=3.7',
)
