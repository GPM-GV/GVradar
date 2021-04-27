# GVradar
Global Precipitation Measurement (GPM) Ground Validation (GV) radar processing software (GVradar). 
* Python based Dual Pol Quality Control (DPQC), utilizing the Python Atmospheric Radiation Measurement (ARM) Radar Toolkit (Py-ART) and CSU Radar Tools.
* Precipitation product generation from Dual Pol, utilizing the Python Atmospheric Radiation Measurement (ARM) Radar Toolkit (Py-ART) and CSU Radar Tools.

Dependencies
============

Required dependencies to run GVradar:

* [PyArt](https://arm-doe.github.io/pyart/)
* [NumPy](https://www.numpy.org/)
* [SciPy](https://www.scipy.org)
* [matplotlib](https://matplotlib.org/)
* [netCDF4](https://github.com/Unidata/netcdf4-python)
* [CSU_RadarTools](https://github.com/CSU-Radarmet/CSU_RadarTools)
* [SkewT](https://github.com/tjlang/SkewT)
* [cartopy](https://anaconda.org/conda-forge/cartopy)

If you would like to merge NEXRAD split cuts and remove MRLE scans:

* [rsl_in_idl](https://trmm-fc.gsfc.nasa.gov/trmm_gv/software/rsl/)

Installing GVradar
==================

We suggest creating an environment, GVradar.  Intall the following programs.

Download [CSU Radar Tools](https://pmm-gv.gsfc.nasa.gov/pub/NPOL/temp/GVradar/CSU_RadarTools-master.tar.gz) tarball.
Download [SkewT](https://pmm-gv.gsfc.nasa.gov/pub/NPOL/temp/GVradar/SkewT-master.tar.gz) tarball.
Download [GVradar](https://pmm-gv.gsfc.nasa.gov/pub/NPOL/temp/GVradar/GVradar.tar.gz) tarball.

Execute the following command in the active GVradar environment for each above program:

    python setup.py install

Running GVradar
===============

usage: GVradar.py [-h] [--thresh_dict THRESH_DICT] [--product_dict PRODUCT_DICT] [--do_qc] [--dp_products] file

User information

positional arguments:

    file  File to process

optional arguments:

    -h, --help                      show this help message and exit

    --thresh_dict THRESH_DICT       Threshold dictionary

    --product_dict PRODUCT_DICT     DP product dictionary

    --do_qc                         Run QC

    --dp_products                   Create DP products

Perform DPQC
============

    python GVradar.py --do_qc  file
    
Calculate Precipitation Products
===============================

    python GVradar.py --dp_products  file
