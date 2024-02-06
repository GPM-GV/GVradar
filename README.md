# GVradar
Global Precipitation Measurement (GPM) Ground Validation (GV) radar processing software (GVradar). 
* Python based Dual Pol Quality Control (DPQC), utilizing the Python Atmospheric Radiation Measurement (ARM) Radar Toolkit (Py-ART) and CSU Radar Tools.
* Precipitation product generation from Dual Pol, utilizing the Python Atmospheric Radiation Measurement (ARM) Radar Toolkit (Py-ART) and CSU Radar Tools.

Dependencies
============

Required dependencies to run GVradar:

* [PyArt](https://github.com/ARM-DOE/pyart)
* [NumPy](https://www.numpy.org)
* [SciPy](https://www.scipy.org)
* [matplotlib](https://matplotlib.org)
* [netCDF4](https://github.com/Unidata/netcdf4-python)
* [CSU_RadarTools](https://github.com/CSU-Radarmet/CSU_RadarTools)
* [SkewT](https://github.com/tjlang/SkewT)
* [cartopy](https://anaconda.org/conda-forge/cartopy)

Installing GVradar
==================

We suggest creating an environment, GVradar.  Install the following programs.

* Download [CSU Radar Tools](https://pmm-gv.gsfc.nasa.gov/pub/NPOL/temp/GVradar/CSU_RadarTools-master.tar.gz) tarball.
* Download [SkewT](https://pmm-gv.gsfc.nasa.gov/pub/NPOL/temp/GVradar/SkewT-master.tar.gz) tarball.
* Download [GVradar](https://pmm-gv.gsfc.nasa.gov/pub/NPOL/temp/GVradar/GVradarV1.5.tar.gz) tarball.
* If you would like to test with data used in notebooks, Download [GVradar](https://pmm-gv.gsfc.nasa.gov/pub/NPOL/temp/GVradar/GVradarV1.5_raw.tar.gz) tarball.

Execute the following command in the active GVradar environment for each above program:

    python setup.py install
    
Installing from source
======================

Checkout the GVradar repository:

    git clone https://github.com/GPM-GV/GVradar.git
    
Updating from source in clone directory:

    git pull https://github.com/GPM-GV/GVradar.git

Install using pip from within GVradar directory:

    pip install -e .
    
or

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
    
Process Large Data Sets with run_multi_GVradar.py
=============================================

usage: run_multi_GVradar.py [-h] [--stime STIME [STIME ...]]
                            [--etime ETIME [ETIME ...]]
                            [--thresh_dict THRESH_DICT]
                            [--product_dict PRODUCT_DICT] [--do_qc]
                            [--dp_products]
                            in_dir

User information

positional arguments:
  
    in_dir                Input Directory

optional arguments:
  
    -h, --help            show this help message and exit
  
    --stime STIME [STIME ...]    Year Month Day Hour Minute ex: 2020 1 1 0 59
  
    --etime ETIME [ETIME ...]    Year, Month, Day, Hour, Minute ex: 2020 1 1 23 59
  
    --thresh_dict THRESH_DICT    Threshold dictionary
  
    --product_dict PRODUCT_DICT    DP product dictionary
  
    --do_qc               Run QC
  
    --dp_products         Create DP products    
    
python run_multi_GVradar.py /raw/dir/ --stime 2020 10 29 15 0 --etime 2020 10 29 15 5 --do_qc --dp_products
