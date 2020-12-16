# pyDPQC
Dual Pol Quality Control (DPQC)  and precipitation product package which utilizes the Python ARM Radar Toolkit ( Py -ART) and CSU Radar Tools. 

Dependencies
============

pyDPQC is tested to work under Python 3.8

Required dependencies to run pyDPQC in addition to Python are:

* PyArt <https://arm-doe.github.io/pyart/>
* NumPy <https://www.numpy.org/>
* SciPy <https://www.scipy.org>
* matplotlib <https://matplotlib.org/>
* netCDF4 <https://github.com/Unidata/netcdf4-python>
* CSU_RadarTools <https://github.com/CSU-Radarmet/CSU_RadarTools>
* SkewT <https://github.com/tjlang/SkewT>

If you would like to merge NEXRAD split cuts and remove MRLE scans:

* rsl_in_idl <https://trmm-fc.gsfc.nasa.gov/trmm_gv/software/rsl/>

Running pyDPQC
==============

To run DPQC:

         python DPQC.py --file /file/location/data_file --thresh_dict optional_threshold_dictionary.txt

For thresh_dict description see:
        thresh_dict_description.txt

Example thresh_dict:
        thresh_dict_example.txt

Calculating Precipitation Products
==================================

To run calc_DP_products:

        python calc_DP_products.py 'DP_product_dict.txt'
