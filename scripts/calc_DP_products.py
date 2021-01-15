#
# Software utilizing PyArt to create rainfall 
# products from Dual Pol data
#
# Develop by the NASA GPM-GV group
# V0.2 - 12/16/2020 - update by Jason Pippitt NASA/GSFC/SSAI
#

from __future__ import print_function
import numpy as np
import pyart
import glob
import ast
import sys
import os
import argparse
import plot_dpqc_images as pltqc
import gpm_dp_utils as gu
import gpm_dpqc as gpm
import datetime
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain, 
                            csu_dsd, csu_kdp, csu_misc, fundamentals)
import warnings
warnings.filterwarnings("ignore")

# ******************************************************************************************************

if __name__ == '__main__':
    
    runargs = argparse.ArgumentParser(description='User information')
    runargs.add_argument('--file', dest='file', type=str, help='File to process')
    runargs.add_argument('--product_dict', dest='product_dict', type=str, help='DP product dictionary')

    args = runargs.parse_args() 

    if args.file == None: sys.exit("Please specify input file, --file /file/location/radar_file") 
    if args.product_dict == None: print('No product dictionary, applying defaults.')

    file = args.file     

    #Import thresholds and in out dirs
    if args.product_dict == None:
        DP_product_dict = gpm.get_default_product_dict()
    else:
        current_dict = args.product_dict
        DP_product_dict = ast.literal_eval(open(current_dict).read())
    
    print('DP products parameters:    ')
    print(DP_product_dict)

    print()
    print('<-- ' + file)
    fileb = os.path.basename(file)

    # Load radar object
    radar = pyart.io.read(file, file_field_names=True)

    # Create Temperature field   
    if thresh_dict['use_sounding'] == True:
        if thresh_dict['sounding_type'] == 'ruc':
            radar = gpm.use_ruc_sounding(radar, thresh_dict)
        if thresh_dict['sounding_type'] == 'uwy':
            radar = gpm.use_uwy_sounding(radar, thresh_dict)
        if thresh_dict['sounding_type'] == 'ruc_archive':
            radar = gpm.get_ruc_sounding(radar, thresh_dict)
    if thresh_dict['use_sounding'] == False:
        radar = gpm.get_beam_height(radar)  

    dz = radar.fields['CZ']['data']
    dr = radar.fields['DR']['data']
    kd = radar.fields['KD']['data']
    rh = radar.fields['RH']['data']
    radar_T = radar.fields['TEMP']['data']
    radar_z = radar.fields['HEIGHT']['data']

    # Get HID scores and add FH to radar
        
    print('Add HID field to radar...')
    radar_band = 'S'
    radar, fh = gu.add_csu_fhc(radar, dz, dr, rh, kd, radar_band, radar_T)

    # Calculate Cifelli et al. 2011 ice and water mass fields and add to radar.
    # Function expects, reflectivity, differential reflectivity, 
    # and altitude (km MSL) at a minimum. Temperature is optional.
    print('Calculating water and ice mass...')
    radar = gu.add_csu_liquid_ice_mass(radar, dz, dr, radar_z/1000.0, radar_T)
        
    # Calculate Blended-rain from Cifelli et al. 2011 and addr to radar
    print('Calculating blended rainfall field...')
    radar = gu.add_csu_blended_rain(radar, dz, dr, kd, fh)           
              
    # Calculate Drop-Size Distribution from Tokay et al. 2020 and add to radar
    print('Calculating Drop-Size Distribution...')
    radar = gu.add_calc_dsd_sband_tokay_2020(radar, dz, dr, 'wff')
        
    # Plot radar images
    if DP_product_dict['plot_images'] == True:
        print()
        pltqc.plot_fields(radar, DP_product_dict)       
        
    # Remove unwanted fields from radar and write cfRadial
    if DP_product_dict['output_cf'] == True:
        gu.remove_undesirable_fields(radar, DP_product_dict)
        # Write cfRadial file
        gpm.output_cf(radar, DP_product_dict)

    print("Done.")

