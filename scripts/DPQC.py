#
# Python based Quality Control software utilizing PyArt 
# Developed by the NASA GPM-GV group
# V0.2 - 12/16/2020 - update by Jason Pippitt NASA/GSFC/SSAI
#

import numpy as np
import pyart
import pylab
import netCDF4
import math
import sys
import glob
import os
import ast
import gc
import argparse
from copy import deepcopy
import gpm_dp_utils as gu
import gpm_dpqc as qc
import plot_dpqc_images as pltqc
import warnings
warnings.filterwarnings("ignore")

# *******************************************  M  A  I  N  **************************************************

if __name__ == '__main__':
 
    runargs = argparse.ArgumentParser(description='User information')
    runargs.add_argument('--file', dest='file', type=str, help='File to process')
    runargs.add_argument('--thresh_dict', dest='thresh_dict', type=str, help='Threshold dictionary')

    args = runargs.parse_args() 

    if args.file == None: sys.exit("Please specify input file, --file /file/location/radar_file") 
    if args.thresh_dict == None: print('No threshold dictionary, applying default thresholds.')

    file = args.file

    #Import thresholds and in out dirs
    if args.thresh_dict == None:
        thresh_dict = qc.get_default_thresh_dict()
    else:
        current_dict = args.thresh_dict
        thresh_dict = ast.literal_eval(open(current_dict).read())

    print()
    print('QC parameters:    ')
    print()
    print(thresh_dict)
    print()
    
    print('Processing --> ' + file)
    fileb = os.path.basename(file)

    # Load radar object
    radar = pyart.io.read(file, file_field_names=True)

    # If radar is 88D convert file to cfRadial to organize split cuts and remove MRLE scans
    if radar.metadata['original_container'] == 'NEXRAD Level II':
        radar = qc.convert_to_cf(radar, file)
    # Create a filter to remove data beyond 200km
        radar = qc.mask_88D_200(radar, thresh_dict)
                
    # Rename fields with GPM, 2-letter IDs (e.g. CZ, DR, KD)
    qc.rename_fields_in_radar(radar)
    print(radar.fields.keys())

    # Save raw reflectivity, will be applied to DZ later
    zz = deepcopy(radar.fields['DZ'])
         
    #Apply calibration numbers to Reflectivity and ZDR fields
    if thresh_dict['apply_cal'] == True:
        radar = qc.calibrate(radar, thresh_dict)
                        
    # Create a filter to remove data close to radar location (e.g. cone of silence)
    if thresh_dict['do_cos'] == True:
        radar = qc.mask_cone_of_silence(radar, thresh_dict)

    # Create mask to filter a range/azm/height sector of radar
    if thresh_dict['do_sector'] == True:
        radar = qc.sector_wipeout(radar, thresh_dict)
           
    # Create mask to filter a range/azm/height sector of radar based on RHOHV threshold
    if thresh_dict['do_rh_sector'] == True:
        radar = qc.rh_sector(radar, thresh_dict)
                
    # Create Temperature and Height field
    radar = qc.get_ruc_sounding(radar, thresh_dict)

    # Apply CSU_RT filters
    radar = qc.csu_filters(radar, thresh_dict)
                                                  
    #Apply gatefilters on DP fields
    qc.threshold_qc_dpfields(radar, thresh_dict)

    # Get PhiDP
    qc.unfold_phidp(radar)
        
    # Get KDP and Std(PhiDP)
    qc.calculate_kdp(radar)
        
    # Create mask to filter a range/azm/height sector of radar based on SD threshold
    if thresh_dict['do_sd_sector'] == True:
        radar = qc.sd_sector(radar, thresh_dict)
                  
    # Create mask to filter a range/azm/height sector of radar based on PH threshold
    if thresh_dict['do_ph_sector'] == True:
        radar = qc.ph_sector(radar, thresh_dict)
                
    # Perform gatefilters for calculated fields
    qc.threshold_qc_calfields(radar, thresh_dict)
            
    # Remove unwanted fields from radar
    radar.add_field('DZ', zz, replace_existing=True)
    print("Removing unwanted fields...")
    qc.remove_fields_from_radar(radar)
            
    print()
    print("FINAL FIELDS -->  ")
    print(radar.fields.keys())
    print()
        
    # Write cfRadial file
    qc.output_cf(radar, thresh_dict)

    # Plotting images 
    if thresh_dict['plot_images'] == True:   
        pltqc.plot_fields(radar, thresh_dict)
  
print("Done.")

