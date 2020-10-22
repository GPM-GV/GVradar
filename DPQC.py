#
# Python based Quality Control software utilizing PyArt 
# Develop by the NASA GPM-GV group
# V0.1 - 09/18/2020 - update by Jason Pippitt NASA/GSFC/SSAI
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
from copy import deepcopy
import gpm_dp_utils as gu
import gpm_dpqc as qc
import plot_dpqc_images as pltqc
import warnings
warnings.filterwarnings("ignore")

# *******************************************  M  A  I  N  **************************************************

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        sys.exit("Usage: python DPQC.py threshold_dictionary.txt")

    current_dict = sys.argv[1]

    #Import thresholds and in out dirs from txt file
    print()
    print('QC parameters:    ')
    print()
    thresh_dict = ast.literal_eval(open(current_dict).read())
    print(thresh_dict)

    # Get list of files that fall within the specified start/end times for this day
    files = gu.get_files_within_time_window(thresh_dict)
 
    numf = len(files)
    print()
    print("Processing " + str(numf) + " files.")
    print()
    for file in files:
        print('<-- ' + file)
        fileb = os.path.basename(file)

        # Load radar object
        try:
            radar = pyart.io.read(file, file_field_names=True)
           
        #Add ZZ field (raw reflectivity)
            zz = deepcopy(radar.fields['DBT2'])
         
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
                
        # Create Temperature field   
            radar = qc.get_ruc_sounding(radar, thresh_dict)

        # Apply CSU_RT filters
            radar = qc.csu_filters(radar, thresh_dict)
                                                  
        #Apply gatefilters on DP fields
            qc.threshold_qc_dpfields(radar, thresh_dict)

        # Get PhiDP and Std(PhiDP)
            qc.unfold_phidp(radar)
        
        # Get KDP
            qc.calculate_kdp(radar)
        
        # Create mask to filter a range/azm/height sector of radar based on SD threshold
            if thresh_dict['do_sd_sector'] == True:
                radar = qc.sd_sector(radar, thresh_dict)
                  
        # Create mask to filter a range/azm/height sector of radar based on PH threshold
            if thresh_dict['do_ph_sector'] == True:
                radar = qc.ph_sector(radar, thresh_dict)
                
        # Perform gatefilters for calculated fields
            qc.threshold_qc_calfields(radar, thresh_dict)
  
        # Rename radar fields to GPM standard names (e.g., CZ, DR, KD)
            print("Renaming radar fields...")
            radar.add_field('DBT2', zz, replace_existing=True)
            
        # Lists of current and final field names for QC'd file and fields to drop from radar object
            old_fields = ['DBT2','DBZ2', 'VEL2', 'WIDTH2', 'ZDR2', 'KDPB', 'PHM', 'RHOHV2', 'STDPHIB', 'SQI2']
            new_fields = ['ZZ',  'CZ',   'VR',   'SW',     'DR',   'KD',   'PH',  'RH',     'SD',      'SQ']
            qc.rename_fields_in_radar(radar, old_fields, new_fields)
            
        # Remove unwanted fields from radar
            print("Removing unwanted fields...")
            qc.remove_fields_from_radar(radar, thresh_dict)
            
            print()
            print("FINAL FIELDS -->  ")
            print(radar.fields.keys())
            print()
        
        # Write cfRadial file
            qc.output_cf(radar, thresh_dict)

        # Plotting images 
            if thresh_dict['plot_images'] == True:   
                pltqc.plot_fields(radar, thresh_dict)

        except:
            print()
            print("An error occured:  " + file)

        # Clean up python memory before next file
        del radar       
            
print("Done.")

