# ***************************************************************************************
'''
GPM-GV radar processing software.
    -Python based Quality Control utilizing PyArt 
    -Rainfall product generation from Dual Pol data

Developed by the NASA GPM-GV group
V0.4 - 04/22/2021 - update by Jason Pippitt NASA/GSFC/SSAI
'''
# ***************************************************************************************

import pyart
import sys, os
import ast
import argparse
import pathlib
from gvradar import (dp_products as dp, dpqc as qc, 
                     common as cm, plot_images as pi)
import warnings
warnings.filterwarnings("ignore")

# ***************************************************************************************

class QC:
    '''
    Creates a class QC with a constructor and 
    an instance method (run_dpqc).
    '''

    # Constructor:  Specifies necessary initialization parameters
    def __init__(self, file, **kwargs):
        
        self.file = file
        
        # Uncompress input file, and create radar structure
        cfy = pathlib.Path(file).suffix
        if cfy == '.gz': 
            self.file = cm.unzip_file(self.file)
            file_unzip = self.file      
            radar = pyart.io.read(self.file, file_field_names=True)
        else:
            radar = pyart.io.read(self.file, file_field_names=True)
        
        self.radar = radar
        
        # Check and fix missing user defined kwargs
        default_kw = qc.get_default_thresh_dict()
        kwargs = cm.check_kwargs(kwargs, default_kw)
        
        # Get site name and date time from radar
        site_time = cm.get_site_date_time(self.radar)
        kwargs = {**site_time, **kwargs}
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        print('', 'QC parameters:    ', '', kwargs, '',
              'Processing --> ' + self.file, sep='\n')     
        
        # Remove temp file
        if cfy == '.gz': os.remove(file_unzip)

    # Instance Method
    def run_dpqc(self):

    # If radar is 88D convert file to cfRadial to organize split cuts and remove MRLE scans
        if self.radar.metadata['original_container'] == 'NEXRAD Level II':
            self.radar = qc.convert_to_cf(self)
    
    # Rename fields with GPM, 2-letter IDs (e.g. CZ, DR, KD)
    # Save raw reflectivity, will be applied to DZ later
        self.radar, zz = cm.rename_fields_in_radar(self)
        
    # Create a filter to remove data beyond 200km
        if self.radar.metadata['instrument_name'] == 'WSR-88D':
            self.radar = qc.mask_88D_200(self)
    
    # Apply calibration numbers to Reflectivity and ZDR fields
        if self.apply_cal == True:
             self.radar = qc.calibrate(self)    
                        
    # Create a filter to remove data close to radar location (e.g. cone of silence)
        if self.do_cos == True:
            self.radar = qc.mask_cone_of_silence(self)

    # Create mask to filter a range/azm/height sector of radar
        if self.do_sector == True:
            self.radar = qc.sector_wipeout(self)
           
    # Create mask to filter a range/azm/height sector of radar based on RHOHV threshold
        if self.do_rh_sector == True:
            self.radar = qc.rh_sector(self)
                
    # Create Temperature and/or Height field
        if self.use_sounding == True:
            if self.sounding_type == 'ruc':
                self.radar = cm.use_ruc_sounding(self)
            if self.sounding_type == 'uwy':
                self.radar = cm.use_uwy_sounding(self)
            if self.sounding_type == 'ruc_archive':
                self.radar = cm.get_ruc_sounding(self)
        if self.use_sounding == False:
            self.radar = cm.get_beam_height(self)        
        
    # Apply CSU_RT filters
        self.radar = qc.csu_filters(self)
                                                  
    # Apply gatefilters on DP fields
        self.radar = qc.threshold_qc_dpfields(self)
        
    # Get PhiDP
        self.radar = qc.unfold_phidp(self)
        
    # Get KDP and Std(PhiDP)
        self.radar = qc.calculate_kdp(self)
        
    # Create mask to filter a range/azm/height sector of radar based on SD threshold
        if self.do_sd_sector == True:
            self.radar = qc.sd_sector(self)
                  
    # Create mask to filter a range/azm/height sector of radar based on PH threshold
        if self.do_ph_sector == True:
            self.radar = qc.ph_sector(self)
                
    # Perform gatefilters for calculated fields
        self.radar = qc.threshold_qc_calfields(self)
        
    # Dealiasing Velocity
        if self.dealias_velocity == True:
            self.radar = qc.velocity_dealias(self)            
        
    # Add uncorrected reflectiviy field to radar structure.
        self.radar.add_field('DZ', zz, replace_existing=True)
    
    # Remove QC threshold fields from radar
        qc.remove_fields_from_radar(self)       
    
    # Plotting images 
        if self.plot_images == True:   
            pi.plot_fields(self)
    
    # Write cfRadial file
        if self.output_cf == True:
            cm.remove_undesirable_fields(self)
        # Write cfRadial file
            cm.output_cf(self)        
    
        return self.radar   
    
# ***************************************************************************************   

class DP_products:
    '''
    Creates a class DP_products with a constructor and 
    an instance method (run_DP_products).
    '''

    # Constructor:  Specifies necessary initialization parameters
    def __init__(self, file, radar, **kwargs):
        self.file = file
        
        # Uncompress input file, and create radar structure
        cfy = pathlib.Path(file).suffix
        if self.file == 'QC_radar':
            self.radar = radar
        else:
            self.file = file
            #cfy = pathlib.Path(file).suffix
            if cfy == '.gz': 
                self.file = cm.unzip_file(self.file)
                file_unzip = self.file      
                radar = pyart.io.read(self.file, file_field_names=True)
            else:
                radar = pyart.io.read(self.file, file_field_names=True)        
            
            self.radar = radar
        
        # Check and fix missing user defined kwargs
        default_product = dp.get_default_product_dict()
        kwargs = cm.check_kwargs(kwargs, default_product)
        
        # Get site name and date time from radar
        site_time = cm.get_site_date_time(self.radar)
        kwargs = {**site_time, **kwargs}
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        print('', 'DP products parameters:    ', '', kwargs, '', 
              'Processing --> ' + file, sep='\n')    
        
        # Remove temp file
        if cfy == '.gz': os.remove(file_unzip)

    # Instance Method
    def run_DP_products(self):
        
    # Rename fields with GPM, 2-letter IDs (e.g. CZ, DR, KD)
        self.radar, zz = cm.rename_fields_in_radar(self)
        
    # If no KDP, calculate field
        if 'KD' not in self.radar.fields.keys():  
            self.radar = dp.get_kdp(self)
        
        dz = self.radar.fields['CZ']['data']
        dr = self.radar.fields['DR']['data']
        kd = self.radar.fields['KD']['data']
        rh = self.radar.fields['RH']['data']
        
        print('', 'Calculating DP products:  ', sep='\n')
    
    # Calculate Drop-Size Distribution from Tokay et al. 2020 and add to radar
        if self.do_tokay_DSD == True: 
            self.radar = dp.add_calc_dsd_sband_tokay_2020(self.radar, dz, dr, 'wff')
       
    # Create Temperature field   
        if self.use_sounding == True:
            if self.sounding_type == 'ruc':
                self.radar = cm.use_ruc_sounding(self)
            if self.sounding_type == 'uwy':
                self.radar = cm.use_uwy_sounding(self)
            if self.sounding_type == 'ruc_archive':
                self.radar = cm.get_ruc_sounding(self)
                
            radar_T = self.radar.fields['TEMP']['data']
            radar_z = self.radar.fields['HEIGHT']['data']

    # Get HID scores and add FH to radar
            if self.do_HID == True:
                radar_band = 'S'
                self.radar, fh = dp.add_csu_fhc(self.radar, dz, dr, rh, kd, radar_band, radar_T)

    # Calculate Cifelli et al. 2011 ice and water mass fields and add to radar.
    # Function expects, reflectivity, differential reflectivity, 
    # and altitude (km MSL) at a minimum. Temperature is optional.
            if self.do_mass == True:
                self.radar = dp.add_csu_liquid_ice_mass(self.radar, dz, dr, radar_z/1000.0, radar_T)
        
    # Calculate Blended-rain from Cifelli et al. 2011 and addr to radar
            if self.do_RC == True:
                self.radar = dp.add_csu_blended_rain(self.radar, dz, dr, kd, fh)
        
        else:  
            print('', 'Sounding file required to create HID, Ice and Water Mass and RC', '', sep='\n')
        
        print('DP products complete.')
    
    # Plot radar images
        if self.plot_images == True:
            print()
            pi.plot_fields(self)       
        
    # Remove unwanted fields from radar and write cfRadial
        if self.output_cf == True:
            cm.remove_undesirable_fields(self)
        # Write cfRadial file
            cm.output_cf(self)
 
# *******************************************  M  A  I  N  **************************************************

if __name__ == '__main__':
            
    runargs = argparse.ArgumentParser(description='User information')
    runargs.add_argument('file', type=str, help='File to process')
    runargs.add_argument('--thresh_dict', dest='thresh_dict', type=str, help='Threshold dictionary')
    runargs.add_argument('--product_dict', dest='product_dict', type=str, help='DP product dictionary')
    runargs.add_argument('--do_qc', action="store_true", help='Run QC')
    runargs.add_argument('--dp_products', action="store_true", help='Create DP products')

    args = runargs.parse_args() 

    if args.do_qc and args.thresh_dict == None: print('No threshold dictionary, applying default thresholds.')
    if args.dp_products and args.product_dict == None: print('No product dictionary, applying defaults.')
    if args.do_qc == False and args.dp_products == False: sys.exit('Please specify actions: --do_qc --dp_products at least 1 must be selected.')

    file = args.file

    # Import QC thresholds and in out dirs
    if args.thresh_dict == None:
        kwargs = qc.get_default_thresh_dict()
    else:
        current_dict = args.thresh_dict
        kwargs = ast.literal_eval(open(current_dict).read())

    # Import DP products thresholds and in out dirs
    if args.product_dict == None:
        kwargs_product = dp.get_default_product_dict()
    else:
        product_dict = args.product_dict
        kwargs_product = ast.literal_eval(open(product_dict).read())
        
    # If do_qc = True create QC Class with class variables
    if args.do_qc:
        q = QC(file, **kwargs)
        qc_radar = q.run_dpqc()

    # If dp_products = True create  DP_products Class with class variables
    if args.dp_products:
        radar = []
        if args.dp_products and args.do_qc:
            file = 'QC_radar'
            radar = qc_radar
            
        d = DP_products(file, radar, **kwargs_product)
        d.run_DP_products()

print('Done.', '', sep='\n')
