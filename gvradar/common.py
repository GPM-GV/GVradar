# ***************************************************************************************
'''
Functions used by both dpqc.py and dp_products.py.

Developed by the NASA GPM-GV group
V0.5 - 12/06/2021 - update by Jason Pippitt NASA/GSFC/SSAI
V1.0 - 11/01/2022 - update by Jason Pippitt NASA/GSFC/SSAI
V1.5 - 02/02/2024 - update by Jason Pippitt NASA/GSFC/SSAI
'''
# ***************************************************************************************

import numpy as np
import copy
from copy import deepcopy
import pyart
import os, sys, re
import datetime
from cftime import date2num, num2date
import gzip
import shutil
import mmap
import xarray
import pandas as pd
from skewt import SkewT
from collections import defaultdict
import urllib.request
from urllib.request import urlopen
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain,
                            csu_dsd, csu_kdp, csu_misc, fundamentals)

# ***************************************************************************************

def get_ruc_sounding(self):

    """
    Grabs RUC sounding from website for radar lat, lon, and datetime.
    
    Reads text sounding file and creates dictionary for input to SkewT.
    returns:  sounding dictionary

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    
    print('    Retrieving RUC and Interpolating sounding to radar structure...', sep='\n')
    
    RADAR_SITE = (self.site, str(self.radar.latitude['data'][0]), str(self.radar.longitude['data'][0]))
    timestamp = self.year + self.month + self.day + self.hh
    
    # Grab RUC sounding from website
    sound_dict = twister_data(timestamp, RADAR_SITE)
    #sound_dict = retrieveData(timestamp, RADAR_SITE)
    
    # Create data framne from dictionary
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    sound = pd.DataFrame.from_dict(sound_dict)
    
    presssure_pa = sound.PRES.values
    height_m = sound.HGHT.values
    temperature_c = sound.TEMP.values
    dewpoint_c = sound.DWPT.values

    mydata=dict(zip(('hght','pres','temp','dwpt'),(height_m,presssure_pa,temperature_c,dewpoint_c)))
    sounding=SkewT.Sounding(soundingdata=mydata)
   
    radar_T, radar_z = interpolate_sounding_to_radar(sounding, self.radar)
    
    add_field_to_radar_object(radar_T, self.radar, field_name='TEMP', units='deg C',
                                 long_name='Temperature',
                                 standard_name='Temperature',
                                 dz_field=self.ref_field_name)

    add_field_to_radar_object(radar_z, self.radar, field_name='HEIGHT', units='km',
                                 long_name='Height',
                                 standard_name='Height', 
                                 dz_field=self.ref_field_name)

    self.expected_ML = retrieve_ML(mydata)
    print('',sound,'',sep='\n')

    return self.radar

# ***************************************************************************************
def get_ruc_archive(self):

    """
    Finds correct RUC hourly sounding based on radar time stamp, from RUC sounding archive.
    
    Reads text sounding file and creates dictionary for input to SkewT.
    returns:  sounding dictionary

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    
    print('    Interpolating sounding to radar structure...', sep='\n')
    
    snd_dir = self.sounding_dir

    # Retrieve proper sounding for date and time    
    radar_DT = pyart.util.datetime_from_radar(self.radar)   
    hour =  radar_DT.hour
    month = self.month
    day = self.day
    year = self.year
    hh = self.hh
    mm = self.mm
    mdays = [00,31,28,31,30,31,30,31,31,30,31,30,31]
                
    if radar_DT.minute >= 30: hour = radar_DT.hour + 1
    if hour == 24: 
        mday = radar_DT.day + 1
        hour = 0
        if mday > mdays[radar_DT.month]:
            cmonth = radar_DT.month + 1
            mday = 1
            if(cmonth > 12):
                cmonth = 1
                mday = 1
                cyear = radar_DT.year + 1
                year = str(cyear).zfill(4)
            month = str(cmonth).zfill(2)
        day = str(mday).zfill(2)
    hh = str(hour).zfill(2)
    if self.site == 'XPOL':
        sounding_dir = snd_dir + year + '/' + month + day + '/TM1/TM1_' + year + '_' + month + day + '_' + hh + 'UTC.txt'
    elif self.site == 'KuD3R' or self.site == 'KaD3R':
        sounding_dir = snd_dir + year + '/' + month + day + '/NPOL/NPOL_' + year + '_' + month + day + '_' + hh + 'UTC.txt'
    else:
        sounding_dir = snd_dir + year + '/' + month + day + '/' + self.site + '/' + self.site + '_' + year + '_' + month + day + '_' + hh + 'UTC.txt'
    
    soundingb = os.path.basename(sounding_dir)
    print('    Sounding file -->  ' + soundingb, sep='\n')

    headings = ["PRES","HGHT","TEMP","DWPT","RELH","MIXR","DRCT","SKNT","THTA","THTE","THTV"]

    try:
        #colspecs = [(3, 9), (11, 18), (20, 26), (28, 34), (36, 38), (40, 42),
        #            (44, 46), (48, 50), (52, 54), (56, 58), (60, 62)]
    
        #sound = pd.read_fwf(sounding_dir, names=headings, header=None, colspecs=colspecs,skiprows=2)
        sound = pd.read_csv(sounding_dir, sep='\\s+', encoding = "ISO-8859-1",skiprows = [1])
        presssure_pa = sound.PRES
        height_m = sound.HGHT
        temperature_c = sound.TEMP
        dewpoint_c = sound.DWPT

        mydata=dict(zip(('hght','pres','temp','dwpt'),(height_m,presssure_pa,temperature_c,dewpoint_c)))

        sounding=SkewT.Sounding(soundingdata=mydata)

        radar_T, radar_z = interpolate_sounding_to_radar(sounding, self.radar)
    except:
        colspecs = [(3, 10), (11, 22), (23, 32), (33, 42), (42, 52), (52, 62),
                    (62, 72), (72, 82), (82, 92), (92, 102), (102, 112)]
    
        sound = pd.read_fwf(sounding_dir, names=headings, header=None, colspecs=colspecs,skiprows=2)
        
        presssure_pa = sound.PRES
        height_m = sound.HGHT
        temperature_c = sound.TEMP
        dewpoint_c = sound.DWPT

        mydata=dict(zip(('hght','pres','temp','dwpt'),(height_m,presssure_pa,temperature_c,dewpoint_c)))

        sounding=SkewT.Sounding(soundingdata=mydata)
           
        radar_T, radar_z = interpolate_sounding_to_radar(sounding, self.radar)
    
    add_field_to_radar_object(radar_T, self.radar, field_name='TEMP', units='deg C',
                                 long_name='Temperature',
                                 standard_name='Temperature',
                                 dz_field=self.ref_field_name)

    add_field_to_radar_object(radar_z, self.radar, field_name='HEIGHT', units='km',
                                 long_name='Height',
                                 standard_name='Height', 
                                 dz_field=self.ref_field_name)

    self.expected_ML = retrieve_ML(mydata)
    print('',sound,'',sep='\n')

    return self.radar

# ***************************************************************************************

def kwaj_sounding(self):

    """
    Imports KWAJ sounding data into skewT

    Format:

    PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV
    hPa     m      C      C      %    g/kg    deg   knot     K      K      K
   1010.7      4.0   27.70   21.00 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    977.0    304.8   24.50   19.50 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    943.6    609.6   21.60   19.00 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    911.1    914.4   19.70   15.70 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    879.5   1219.2   18.00   13.00 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    848.8   1524.0   17.70    9.60 0.0 0.0 0.0 0.0 0.0 0.0 0.0

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    print('    Interpolating KWAJ sounding to radar structure...', sep='\n')

    snd_dir = self.sounding_dir

    # Retrieve proper sounding for date and time
    radar_DT = pyart.util.datetime_from_radar(self.radar)
    month = self.month
    day = self.day
    year = self.year

    sounding_dir = snd_dir + year + '/' + month + day + '/' + self.site + '/' + self.site + '_' + year + '_' + month + day + '_00UTC.txt'

    sounding_dir = sounding_dir
    soundingb = os.path.basename(sounding_dir)

    print('    Sounding file -->  ' + soundingb, sep='\n')

    headings = ["PRES","HGHT","TEMP","DWPT","RELH","MIXR","DRCT","SKNT","THTA","THTE","THTV"]
    colspecs = [(3, 9), (11, 18), (20, 26), (28, 34), (36, 38), (40, 42),
                (44, 46), (48, 50), (52, 54), (56, 58), (60, 62)]

    sound = pd.read_fwf(sounding_dir, names=headings, header=None, colspecs=colspecs,skiprows=2)

    presssure_pa = sound.PRES
    height_m = sound.HGHT
    temperature_c = sound.TEMP
    dewpoint_c = sound.DWPT

    mydata=dict(zip(('hght','pres','temp','dwpt'),(height_m,presssure_pa,temperature_c,dewpoint_c)))

    sounding=SkewT.Sounding(soundingdata=mydata)

    radar_T, radar_z = interpolate_sounding_to_radar(sounding, self.radar)

    add_field_to_radar_object(radar_T, self.radar, field_name='TEMP', units='deg C',
                                 long_name='Temperature',
                                 standard_name='Temperature',
                                 dz_field=self.ref_field_name)

    add_field_to_radar_object(radar_z, self.radar, field_name='HEIGHT', units='km',
                                 long_name='Height',
                                 standard_name='Height',
                                 dz_field=self.ref_field_name)

    self.expected_ML = retrieve_ML(mydata)
    print('',sound,'',sep='\n')

    return self.radar

# ***************************************************************************************                
def use_ruc_sounding(self):

    """
    Imports RUC sounding data into skewT
    
    Soundings can be downloaded from following website:  
    https://rucsoundings.noaa.gov/

    Format:
     PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV
     hPa     m      C      C      %    g/kg    deg   knot     K      K      K
       1019.4     19.0   29.00   24.60 0.0 0.0 0.0 0.0 0.0 0.0 0.0
       1016.3     49.0   28.70   24.30 0.0 0.0 0.0 0.0 0.0 0.0 0.0
       1010.7    102.0   28.20   24.10 0.0 0.0 0.0 0.0 0.0 0.0 0.0
       1000.9    190.0   27.20   23.90 0.0 0.0 0.0 0.0 0.0 0.0 0.0
       1000.0    197.0   27.10   23.90 0.0 0.0 0.0 0.0 0.0 0.0 0.0   
        986.6    317.0   26.00   23.60 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        968.1    483.0   24.30   23.00 0.0 0.0 0.0 0.0 0.0 0.0 0.0

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sounding_dir = self.sounding_dir
    soundingb = os.path.basename(sounding_dir)
    
    print('    RUC Sounding file -->  ' + soundingb, sep='\n')

    headings = ["PRES","HGHT","TEMP","DWPT","RELH","MIXR","DRCT","SKNT","THTA","THTE","THTV"]
    
    try:
        colspecs = [(3, 9), (11, 18), (20, 26), (28, 34), (36, 38), (40, 42),
                    (44, 46), (48, 50), (52, 54), (56, 58), (60, 62)]
    
        sound = pd.read_fwf(sounding_dir, names=headings, header=None, colspecs=colspecs,skiprows=2)

        presssure_pa = sound.PRES
        height_m = sound.HGHT
        temperature_c = sound.TEMP
        dewpoint_c = sound.DWPT

        mydata=dict(zip(('hght','pres','temp','dwpt'),(height_m,presssure_pa,temperature_c,dewpoint_c)))

        sounding=SkewT.Sounding(soundingdata=mydata)
    except:
        colspecs = [(3, 9), (11, 18), (20, 26), (28, 34), (36, 38), (40, 42),
                    (44, 46), (48, 50), (52, 54), (56, 58), (60, 62)]
    
        sound = pd.read_fwf(sounding_dir, names=headings, header=None, colspecs=colspecs,skiprows=2)
        
        presssure_pa = sound.PRES
        height_m = sound.HGHT
        temperature_c = sound.TEMP
        dewpoint_c = sound.DWPT

        mydata=dict(zip(('hght','pres','temp','dwpt'),(height_m,presssure_pa,temperature_c,dewpoint_c)))

        sounding=SkewT.Sounding(soundingdata=mydata)
           
    radar_T, radar_z = interpolate_sounding_to_radar(sounding, self.radar)
    
    add_field_to_radar_object(radar_T, self.radar, field_name='TEMP', units='deg C',
                                 long_name='Temperature',
                                 standard_name='Temperature',
                                 dz_field=self.ref_field_name)
   
    add_field_to_radar_object(radar_z, self.radar, field_name='HEIGHT', units='km',
                                 long_name='Height',
                                 standard_name='Height', 
                                 dz_field=self.ref_field_name)

    self.expected_ML = retrieve_ML(mydata)
    print('',sound,'',sep='\n')

    return self.radar

# ***************************************************************************************
def get_uwy_archive(self):
    
    """
    Imports UWY sounding data into skewT
    
    Soundings can be downloaded from following website:  
    http://weather.uwyo.edu/upperair/sounding.html

    Format:
    94975 YMHB Hobart Airport Observations at 00Z 02 Jul 2013

    -----------------------------------------------------------------------------
       PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV
       hPa     m      C      C      %    g/kg    deg   knot     K      K      K
    -----------------------------------------------------------------------------
     1004.0     27   12.0   10.2     89   7.84    330     14  284.8  306.7  286.2
     1000.0     56   12.4   10.3     87   7.92    325     16  285.6  307.8  286.9
      993.0    115   12.8    9.7     81   7.66    311     22  286.5  308.1  287.9    

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    snd_dir = self.sounding_dir

    # Retrieve proper sounding for date and time    
    radar_DT = pyart.util.datetime_from_radar(self.radar)   
    hour =  radar_DT.hour
    month = self.month
    day = self.day
    year = self.year
    hh = self.hh
    mm = self.mm
    mdays = [00,31,28,31,30,31,30,31,31,30,31,30,31]

    if self.site == 'CPOL':
        if radar_DT.hour <= 6:
            sounding_dir = snd_dir + year + '/' + month + '/DARW/DARW_YPDN_' + year + '_' + month + day + '_00UTC.txt'
        if radar_DT.hour > 6 and radar_DT.hour <= 18:
            sounding_dir = snd_dir + year + '/' + month + '/DARW/DARW_YPDN_' + year + '_' + month + day + '_12UTC.txt'
        if radar_DT.hour > 18:
            mday = radar_DT.day + 1
            day = str(mday).zfill(2)
            if mday > mdays[radar_DT.month]:
                cmonth = radar_DT.month + 1
                mday = 1
                if(cmonth > 12):
                    cmonth = 1
                    mday = 1
                    cyear = radar_DT.year + 1
                    year = str(cyear).zfill(4)
                month = str(cmonth).zfill(2)
                day = str(mday).zfill(2)
            sounding_dir = snd_dir + year + '/' + month + '/DARW/DARW_YPDN_' + year + '_' + month + day + '_00UTC.txt'
    elif self.site == 'Reunion':
        sounding_dir = snd_dir + year + '/' + month + '/Reunion/Reunion_FMEE_' + year + '_' + month + day + '_12UTC.txt'
    elif self.site == 'CP2':
        if radar_DT.hour <= 6:
            sounding_dir = snd_dir + year + '/' + month + '/CP2/CP2_YBBN_' + year + '_' + month + day + '_00UTC.txt'
        if radar_DT.hour > 6 and radar_DT.hour <= 18:
            sounding_dir = snd_dir + year + '/' + month + '/CP2/CP2_YBBN_' + year + '_' + month + day + '_12UTC.txt'
        if radar_DT.hour > 18:
            mday = radar_DT.day + 1
            day = str(mday).zfill(2)
            if mday > mdays[radar_DT.month]:
                cmonth = radar_DT.month + 1
                mday = 1
                if(cmonth > 12):
                    cmonth = 1
                    mday = 1
                    cyear = radar_DT.year + 1
                    year = str(cyear).zfill(4)
                month = str(cmonth).zfill(2)
                day = str(mday).zfill(2)
            sounding_dir = snd_dir + year + '/' + month + '/CP2/CP2_YBBN_' + year + '_' + month + day + '_00UTC.txt'        
        
    else:
        if radar_DT.minute >= 30: hour = radar_DT.hour + 1
        if hour == 24: 
            mday = radar_DT.day + 1
            hour = 0
            if mday > mdays[radar_DT.month]:
                cmonth = radar_DT.month + 1
                mday = 1
                if(cmonth > 12):
                    cmonth = 1
                    mday = 1
                    cyear = radar_DT.year + 1
                    year = str(cyear).zfill(4)
                month = str(cmonth).zfill(2)
            day = str(mday).zfill(2)
        hh = str(hour).zfill(2)
        sounding_dir = snd_dir + year + '/' + month + day + '/' + self.site + '/' + self.site + '_' + year + '_' + month + day + '_' + hh + 'UTC.txt'
    
    soundingb = os.path.basename(sounding_dir)
    
    print('    UWY Sounding file -->  ' + soundingb, sep='\n')

    headings = ["PRES","HGHT","TEMP","DWPT","RELH","MIXR","DRCT","SKNT","THTA","THTE","THTV"]

    try:
        colspecs = [(1, 9), (9, 15), (16, 22), (23, 30), (31, 37), (37, 42),
                (43, 50), (50, 57), (57, 63), (63, 70), (70, 78)]
        sound = pd.read_fwf(sounding_dir, names=headings, header=None, colspecs=colspecs,skiprows=2)

        presssure_pa = sound.PRES
        height_m = sound.HGHT
        temperature_c = sound.TEMP
        dewpoint_c = sound.DWPT

        mydata=dict(zip(('hght','pres','temp','dwpt'),(height_m,presssure_pa,temperature_c,dewpoint_c)))

        sounding=SkewT.Sounding(soundingdata=mydata)
           
        radar_T, radar_z = interpolate_sounding_to_radar(sounding, self.radar)
    except:
        colspecs = [(1, 9), (9, 15), (16, 22), (23, 30), (31, 37), (37, 42),
                (43, 50), (50, 57), (57, 63), (63, 70), (70, 78)]
        sound = pd.read_fwf(sounding_dir, names=headings, header=None, colspecs=colspecs,skiprows=3)

        presssure_pa = sound.PRES
        height_m = sound.HGHT
        temperature_c = sound.TEMP
        dewpoint_c = sound.DWPT

        mydata=dict(zip(('hght','pres','temp','dwpt'),(height_m,presssure_pa,temperature_c,dewpoint_c)))

        sounding=SkewT.Sounding(soundingdata=mydata)
           
        radar_T, radar_z = interpolate_sounding_to_radar(sounding, self.radar)

    add_field_to_radar_object(radar_T, self.radar, field_name='TEMP', units='deg C',
                                 long_name='Temperature',
                                 standard_name='Temperature',
                                 dz_field=self.ref_field_name)
   
    add_field_to_radar_object(radar_z, self.radar, field_name='HEIGHT', units='km',
                                 long_name='Height',
                                 standard_name='Height', 
                                 dz_field=self.ref_field_name)

    self.expected_ML = retrieve_ML(mydata)
    print('',sound,'',sep='\n')

    return self.radar

# ***************************************************************************************

def rename_fields_in_radar(self):

    """
    Rename fields we want to keep with GPM, 2-letter IDs (e.g. CZ, DR, KD)
    Written by: David B. Wolff, NASA/WFF

    Parameters:
    -----------
    radar: pyart radar object
    old_fields: List of current field names that we want to change
    new_fields: List of field names we want to change the name to

    Return:
    -------
    radar: radar with more succinct field names

    """    
    print('', "Renaming radar fields...", sep='\n')
    print(self.radar.fields.keys())
    if 'PHIDP2' in self.radar.fields.keys():
        old_fields = ['DBZ2', 'VEL2', 'WIDTH2', 'ZDR2', 'PHIDP2', 'RHOHV2', 'SQI2']
        new_fields = ['DZ',     'VR',    'SW',   'DR', 'PH',     'RH',     'SQ']
    if 'KDP2' in self.radar.fields.keys():
        old_fields = ['DBZ2', 'VEL2', 'WIDTH2', 'ZDR2', 'KDP2', 'PHIDP2', 'RHOHV2', 'SQI2']
        new_fields = ['DZ',     'VR',    'SW',   'DR',   'KD',   'PH',     'RH',     'SQ']    
    elif 'DBZ' in self.radar.fields.keys():
        old_fields = ['DBZ', 'VEL', 'WIDTH', 'ZDR', 'KDP', 'PHIDP', 'RHOHV']
        new_fields = ['DZ',  'VR',   'SW',   'DR',  'KD',   'PH',     'RH']
    elif 'SQI' in self.radar.fields.keys():
        old_fields = ['DBZ', 'VEL', 'WIDTH', 'ZDR', 'KDP', 'PHIDP', 'SQI', 'RHOHV']
        new_fields = ['DZ',  'VR',   'SW',   'DR',  'KD',   'PH',    'SQ',    'RH']
    elif 'MaskSecondTrip' in self.radar.fields.keys():
        old_fields = ['ReflectivityHV', 'Velocity', 'SpectralWidth', 
                      'DifferentialReflectivity', 'DifferentialPhase', 
                      'CopolarCorrelation']
        new_fields = ['DZ', 'VR', 'SW', 'DR', 'PH', 'RH']
    elif 'reflectivity' in self.radar.fields.keys():
        old_fields = ['differential_phase', 'velocity', 'spectrum_width', 'reflectivity', 'differential_reflectivity', 'cross_correlation_ratio']
        new_fields = ['PH',     'VR',     'SW',   'DZ',   'DR', 'RH']
    elif 'REF' in self.radar.fields.keys():
        old_fields = ['SW', 'PHI', 'ZDR', 'REF', 'VEL', 'RHO']
        new_fields = ['SW', 'PH' , 'DR' , 'DZ' , 'VR' , 'RH' ]
    elif 'radar_echo_classification' in self.radar.fields.keys():
        old_fields = ['radar_echo_classification', 'radar_estimated_rain_rate', 'D0', 'NW', 'velocity', 
         'corrected_velocity', 'total_power', 'corrected_reflectivity', 'cross_correlation_ratio', 
         'differential_reflectivity', 'corrected_differential_reflectivity', 'differential_phase', 
         'corrected_differential_phase', 'corrected_specific_differential_phase', 'spectrum_width', 
         'signal_to_noise_ratio']
        new_fields = ['radar_echo_classification', 'RR', 'D0', 'NW', 'velocity', 
         'VR', 'total_power', 'CZ', 'RH', 
         'differential_reflectivity', 'DR', 'differential_phase', 
         'PH', 'corrected_specific_differential_phase', 'SW', 
         'signal_to_noise_ratio']
    elif 'ZC' in self.radar.fields.keys():
        old_fields = ['SN', 'ZH', 'ZD', 'RH', 'PH', 'KD', 'VE', 'ZC', 'ZB', 'DO', 'NW', 'RR', 'CM', 'HC']
        new_fields = ['SN', 'DZ', 'ZDR', 'RH', 'PH', 'KD', 'VR', 'CZ', 'DR', 'DO', 'NW', 'RR', 'CM', 'HC']
    elif 'DBZH' in self.radar.fields.keys():
        old_fields = ['DBZH', 'TH', 'RHOHV', 'UPHIDP', 'WRADH', 'PHIDP', 'ZDR', 'KDP', 'SQIH', 'VRADH']
        new_fields = ['DZ',   'TH', 'RH',    'UPHIDP', 'WRADH', 'PH',    'DR',  'KD',  'SQ', 'VR'] 
    elif 'ZZ' in self.radar.fields.keys():
        old_fields = ['ZZ']
        new_fields = ['DZ']    
    elif 'DZ' in self.radar.fields.keys():
        old_fields = []
        new_fields = []
    elif 'CZ' in self.radar.fields.keys():
        old_fields = []
        new_fields = []

    # Change names of old fields to new fields using pop
    nl = len(old_fields)
    for i in range(0,nl):
        old_field = old_fields[i]
        new_field = new_fields[i]
        self.radar.fields[new_field] = self.radar.fields.pop(old_field)
        i += 1  

    # Add Corrected Reflectivity field
    if ('CZ') not in self.radar.fields.keys():
#        if self.site == 'NPOL' or self.site == 'KWAJ':
        if self.site == 'KWAJ':
            zz = deepcopy(self.radar.fields['DBT2'])
            cz = self.radar.fields['DBT2']['data'].copy()
            self.radar = add_field_to_radar_object(cz, self.radar, field_name='CZ', 
                                         units='dBZ',
                                         long_name='Corrected Reflectivity', 
                                         standard_name='Corrected Reflectivity', 
                                         dz_field='DZ')
        elif self.site == 'KaD3R' or self.site == 'KuD3R':
            zz = deepcopy(self.radar.fields['DZ'])
            cz_dict = {'data': zz['data'], 'units': '', 'long_name': 'CZ',
                       '_FillValue': -32767.0, 'standard_name': 'CZ'}
            self.radar.add_field('CZ', cz_dict, replace_existing=True)
        else: 
            zz = deepcopy(self.radar.fields['DZ'])
            cz = self.radar.fields['DZ']['data'].copy()
            self.radar = add_field_to_radar_object(cz, self.radar, field_name='CZ', 
                                         units='dBZ',
                                         long_name='Corrected Reflectivity', 
                                         standard_name='Corrected Reflectivity', 
                                         dz_field='DZ')
         
        print(self.radar.fields.keys(), '', sep='\n')
        return self.radar, zz
    else:
        if self.site == 'CPOL':
            zz = deepcopy(self.radar.fields['CZ'])
        else:
            zz = deepcopy(self.radar.fields['DZ'])
        
        print(self.radar.fields.keys(), '', sep='\n')
        return self.radar, zz

# ***************************************************************************************
 
def remove_undesirable_fields(self):

    print("Removing unwanted output fields...", '', sep='\n')

    if self.site == 'CPOL':
        cf_fields = ['radar_echo_classification', 'RR', 'DM',
                     'D0', 'NW', 'velocity', 'VR', 'total_power', 'CZ', 'RH', 
                     'differential_reflectivity', 'DR', 'differential_phase', 
                     'PH', 'corrected_specific_differential_phase', 'SW', 
                     'signal_to_noise_ratio', 'KD', 'FH', 'MW', 'MI', 'RC']
    elif self.site == 'XPOL':
        cf_fields = ['DZ', 'DR', 'PH', 'RH', 'VR', 'SW', 'SD', 'KD', 
                     'CZ', 'FH', 'MW', 'MI', 'RC']
    else:
        cf_fields = self.output_fields
        
    drop_fields = [i for i in self.radar.fields.keys() if i not in cf_fields]
    for field in drop_fields:
        self.radar.fields.pop(field)

    print("CF FIELDS -->  ", self.radar.fields.keys(), '', sep='\n')
  
    return self.radar

# ***************************************************************************************

def update_metadata(self):

    self.radar.metadata['version'] = 'VN V2.1'
    self.radar.metadata['Description'] = "Quality Controlled Dual Pol Products"
    self.radar.metadata['Release_Date'] = str(datetime.datetime.utcnow())
    self.radar.metadata['institution'] = "NASA GSFC"
    self.radar.metadata['project'] = "Global Precipitation Measurement (GPM)"
    self.radar.metadata['source'] = 'GVradar V1.1'
    self.radar.metadata['references'] = 'https://github.com/GPM-GV/GVradar'

    new_fields = []
    for key in self.radar.fields.keys():
            new_fields.append(key)

    self.radar.metadata['field_names'] = new_fields

    if 'CZ' in self.radar.fields:
        cz = self.radar.fields['CZ']['data'].copy()
        self.radar = add_field_to_radar_object(cz, self.radar, field_name='CZ', 
                                         units='dBZ',
                                         long_name='Corrected Reflectivity', 
                                         standard_name='Corrected Reflectivity', 
                                         dz_field='CZ')

    if 'DZ' in self.radar.fields:
        dz = self.radar.fields['DZ']['data'].copy()
        self.radar = add_field_to_radar_object(dz, self.radar, field_name='DZ', 
                                         units='dBZ',
                                         long_name='RAW Reflectivity', 
                                         standard_name='RAW Reflectivity', 
                                         dz_field='CZ')                                     

    if 'DR' in self.radar.fields:
        dr = self.radar.fields['DR']['data'].copy()
        self.radar = add_field_to_radar_object(dr, self.radar, field_name='DR', 
                                         units='dB',
                                         long_name='Differential Reflectivity', 
                                         standard_name='log_differential_reflectivity_hv', 
                                         dz_field='CZ')     

    if 'RH' in self.radar.fields:
        rh = self.radar.fields['RH']['data'].copy()
        self.radar = add_field_to_radar_object(rh, self.radar, field_name='RH', 
                                         units='none',
                                         long_name='Correlation Coefficient', 
                                         standard_name='cross_correlation_ratio_hv', 
                                         dz_field='CZ')      

    if 'VR' in self.radar.fields:
        vr = self.radar.fields['VR']['data'].copy()
        self.radar = add_field_to_radar_object(vr, self.radar, field_name='VR', 
                                         units='m/s',
                                         long_name='Radial Velocity', 
                                         standard_name='radial_velocity_of_scatterers_away_from_instrument', 
                                         dz_field='CZ')  

    if 'SW' in self.radar.fields:
        sw = self.radar.fields['SW']['data'].copy()
        self.radar = add_field_to_radar_object(sw, self.radar, field_name='SW', 
                                         units='m/s',
                                         long_name='Spectrum Width', 
                                         standard_name='doppler_spectrum_width', 
                                         dz_field='CZ')                                                                     

    #print('', self.radar.metadata, '',sep='\n') 

    return self.radar

# ***************************************************************************************

def output_cf(self):

    # Outputs CF radial file
    # Declare output dir

    out_dir = self.cf_dir
    os.makedirs(out_dir, exist_ok=True)
    if self.site == 'NPOL': self.site = 'NPOL1'

    if self.scan_type == 'RHI':
        out_file = out_dir + '/' + self.site + '_' + self.year + '_' + self.month + self.day + '_' + self.hh + self.mm + self.ss + '_rhi.cf'
    else: 
        out_file = out_dir + '/' + self.site + '_' + self.year + '_' + self.month + self.day + '_' + self.hh + self.mm + self.ss + '.cf'
    
    self.radar = update_metadata(self)

    pyart.io.write_cfradial(out_file,self.radar)
    
    #Gzip cf file
    with open(out_file, 'rb') as f_in:
        with gzip.open(out_file + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(out_file)
    
    gz_file = out_file + '.gz'
    print('Output cfRadial --> ' + gz_file, '', sep='\n')

# ***************************************************************************************       

def output_grid(self):

    # Outputs xarray gridded file

    out_dir = self.grid_dir
    os.makedirs(out_dir, exist_ok=True)
   
    out_file = out_dir + '/' + self.site + '_' + self.year + '_' + self.month + self.day + '_' + self.hh + self.mm + self.ss + '_' + self.scan_type + '.nc'
    
    radar_lat = self.radar.latitude['data'][0]
    radar_lon = self.radar.longitude['data'][0]

    Grid = pyart.map.grid_from_radars((self.radar,),
           grid_shape=(16, 251, 251),
           weighting_function='Barnes2',
           grid_origin=[radar_lat, radar_lon],
           grid_limits=((500, 16000), (-125000, 125000), (-125000, 125000)),
           fields=self.output_fields,
           gridding_algo="map_gates_to_grid")

    xradar = Grid.to_xarray()

    xarray.Dataset.to_netcdf(xradar, path=out_file)
    
    # Gzip cf file
    with open(out_file, 'rb') as f_in:
        with gzip.open(out_file + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(out_file)
    
    gz_file = out_file + '.gz'
    print('Output xarray grid --> ' + gz_file, '', sep='\n')

# ***************************************************************************************
    
def unzip_file(file):
    
    # Unzips input file
    file_unzip = os.path.basename(file)[0:-3]
    with gzip.open(file, 'rb') as f_in:
        with open(file_unzip, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return file_unzip      
    
 # ***************************************************************************************   

def check_kwargs(kwargs, default_kw):
    """
    Check user-provided kwargs against defaults, and if some defaults aren't
    provided by user make sure they are provided to the function regardless.
    """
    for key in default_kw:
        if key not in kwargs:
            kwargs[key] = default_kw[key]
    return kwargs

# ***************************************************************************************

def get_site_date_time(radar):

# Get date/time, site, and scan type from radar.  Retrieve universal field names.

    if 'site_name' in radar.metadata.keys():
        site = radar.metadata['site_name'].upper()
    elif 'instrument_name' in radar.metadata.keys():
        if isinstance(radar.metadata['instrument_name'], bytes):
            site = radar.metadata['instrument_name'].decode().upper()
        else:
            site = radar.metadata['instrument_name'].upper()
    else:
        site=''

    if 'original_container' in radar.metadata.keys():
        if radar.metadata['original_container'] == 'odim_h5':
            try:
                site = radar.metadata['source'].replace(',', ':').split(':')[1].upper()
            except:
                site = radar.metadata['site_name'].upper()

    if site == 'NPOL1' or site == 'NPOL2': site = 'NPOL'         
    if site == 'LAVA1': site = 'KWAJ'
    if site == b'AN1-P\x00\x00\x00' or site == 'AN1-P': site = 'AL1'
    if site == b'JG1-P\x00\x00\x00' or site == b'JG1\x00\x00\x00\x00\x00' or site == 'JG1-P': site = 'JG1'
    if site == b'MC1-P\x00\x00\x00' or site == 'MC1-P' or site == b'MC1\x00\x00\x00\x00\x00': site = 'MC1' 
    if site == b'NT1-P\x00\x00\x00' or site == 'NT1-P': site = 'NT1'
    if site == b'PE1-P\x00\x00\x00' or site == 'PE1-P': site = 'PE1'
    if site == b'SF1-P\x00\x00\x00' or site == 'SF1-P': site = 'SF1'
    if site == b'ST1\x00\x00\x00\x00\x00' or site == b'ST1-P\x00\x00\x00' or site == 'ST1-P': site = 'ST1'
    if site == b'SV1\x00\x00\x00\x00\x00' or site == b'SV1-P\x00\x00\x00' or site == 'SV1-P': site = 'SV1'
    if site == b'TM1-P\x00\x00\x00' or site == 'TM1-P': site = 'TM1'
    if site == 'GUNN_PT': site = 'CPOL'
    if site == 'REUNION': site = 'Reunion'
    if site == 'CP2RADAR': site = 'CP2'
    if 'system' in radar.metadata.keys():
        if radar.metadata['system'] == 'KuD3R': site = 'KuD3R'
        if radar.metadata['system'] == 'KaD3R': site = 'KaD3R'

    radar.metadata['site_name'] = site
    radar.metadata['instrument_name'] = site

    if 'original_container' not in radar.metadata.keys():
        radar.metadata['original_container'] = site
    
    #if site == 'KuD3R' or site == 'KaD3R':
    #    if len(radar.fixed_angle['data'][:]) == 5:
    #            radar.scan_type = 'rhi'

    scan_type = radar.scan_type.upper()
    
    radar_DT = pyart.util.datetime_from_radar(radar)

    if radar_DT.year > 2000:
        if site == 'NPOL' or site == 'KWAJ':
            EPOCH_UNITS = "seconds since 1970-01-01T00:00:00Z"
            dtrad = num2date(0, radar.time["units"])
            epnum = date2num(dtrad, EPOCH_UNITS)
            kwargs = {}
            radar_DT = num2date(epnum, EPOCH_UNITS, **kwargs)
        else:
            radar_DT = pyart.util.datetime_from_radar(radar)
    else:
        if site == 'Reunion':
            EPOCH_UNITS = "seconds since 1970-01-01T00:00:00Z"
            dtrad = num2date(0, radar.time["units"])
            epnum = date2num(dtrad, EPOCH_UNITS)
            kwargs = {}
            radar_DT = num2date(epnum, EPOCH_UNITS, **kwargs)

    month = str(radar_DT.month).zfill(2)
    day = str(radar_DT.day).zfill(2)
    year = str(radar_DT.year).zfill(4)
    hh = str(radar_DT.hour).zfill(2)
    mm = str(radar_DT.minute).zfill(2)
    ss = str(radar_DT.second).zfill(2)

    if site == 'CPOL':
        band = 'C'
    elif site == 'XPOL':
        band = 'X'
    else:
        band = 'S'

    site_time = {'site': site, 'scan_type': scan_type, 'month': month, 'day': day,
                 'year': year, 'hh': hh, 'mm': mm, 'ss': ss, 'ref_field_name': 'CZ',
                 'phi_field_name': 'PH', 'zdr_field_name': 'DR', 'radar_band': band}

    return site_time

# ***************************************************************************************

def extract_unmasked_data(radar, field, bad=-32767.0):
    return radar.fields[field]['data'].filled(fill_value=bad)

# ***************************************************************************************

def add_field_to_radar_object(field, radar, field_name='UN', units='',
                              long_name='UNKNOWN', standard_name='UNKNOWN',
                              dz_field='CZ'):
    """
    From CSU_RadarTools (thanks T. Lang)
    Adds a newly created field to the Py-ART radar object. If reflectivity is a 
    masked array, make the new field masked the same as reflectivity.
    """
    fill_value = -32767.0
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value,
    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        setattr(masked_field, 'mask',
                np.logical_or(masked_field.mask, radar.fields[dz_field]['data'].mask))
        fill_value = radar.fields[dz_field]['_FillValue']
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar

# ***************************************************************************************

def radar_coords_to_cart(rng, az, ele, debug=False):
    """
    TJL - taken from old Py-ART version
    Calculate Cartesian coordinate from radar coordinates
    Parameters
    ----------
    rng : array
    Distances to the center of the radar gates (bins) in kilometers.
    az : array
    Azimuth angle of the radar in degrees.
    ele : array
    Elevation angle of the radar in degrees.
    Returns
    -------
    x, y, z : array
    Cartesian coordinates in meters from the radar.
    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnic [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).
    .. math::
    z = \\sqrt{r^2+R^2+r*R*sin(\\theta_e)} - R
    s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
    x = s * sin(\\theta_a)
    y = s * cos(\\theta_a)
    Where r is the distance from the radar to the center of the gate,
    :math:\\theta_a is the azimuth angle, :math:\\theta_e is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).
    References
    ----------
    .. [1] Doviak and Zrnic, Doppler Radar and Weather Observations, Second
    Edition, 1993, p. 21.
    """
    theta_e = ele * np.pi / 180.0  # elevation angle in radians.
    theta_a = az * np.pi / 180.0  # azimuth angle in radians.
    R = 6371.0 * 1000.0 * 4.0 / 3.0  # effective radius of earth in meters.
    r = rng * 1000.0  # distances to gates in meters.

    z = (r ** 2 + R ** 2 + 2.0 * r * R * np.sin(theta_e)) ** 0.5 - R
    s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    x = s * np.sin(theta_a)
    y = s * np.cos(theta_a)
    return x, y, z

# ***************************************************************************************

def get_z_from_radar(radar):
    """Input radar object, return z from radar (km, 2D)"""
    azimuth_1D = radar.azimuth['data']
    elevation_1D = radar.elevation['data']
    srange_1D = radar.range['data']
    sr_2d, az_2d = np.meshgrid(srange_1D, azimuth_1D)
    el_2d = np.meshgrid(srange_1D, elevation_1D)[1]
    xx, yy, zz = radar_coords_to_cart(sr_2d/1000.0, az_2d, el_2d)
    return zz + radar.altitude['data']

# ***************************************************************************************

def check_sounding_for_montonic(sounding):
    """
    So the sounding interpolation doesn't fail, force the sounding to behave
    monotonically so that z always increases. This eliminates data from
    descending balloons.
    """
    snd_T = sounding.soundingdata['temp']  # In old SkewT, was sounding.data
    snd_z = sounding.soundingdata['hght']  # In old SkewT, was sounding.data
    dummy_z = []
    dummy_T = []
    if not snd_T.mask[0]: #May cause issue for specific soundings
        dummy_z.append(snd_z[0])
        dummy_T.append(snd_T[0])
        for i, height in enumerate(snd_z):
            if i > 0:
                if snd_z[i] > snd_z[i-1] and not snd_T.mask[i]:
                    dummy_z.append(snd_z[i])
                    dummy_T.append(snd_T[i])
        snd_z = np.array(dummy_z)
        snd_T = np.array(dummy_T)
    return snd_T, snd_z

# ***************************************************************************************

def interpolate_sounding_to_radar(sounding, radar):
    """Takes sounding data and interpolates it to every radar gate."""
    radar_z = get_z_from_radar(radar)
    radar_T = None
    snd_T, snd_z = check_sounding_for_montonic(sounding)
    shape = np.shape(radar_z)
    rad_z1d = radar_z.ravel()
    rad_T1d = np.interp(rad_z1d, snd_z, snd_T)
    return np.reshape(rad_T1d, shape), radar_z
    
# ***************************************************************************************

def get_beam_height(self):

    print('', 'Calculating beam height...', '', sep='\n')

    ref_field_name ='CZ'

    radar_z = get_z_from_radar(self.radar)

    add_field_to_radar_object(radar_z, self.radar, field_name='HEIGHT', units='km',
                              long_name='Height',
                              standard_name='Height', 
                              dz_field=ref_field_name)
    return self.radar

# ***************************************************************************************

def convertData(rawData, timeStamp, name):
    sound_dict = {"PRES":[],"HGHT":[],"TEMP":[],"DWPT":[],"RELH":[],
                  "MIXR":[],"WINDDIR":[],"WINDSP":[],"THTA":[],"THTE":[],
                  "THTV":[]}
    theYear  = timeStamp[0:4]
    theMonth = timeStamp[4:6]
    theDay   = timeStamp[6:8]
    theHour  = timeStamp[8:10]

    dataArray = rawData.split("\n")

    for i in range(6, len(dataArray)):
        if (dataArray[i] == ""):
            continue

        pressure = stripNumber(dataArray[i][9:17])
        height = stripNumber(dataArray[i][16:24])
        tempc = stripNumber(dataArray[i][23:30])
        tempd = stripNumber(dataArray[i][29:40])
        relh = 0.0
        mixr = 0.0
        winddir = stripNumber(dataArray[i][39:48])
        windsp = stripNumber(dataArray[i][47:55])
        thta = 0.0
        thte = 0.0
        thtv = 0.0

        bad = '99999'
        too_big = 100000
        if (pressure == bad or height == bad or tempc == bad or tempd == bad or winddir == bad or windsp == bad):
            print('Bad data, line removed.')
        elif (float(pressure) > too_big or float(height) > too_big or float(tempc) > too_big or float(tempd) > too_big or float(winddir) > too_big or float(windsp) > too_big):
            print('Bad data, line removed.')
        else:
            pressure = round(float(pressure)*0.1,1) 
            sound_dict["PRES"].append(pressure)
            height = round(float(height)*1.0,1)
            sound_dict["HGHT"].append(height)
            tempc = round(float(tempc)*0.1,1)
            sound_dict["TEMP"].append(tempc)
            tempd = round(float(tempd)*0.1,1)
            sound_dict["DWPT"].append(tempd)
            sound_dict["RELH"].append(relh)
            sound_dict["MIXR"].append(mixr)
            sound_dict["WINDDIR"].append(winddir)
            sound_dict["WINDSP"].append(windsp)
            sound_dict["THTA"].append(thta)
            sound_dict["THTE"].append(thte)
            sound_dict["THTV"].append(thtv)

    print('', '    Retrieving sounding for ' + name + ' at ' + theHour + 'Z on ' + theMonth + '/' + theDay + '/' + theYear, ' ', sep = '\n')

    return sound_dict

# ***************************************************************************************
    
def retrieveData(timeStamp, radar_site):
    year =  timeStamp[0:4]
    month = timeStamp[4:6]
    day =   timeStamp[6:8]
    hour =  timeStamp[8:10]
    name = radar_site[0]
    lat = radar_site[1]
    lon = radar_site[2]

    datetime_object = datetime.datetime.strptime(month, "%m")
    month_name = datetime_object.strftime("%b")

    # Web address for the archive corresponding to the given timestamp and station ID
    requestURL = "https://rucsoundings.noaa.gov/get_soundings.cgi?data_source=Op40&latest=latest&start_year="+year+"&start_month_name="+month_name+"&start_mday="+day+"&start_hour="+hour+"&start_min=0&n_hrs=1.0&fcst_len=shortest&airport="+lat+"%2C"+lon+"&text=Ascii%20text%20%28GSD%20format%29&hydrometeors=false&start="+hour
    try:
        # Get the data at the web address
        theData = urllib.request.urlopen(requestURL).read()
        theData = theData.decode('ascii')

        # And convert it
        newData = convertData(theData, timeStamp, name)

        return newData
    except:
        throwError("AN ERROR OCCURRED IN RETRIEVING THE DATA")
        return "ERROR"

# ***************************************************************************************

# Function to strip the number element out of a string element
def stripNumber(theString):
    c = 0
    buffer = ""

    while ( c < len(theString) and not(isNumeric(theString[c])) ):
        c = c + 1

    while ( c < len(theString) and isNumeric(theString[c]) ):
        buffer = buffer + theString[c]
        c = c + 1

    # Missing data
    if (buffer == ""):
        buffer = "-9999.00"

    return buffer

# ***************************************************************************************

def isNumeric(theChar):
    cc = ord(theChar)
    
    return ( (cc >= 0x30 and cc <= 0x39) or (cc == 0x2e) or (cc == 0x2d) )

# ***************************************************************************************

def merge_split_cuts(self, time_gap=120):

    sweeps = []
    for i, elev in enumerate(self.radar.fixed_angle['data']):
        start = self.radar.sweep_start_ray_index['data'][i]
        end = self.radar.sweep_end_ray_index['data'][i] + 1
        time = float(np.mean(self.radar.time['data'][start:end]))

        fields_present = set()
        for field in self.radar.fields:
            data = self.radar.fields[field]['data'][start:end]
            if np.ma.is_masked(data) and data.mask.all():
                continue
            fields_present.add(field)

        sweeps.append({
            'index': i,
            'elev': round(float(elev), 3),
            'time': time,
            'has_ref': any(f in fields_present for f in ['REF', 'DZ']),
            'has_vel': 'VEL' in fields_present
        })

    # Group by elevation angle, then split by time gap
    grouped = []
    for elev in sorted(set(s['elev'] for s in sweeps)):
        subset = [s for s in sweeps if s['elev'] == elev]
        subset.sort(key=lambda x: x['time'])

        group = []
        prev_time = None
        for s in subset:
            if prev_time is None or abs(s['time'] - prev_time) > time_gap:
                if group:
                    grouped.append(group)
                group = [s]
            else:
                group.append(s)
            prev_time = s['time']

        if group:
            grouped.append(group)

    # For each group, pick best REF + VEL sweep
    ref_sweeps = []
    vel_sweeps = []

    for group in grouped:
        best_ref = next((s for s in group if s['has_ref']), None)
        best_vel = next((s for s in group if s['has_vel']), None)

        if best_ref:
            ref_sweeps.append(best_ref['index'])
        if best_vel:
            vel_sweeps.append(best_vel['index'])

    # Extract and merge
    radar_dz = self.radar.extract_sweeps(sorted(ref_sweeps))
    radar_vr = self.radar.extract_sweeps(sorted(vel_sweeps))

    radar_new = copy.deepcopy(radar_dz)

    for field_name in ['VEL', 'SW']:
        if field_name in radar_vr.fields:
            data = radar_vr.fields[field_name]['data']
            if data.shape == radar_new.fields['REF']['data'].shape:
                radar_new.add_field(field_name, radar_vr.fields[field_name], replace_existing=True)

    print("Merged Elevation Angles:\n", radar_new.fixed_angle['data'])
    return radar_new

    '''
    Functionm to merge 88D split cut sweeps and output new radar object.
    '''
    '''
    field_names = self.radar.fields.keys()
    grouped_sweeps = []  # List of sweep index lists

    # Group by elevation, but separate if temporally distinct
    visited = np.zeros(len(self.radar.fixed_angle['data']), dtype=bool)
    for i, angle_i in enumerate(self.radar.fixed_angle['data']):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        time_i = self.radar.time['data'][self.radar.sweep_start_ray_index['data'][i]]

        for j in range(i + 1, len(self.radar.fixed_angle['data'])):
            if visited[j]:
                continue
            angle_j = self.radar.fixed_angle['data'][j]
            time_j = self.radar.time['data'][self.radar.sweep_start_ray_index['data'][j]]

            if abs(angle_i - angle_j) <= round_tol and abs(time_i - time_j) <= time_separation_threshold:
                group.append(j)
                visited[j] = True

        grouped_sweeps.append(group)

    # Process groups to merge them
    azimuth_data, elevation_data, time_data = [], [], []
    all_data = {f: [] for f in field_names}
    nyquist_velocity_data, unambiguous_range_data = [], []
    start_indices, end_indices = [], []
    fixed_angles, sweep_modes, sweep_numbers = [], [], []
    ray_index = 0

    for sweep_num, group in enumerate(grouped_sweeps):
        azimuths, elevations, times = [], [], []
        sweep_data = {f: [] for f in field_names}
        rays_in_sweep = 0

        for sweep in group:
            start = self.radar.sweep_start_ray_index['data'][sweep]
            end = self.radar.sweep_end_ray_index['data'][sweep] + 1
            rays = end - start
            rays_in_sweep += rays

            azimuths.append(self.radar.azimuth['data'][start:end])
            elevations.append(self.radar.elevation['data'][start:end])
            times.append(self.radar.time['data'][start:end])
            for f in field_names:
                sweep_data[f].append(self.radar.fields[f]['data'][start:end])

            if self.radar.instrument_parameters is not None:
                if 'nyquist_velocity' in self.radar.instrument_parameters:
                    nyquist_velocity_data.append(self.radar.instrument_parameters['nyquist_velocity']['data'][start:end])
                if 'unambiguous_range' in self.radar.instrument_parameters:
                    unambiguous_range_data.append(self.radar.instrument_parameters['unambiguous_range']['data'][start:end])

        # Concatenate
        azimuth_concat = np.concatenate(azimuths)
        elevation_concat = np.concatenate(elevations)
        time_concat = np.concatenate(times)
        azimuth_data.append(azimuth_concat)
        elevation_data.append(elevation_concat)
        time_data.append(time_concat)

        for f in field_names:
            all_data[f].append(np.ma.concatenate(sweep_data[f]))

        start_indices.append(ray_index)
        ray_index += rays_in_sweep
        end_indices.append(ray_index - 1)

        # Use average elevation from group
        fixed_angles.append(np.mean([self.radar.fixed_angle['data'][i] for i in group]))
        sweep_modes.append(b'azimuth_surveillance')
        sweep_numbers.append(sweep_num)

    # Assemble output Radar object
    radar_out = pyart.core.Radar(
        time={
            'data': np.concatenate(time_data),
            'units': self.radar.time['units'],
            'standard_name': 'time',
            'long_name': 'time_in_seconds_since_volume_start',
            'calendar': 'gregorian',
            'comment': self.radar.time.get('comment', ''),
        },
        _range=self.radar.range,
        fields={
            f: {
                'data': np.ma.concatenate(all_data[f]),
                **{k: v for k, v in self.radar.fields[f].items() if k != 'data'}
            } for f in field_names
        },
        metadata=self.radar.metadata,
        scan_type=self.radar.scan_type,
        latitude=self.radar.latitude,
        longitude=self.radar.longitude,
        altitude=self.radar.altitude,
        sweep_number={'data': np.array(sweep_numbers, dtype='int32')},
        sweep_mode={'data': np.array(sweep_modes)},
        fixed_angle={'data': np.array(fixed_angles, dtype='float32')},
        sweep_start_ray_index={'data': np.array(start_indices, dtype='int32')},
        sweep_end_ray_index={'data': np.array(end_indices, dtype='int32')},
        azimuth={'data': np.concatenate(azimuth_data)},
        elevation={'data': np.concatenate(elevation_data)},
        instrument_parameters={
            'nyquist_velocity': {
                'data': np.concatenate(nyquist_velocity_data) if nyquist_velocity_data else None,
                'units': 'meters_per_second',
                'long_name': 'Nyquist velocity',
                'comments': 'Unambiguous velocity',
                'meta_group': 'instrument_parameters',
            },
            'unambiguous_range': {
                'data': np.concatenate(unambiguous_range_data) if unambiguous_range_data else None,
                'units': 'meters',
                'long_name': 'Unambiguous range',
                'comments': 'Unambiguous range',
                'meta_group': 'instrument_parameters',
            }
        } if nyquist_velocity_data and unambiguous_range_data else None
    )

    print("Merged Elevation Angles:", radar_out.fixed_angle['data'])
    
    return radar_out

    
    # Removes triple base scan
    ele_list = self.radar.fixed_angle['data'][:]
    if ele_list[0] == ele_list[2] and ele_list[3] != ele_list[5] and ele_list[6] != ele_list[8]:
        rm_list = [x for x in range(len(ele_list)) if x != 2]
        self.radar = self.radar.extract_sweeps(rm_list)

    sweep_table = []
    reflectivity = []
    velocity = []

    # Create a list of elevations
    vcp = self.radar.metadata['vcp_pattern']
    print('VCP Pattern:  ', vcp)
    elist = [x.shape[0] for x in self.radar.iter_elevation()]
    n = 0
    while n <= len(elist)-1:
        
        vcp_2 = [215, 35, 212, 32, 12, 31, 34]
        vcp_3 = [112]

        if elist[n] == 720 and elist[n+1] == 720:
            sweep_table.append((n, n+1))
            reflectivity.append(n)
            velocity.append(n+1)
            if  vcp in vcp_2:
                n += 2
            elif vcp in vcp_3:
                n += 3
        elif elist[n] == 360:
            sweep_table.append((n, n))
            reflectivity.append(n)
            velocity.append(n)
            n += 1
        elif (elist[n] == 720 or elist[n] == 360)  and (elist[n+1] != 720 or elist[n+1] != 360):
            print(elist)
            sys.exit("Bad file")

    print(" ","Merging WSR-88D Split Cuts", sep='\n')
    #print("\n Number of reflectivity levels:  %d" % len(reflectivity))
    #print("\n Number of radial velocity  levels:  %d\n" % len(velocity))
    
    #Create DZ and VR radar structures
    radar_dz = self.radar.extract_sweeps(reflectivity)
    radar_vr = self.radar.extract_sweeps(velocity)
    #print(radar_dz.fixed_angle['data'][:])
    #print(radar_vr.fixed_angle['data'][:])

    #Create new radar
    radar = copy.deepcopy(radar_dz)

    fill_value = -32767.0

    #Add VR and SW fleids to new radar
    vr_field = radar_vr.fields['VEL']['data'].copy()
    vr_dict = {'data': vr_field, 'units': '', 'long_name': 'Velocity',
               '_FillValue': fill_value, 'standard_name': 'VEL'}
    radar.add_field('VEL', vr_dict, replace_existing=True)

    sw_field = radar_vr.fields['SW']['data'].copy()
    sw_dict = {'data': sw_field, 'units': '', 'long_name': 'Spectrum Wdith',
               '_FillValue': fill_value, 'standard_name': 'SW'}
    radar.add_field('SW', sw_dict, replace_existing=True)


    print("New merged elevation angles:  ", radar.fixed_angle['data'][:], sep='\n')

    return radar
    
    '''
# ***************************************************************************************

def remove_mrle(self):

    '''
    Function to remove 88D MRLE and SAILS sweeps.
    '''  

    #Get list of elevations
    elev_list = self.radar.fixed_angle['data'][:]

    last_num = elev_list[0]
    new_list = [last_num]
    sw = 1
    sweep_index = [0]

    #Get index of elevations greater than last elevation 
    for x in elev_list[1:]:
        if x >= last_num:
            sweep_index.append(sw)
            new_list.append(x)
            last_num = x
        sw = sw+1
        
    print(" ", "Removing MRLE sweeps", "Following sweeps will be kept:  ", sweep_index, sep='\n')     

    final_radar = self.radar.extract_sweeps(sweep_index)

    print("With the following elevations:  ", final_radar.fixed_angle['data'][:], sep='\n')

    return final_radar

# ***************************************************************************************

def reorder_sweeps(radar):

    # Reorder sweeps in ascending order
    # sweep_index = np.argsort(radar.elevation['data'][radar.sweep_start_ray_index['data']])
    # final_radar = radar.extract_sweeps(sweep_index)
    final_radar = pyart.util.subset_radar(radar, list(radar.fields), ele_min=0., ele_max=90.)
    print('New sweep angles:  ',final_radar.fixed_angle['data'][:], sep='\n')

    # Azimuths are negative, modify them to fit 0-360
    if final_radar.azimuth['data'][0] < 0:
        final_radar.azimuth['data'] = np.mod(final_radar.azimuth['data'], 360)
        az = final_radar.get_azimuth(0)
        print(f"Azimuth Min/Max:  {az.min()} {az.max()}")

    return final_radar

# ***************************************************************************************

def get_cal_numbers(self):

    # Grabs calibration data from text file
    DT = datetime.datetime(2023, 4, 1, 0, 0, 0)
    radar_DT = pyart.util.datetime_from_radar(self.radar)
    cal_file = self.cal_dir + self.site + '_' + self.year + '-' + self.month + '_cal.txt'

    headings = ["date","p1","p2","Zcal","ZDRcal"]
    cal = pd.read_csv(cal_file, header=None, delimiter=r"\s+", names=headings)

    # Account for IDL vs pyART calbration code
    if radar_DT < DT:
        self.ref_cal = cal.Zcal[radar_DT.day-1] * -1.0
    else:  
        self.ref_cal = cal.Zcal[radar_DT.day-1]
    
    self.zdr_cal = cal.ZDRcal[radar_DT.day-1]

    print(" ", headings, sep='\n')
    print(cal.date[radar_DT.day-1], cal.p1[radar_DT.day-1], cal.p2[radar_DT.day-1],
          cal.Zcal[radar_DT.day-1], cal.ZDRcal[radar_DT.day-1])

# ***************************************************************************************

def retrieve_ML(mydata):

    # Retrieve expected ML for winter HID

    is_all_neg = np.all(mydata['temp'] < 0)
    if is_all_neg:
        expected_ML = 0
    else:
        a = mydata['temp']
        idx=np.where(np.diff(np.sign(a)) != 0)[0] + 1
        int1 = (idx[0])
        int2 = int1 - 1
        atol = a[int2] + a[int2]
        wh0 = np.where(np.isclose(np.abs(mydata['temp']),0.0,atol=atol))
        try:
            expected_ML = np.array(mydata['hght'])[wh0[0]][0]/1000.
        except:
            expected_ML = 0
    
    print('    Expected ML:  ',expected_ML)   

    return expected_ML

# ***************************************************************************************

def twister_data(timeStamp, radar_site):

    year =  timeStamp[0:4]
    month = timeStamp[4:6]
    day =   timeStamp[6:8]
    hour =  timeStamp[8:10]
    name = radar_site[0]
    lat = radar_site[1]
    lon = radar_site[2]

    print(name,lat,lon)
    model ='RAP'
    if model == 'RAP':  grid='255'
    if model == 'GFS':  grid='3'

    requestURL = 'http://www.twisterdata.com/index.php?sounding.\
                  lat='+lat+'&sounding.lon='+lon+'&sndclick=y&\
                  prog=forecast&model='+model+'&grid='+grid+'&model_yyyy='+year+'&\
                  model_mm='+month+'&model_dd='+day+'&model_init_hh='+hour+'&\
                  fhour=00&parameter=TMPF&level=2&unit=M_ABOVE_GROUND&\
                  maximize=n&mode=singlemap&sounding=y&output=text&\
                  view=large&archive=false'

    url = requestURL.replace(" ","")

    rawData = urlopen(url).read()
    theData = rawData.decode('ascii')
    dataArray = theData.split("\n")

    start_sounding = model + " Text Sounding"
    end_sounding = '</td></tr></table></div></div><div class="textAdBar">'
    for i in range(0, len(dataArray)):
        if start_sounding in dataArray[i]:
            print('', dataArray[i][34:118], '', sep='\n')
            sline = i
            #print(sline)
    for j in range(sline, len(dataArray)):
        if end_sounding in dataArray[j]:
            #print(dataArray[j])
            eline = j
            #print(eline)

    sound_dict = {"PRES":[],"HGHT":[],"TEMP":[],"DWPT":[],"RELH":[],
                  "MIXR":[],"WINDDIR":[],"WINDSP":[],
                  "THTA":[],"THTE":[],"THTV":[]}

    for k in range(sline+1, eline-1):
        cline = re.findall(r"[-+]?\d*\.\d+|\d+",dataArray[k])
        #print(cline)
        pressure = round(float(cline[0])*1.0,1)
        height = round(float(cline[1])*1.0,1)
        tempc = round(float(cline[2])*1.0,1)
        tempd = round(float(cline[3])*1.0,1)
        relh = cline[4]
        mixr = cline[5]
        winddir = cline[6]
        windsp = cline[7]
        thta = cline[10]
        thte = cline[11]
        thtv = cline[12]
    
        sound_dict["PRES"].append(pressure)
        sound_dict["HGHT"].append(height)
        sound_dict["TEMP"].append(tempc)
        sound_dict["DWPT"].append(tempd)
        sound_dict["RELH"].append(relh)
        sound_dict["MIXR"].append(mixr)
        sound_dict["WINDDIR"].append(winddir)
        sound_dict["WINDSP"].append(windsp)
        sound_dict["THTA"].append(thta)
        sound_dict["THTE"].append(thte)
        sound_dict["THTV"].append(thtv)

    return sound_dict
# ***************************************************************************************

def remove_HDF_header(file):

    dn = os.path.abspath(file)
    os.makedirs(os.path.dirname(dn) + '/temp/',exist_ok=True)
    temp_dir = os.path.dirname(dn) + '/temp/'
    temp_file = shutil.copy(file, temp_dir)

    with open(temp_file, "r+b") as f:
        mmapped_file = mmap.mmap(f.fileno(), 0)
    
        # Find the first two line breaks and remove them
        first_newline = mmapped_file.find(b"\n")
        second_newline = mmapped_file.find(b"\n", first_newline + 1)
    
        if second_newline != -1:
            mmapped_file.move(0, second_newline + 1, len(mmapped_file) - (second_newline + 1))
            mmapped_file.flush()
    
        mmapped_file.close()

    return temp_file

# ***************************************************************************************

    