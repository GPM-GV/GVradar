# ***************************************************************************************
'''
Functions used by both dpqc.py and dp_products.py.

Developed by the NASA GPM-GV group
V0.3 - 02/19/2021 - update by Jason Pippitt NASA/GSFC/SSAI
'''
# ***************************************************************************************

import numpy as np
from copy import deepcopy
import pyart
import os
import pandas as pd
from skewt import SkewT
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain,
                            csu_dsd, csu_kdp, csu_misc, fundamentals)

# ***************************************************************************************

def get_ruc_sounding(self):

    """
    Finds correct RUC hourly sounding based on radar time stamp, from RUC sounding archive.
    
    Reads text sounding file and creates dictionary for input to SkewT.
    returns:  sounding dictionary

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    print('', 'Interpolating sounding to radar structure...', sep='\n')
    
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
    sounding_dir = snd_dir + year + '/' + month + day + '/' + self.site + '/' + self.site + '_' + year + '_' + month + day + '_' + hh + 'UTC.txt'
    
    print('Sounding file -->  ' + sounding_dir, '', sep='\n')

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
    
    print('', 'Sounding file -->  ' + sounding_dir, '', sep='\n')

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
    return self.radar

# ***************************************************************************************

def use_uwy_sounding(self):
    
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

    sounding_dir = self.sounding_dir
    
    print('', 'Sounding file -->  ' + sounding_dir, '', sep='\n')

    sounding=SkewT.Sounding(sounding_dir)
           
    radar_T, radar_z = interpolate_sounding_to_radar(sounding, self.radar)
    
    add_field_to_radar_object(radar_T, self.radar, field_name='TEMP', units='deg C',
                                 long_name='Temperature',
                                 standard_name='Temperature',
                                 dz_field=self.ref_field_name)
   
    add_field_to_radar_object(radar_z, self.radar, field_name='HEIGHT', units='km',
                                 long_name='Height',
                                 standard_name='Height', 
                                 dz_field=self.ref_field_name)
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
    if 'PHIDP2' in self.radar.fields.keys():
        old_fields = ['DBZ2', 'VEL2', 'WIDTH2', 'ZDR2', 'KDP2', 'PHIDP2', 'RHOHV2', 'SQI2']
        new_fields = ['DZ',     'VR',    'SW',   'DR',   'KD',   'PH',     'RH',     'SQ']
    elif 'DBZ' in self.radar.fields.keys():
        old_fields = ['DBZ', 'VEL', 'WIDTH', 'ZDR', 'KDP', 'PHIDP', 'SQI', 'RHOHV']
        new_fields = ['DZ',  'VR',   'SW',   'DR',  'KD',   'PH',    'SQ',    'RH']
    elif 'DBZ2' in self.radar.fields.keys():
        old_fields = ['DBZ2', 'VEL2', 'WIDTH2', 'ZDR2', 'KDP2', 'PHIDP2', 'RHOHV2',  'SQI2']
        new_fields = ['DZ',   'VR',   'SW',     'DR',   'KD',   'PH',     'RH',      'SQ']
    elif 'reflectivity' in self.radar.fields.keys():
        old_fields = ['differential_phase', 'velocity', 'spectrum_width', 'reflectivity', 'differential_reflectivity', 'cross_correlation_ratio']
        new_fields = ['PH',     'VR',     'SW',   'DZ',   'DR', 'RH']
    elif 'REF' in self.radar.fields.keys():
        old_fields = ['SW', 'PHI', 'ZDR', 'REF', 'VEL', 'RHO']
        new_fields = ['SW', 'PH' , 'DR' , 'DZ' , 'VR' , 'RH' ]
    elif 'DZ' in self.radar.fields.keys():
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
    if ('CZ' or 'DZ') not in self.radar.fields.keys():
        if self.site == 'NPOL' or self.site == 'KWAJ':
            zz = deepcopy(self.radar.fields['DBT2'])
            cz = self.radar.fields['DBT2']['data'].copy()
            add_field_to_radar_object(cz, self.radar, field_name='CZ', 
                                         units=' ',
                                         long_name='Corrected Reflectivity', 
                                         standard_name='Corrected Reflectivity', 
                                         dz_field='DZ') 
        else: 
            zz = deepcopy(self.radar.fields['DZ'])
            cz = self.radar.fields['DZ']['data'].copy()
            add_field_to_radar_object(cz, self.radar, field_name='CZ', 
                                         units=' ',
                                         long_name='Corrected Reflectivity', 
                                         standard_name='Corrected Reflectivity', 
                                         dz_field='DZ')
         
        print(self.radar.fields.keys())
        return self.radar, zz
    else:
        zz = deepcopy(self.radar.fields['DZ'])
        print(self.radar.fields.keys())
        return self.radar, zz

# ***************************************************************************************

def output_cf(self):

    # Outputs CF radial file
    # Declare output dir

    out_dir = self.cf_dir
    os.makedirs(out_dir, exist_ok=True)
   
    out_file = out_dir + '/' + self.site + '_' + self.year + '_' + self.month + self.day + '_' + self.hh + self.mm + self.ss + '_' + self.scan_type + '.cf'
    
    pyart.io.write_cfradial(out_file,self.radar)
    
    print('Output cfRadial --> ' + out_file, '', sep='\n')

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

    if site == 'NPOL1': site = 'NPOL'         
    if site == 'LAVA1': site = 'KWAJ'
    
    scan_type = radar.scan_type.upper()
    
    radar_DT = pyart.util.datetime_from_radar(radar)   
    month = str(radar_DT.month).zfill(2)
    day = str(radar_DT.day).zfill(2)
    year = str(radar_DT.year).zfill(4)
    hh = str(radar_DT.hour).zfill(2)
    mm = str(radar_DT.minute).zfill(2)
    ss = str(radar_DT.second).zfill(2)

    site_time = {'site': site, 'scan_type': scan_type, 'month': month, 'day': day,
                 'year': year, 'hh': hh, 'mm': mm, 'ss': ss, 'ref_field_name': 'CZ',
                 'phi_field_name': 'PH', 'zdr_field_name': 'DR'}

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