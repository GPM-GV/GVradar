import numpy as np
import math
from copy import deepcopy
import pyart
import os
import subprocess
import shlex
import pandas as pd
from skewt import SkewT
import gpm_dp_utils as gu
from csu_radartools.csu_liquid_ice_mass import linearize
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain,
                            csu_dsd, csu_kdp, csu_misc, fundamentals)

# ***************************************************************************************

def unfold_phidp(radar):

    """
    Function for unfolding phidp
    Written by: David A. Marks, NASA/WFF/SSAI

    Parameters:
    radar: pyart radar object
    ref_field_name: name of reflectivty field (should be QC'd)
    phi_field_name: name of PhiDP field (should be unfolded)

    Return
    radar: radar object with unfolded PHM field included
    """
                
    print('    Unfolding PhiDP...')

    BAD_DATA       = -32767.0        # Fill in bad data values
    FIRST_GATE     = 5000           # First gate to begin unfolding
    MAX_PHIDP_DIFF = 360.0          # Set maximum phidp gate-to-gate difference allowed
    ref_field_name = 'CZ'
    phi_field_name = 'PH'

    # Copy current PhiDP field to phm_field
    phm_field = radar.fields[phi_field_name]['data'].copy()

    # Get gate spacing info and determine start gate for unfolding
    # Start at 5 km from radar to avoid clutter gates with bad phase 
    gate_spacing = radar.range['meters_between_gates']
    start_gate = int(FIRST_GATE / gate_spacing)
    nsweeps = radar.nsweeps
    nrays = phm_field.data.shape[0]

    # Loop through the rays and perform unfolding if needed
    # By specifying iray for data and mask provides gate info
    # for iray in range(0, 1):

    for iray in range(0, nrays-1):
        gate_data = phm_field.data[iray]
        ngates = gate_data.shape[0]

        # Conditional where for valid data -- NPOL only.
        # Phase data from other radars should be evaluated for phase range values
        good = np.ma.where(gate_data >= 0)
        bad = np.ma.where(gate_data < 0)
        final_data = gate_data[good]
        num_final = final_data.shape[0]
        #print("Num_Final = ", str(num_final))

        folded_gates = 0
        for igate in range(start_gate,num_final-2):
            diff = final_data[igate+1] - final_data[igate]
            if abs(diff) > MAX_PHIDP_DIFF:
                #print('igate: igate+1: ',final_data[igate],final_data[igate+1])
                final_data[igate+1] += 360
                folded_gates += 1

        # Put corrected data back into ray
        gate_data[good] = final_data
        gate_data[bad] = BAD_DATA

        # Replace corrected data in phm_field. Original phidp remains the same
        phm_field.data[iray] = gate_data

    # Create new field for corrected PH -- name it phm
    radar = gu.add_field_to_radar_object(phm_field, radar, field_name='PH', 
		units='deg',
		long_name=' Differential Phase (Marks)',
		standard_name='Specific Differential Phase (Marks)',
		dz_field=ref_field_name)
    
    return radar

# ***************************************************************************************

def calculate_kdp(radar):

    """
    Wrapper for calculating Kdp using csu_kdp.calc_kdp_bringi from CSU_RadarTools
    Thank Timothy Lang et al.
    Parameters:
    -----------
    radar: pyart radar object
    ref_field_name: name of reflectivty field (should be QC'd)
    phi_field_name: name of PhiDP field (should be unfolded)

    Return
    ------
    radar: radar object with KDPB, PHIDPB and STDPHIDP added to original

    NOTE: KDPB: Bringi Kdp, PHIDPB: Bringi-filtered PhiDP, STDPHIB: Std-dev of PhiDP
    """
    print('    Getting new Kdp...')

    ref_field_name = 'CZ'
    phi_field_name = 'PH'
    radar_lon = radar.longitude['data'][:][0]
    radar_lat = radar.latitude['data'][:][0]

#    DZ = gu.extract_unmasked_data(radar, ref_field_name)
#    DP = gu.extract_unmasked_data(radar, phi_field_name)
    DZ = radar.fields[ref_field_name]['data'].copy()
    DP = radar.fields[phi_field_name]['data'].copy()

    # Range needs to be supplied as a variable, with same shape as DZ
    rng2d, az2d = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    gate_spacing = radar.range['meters_between_gates']

    KDPB, PHIDPB, STDPHIB = csu_kdp.calc_kdp_bringi(dp=DP, dz=DZ, rng=rng2d/1000.0, 
                                                    thsd=25, gs=gate_spacing, window=5)

    radar = gu.add_field_to_radar_object(KDPB, radar, field_name='KD', 
		units='deg/km',
		long_name='Specific Differential Phase (Bringi)',
		standard_name='Specific Differential Phase (Bringi)',
		dz_field=ref_field_name)

    radar = gu.add_field_to_radar_object(PHIDPB, radar, 
		field_name='PHIDPB', units='deg',
		long_name='Differential Phase (Bringi)',
		standard_name='Differential Phase (Bringi)',
		dz_field=ref_field_name)

    radar = gu.add_field_to_radar_object(STDPHIB, radar, 
		field_name='SD', units='deg',
		long_name='STD Differential Phase (Bringi)',
		standard_name='STD Differential Phase (Bringi)',
		dz_field=ref_field_name)

    return radar

# ***************************************************************************************

def calibrate(radar, thresh_dict):

    """
    Applies calibration adjustments to DBZ and ZDR fields.

    returns:  radar with calibrated fields.

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    print()
    print('Calibrating Reflectivity and ZDR...')
    ref_field_name = 'CZ'
    zdr_field_name = 'DR'

    ref_cal = thresh_dict['ref_cal']
    zdr_cal = thresh_dict['zdr_cal']

    fill_value = -32767.0
    #Calibrate Reflectivity field
    ref_field = radar.fields[ref_field_name]['data'].copy()
    corr_dbz = pyart.correct.correct_bias(radar, bias=ref_cal, field_name=ref_field_name)
    corz_dict = {'data': corr_dbz['data'], 'units': '', 'long_name': 'CZ',
                 '_FillValue': fill_value, 'standard_name': 'CZ'}
    radar.add_field('CZ', corz_dict, replace_existing=True)
    
    #Calibrate ZDR field
    zdr_field = radar.fields[zdr_field_name]['data'].copy()
    corr_zdr = pyart.correct.correct_bias(radar, bias=zdr_cal, field_name=zdr_field_name)
    corzdr_dict = {'data': corr_zdr['data'], 'units': '', 'long_name': 'DR',
                 '_FillValue': fill_value, 'standard_name': 'DR'}
    radar.add_field('DR', corzdr_dict, replace_existing=True)

    return radar

# ***************************************************************************************

def get_ruc_sounding(radar, thresh_dict):

    """
    Finds correct RUC hourly sounding based on radar time stamp.
    Reads text sounding file and creates dictionary for input to SkewT.
    returns:  sounding dictionary

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    ref_field_name = 'CZ'

    print()
    if thresh_dict['use_sounding'] == True:
        print('Interpolating sounding to radar structure...')
        snd_dir = thresh_dict['sounding_dir']
        if 'site_name' in radar.metadata.keys():
            site = radar.metadata['site_name'].upper()
        elif 'instrument_name' in radar.metadata.keys():
            if isinstance(radar.metadata['instrument_name'], bytes):
                site = radar.metadata['instrument_name'].decode().upper()
            else:
                site = radar.metadata['instrument_name'].upper()
        else:
            site=''
                 

    #Retrieve proper sounding for date and time
    
        radar_DT = pyart.util.datetime_from_radar(radar)
     
        month = str(radar_DT.month).zfill(2)
        day = str(radar_DT.day).zfill(2)
        year = str(radar_DT.year).zfill(4)
        hh = str(radar_DT.hour).zfill(2)
        mm = str(radar_DT.minute).zfill(2)
        hour =  radar_DT.hour

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
        sounding_dir = snd_dir + year + '/' + month + day + '/' + site + '/' + site + '_' + year + '_' + month + day + '_' + hh + 'UTC.txt'
    
        print('Sounding file -->  ' + sounding_dir)
        print()

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
           
        radar_T, radar_z = gu.interpolate_sounding_to_radar(sounding, radar)
    
        gu.add_field_to_radar_object(radar_T, radar, field_name='TEMP', units='deg C',
                                     long_name='Temperature',
                                     standard_name='Temperature',
                                     dz_field=ref_field_name)

    if thresh_dict['use_sounding'] == False:
        print('Calculating beam height...')
        print()
        radar_z = gu.get_beam_height(radar)

    gu.add_field_to_radar_object(radar_z, radar, field_name='HEIGHT', units='km',
                                 long_name='Height',
                                 standard_name='Height', 
                                 dz_field=ref_field_name)

    return radar

# ***************************************************************************************

def mask_cone_of_silence(radar, thresh_dict):

    """
    filter out any data inside the cone of silence

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    thresh_dict : dict
            a dictionary defining the region of interest

    Returns
    -------
    cos_flag : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': thresh_dict['coshmin'], 'hmax': thresh_dict['coshmax'],
              'rmin':  thresh_dict['cosrmin'] * 1000, 'rmax':  thresh_dict['cosrmax'] * 1000,
              'azmin': thresh_dict['cosazmin'], 'azmax': thresh_dict['cosazmax'],
              'elmin': thresh_dict['coselmin'], 'elmax': thresh_dict['coselmax']}
    
    cos_flag = np.ma.ones((radar.nrays, radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        cos_flag[radar.gate_altitude['data'] < sector['hmin']] = 0
    if sector['hmax'] is not None:
        cos_flag[radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits

    if sector['rmin'] is not None:
        cos_flag[:, radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        cos_flag[:, radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        cos_flag[radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        cos_flag[radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            cos_flag[radar.azimuth['data'] < sector['azmin'], :] = 0
            cos_flag[radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            cos_flag[np.logical_and(
            radar.azimuth['data'] < sector['azmin'],
            radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        cos_flag[radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        cos_flag[radar.azimuth['data'] > sector['azmax'], :] = 0

    cos_field = cos_flag
    gu.add_field_to_radar_object(cos_field, radar, field_name='COS', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='COS',
                                 standard_name='COS', 
                                 dz_field='CZ')
    return radar

# ***************************************************************************************

def mask_88D_200(radar, thresh_dict):

    """
    filter out any data outside 200 KM

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    thresh_dict : dict
            a dictionary defining the region of interest

    Returns
    -------
    cos_flag : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': 0, 'hmax': None,
              'rmin': 200 * 1000, 'rmax':  300 * 1000,
              'azmin': 0, 'azmax': 360,
              'elmin': 0, 'elmax': None}
    
    WSR_flag = np.ma.ones((radar.nrays, radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        WSR_flag[radar.gate_altitude['data'] < sector['hmin']] = 0
    if sector['hmax'] is not None:
        WSR_flag[radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits

    if sector['rmin'] is not None:
        WSR_flag[:, radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        WSR_flag[:, radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        WSR_flag[radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        WSR_flag[radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            WSR_flag[radar.azimuth['data'] < sector['azmin'], :] = 0
            WSR_flag[radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            WSR_flag[np.logical_and(
            radar.azimuth['data'] < sector['azmin'],
            radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        WSR_flag[radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        WSR_flag[radar.azimuth['data'] > sector['azmax'], :] = 0

    WSR_field = WSR_flag
    gu.add_field_to_radar_object(WSR_field, radar, field_name='WSR', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='WSR',
                                 standard_name='WSR', 
                                 dz_field='DZ')
    return radar

# ***************************************************************************************

def sector_wipeout(radar, thresh_dict):
    
    """
    filter out any data inside the region of interest defined by sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    thresh_dict : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_wipeout : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': thresh_dict['sechmin'], 'hmax': thresh_dict['sechmax'],
	      'rmin':  thresh_dict['secrmin'] * 1000, 'rmax':  thresh_dict['secrmax'] * 1000,
              'azmin': thresh_dict['secazmin'], 'azmax': thresh_dict['secazmax'],
	      'elmin': thresh_dict['secelmin'], 'elmax': thresh_dict['secelmax']}

    sector_wipeout = np.ma.ones((radar.nrays, radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            radar.azimuth['data'] < sector['azmin'],
            radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0

    sector_field = sector_wipeout
    gu.add_field_to_radar_object(sector_field, radar, field_name='SEC', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='Sector Mask', 
                                 standard_name='Sector Mask', 
                                 dz_field='CZ')

    return radar

# ***************************************************************************************

def rh_sector(radar, thresh_dict):
    
    """
    filter out any data inside the region of interest that is < rh_sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    thresh_dict : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_rh : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': thresh_dict['rhhmin'], 'hmax': thresh_dict['rhhmax'],
	      'rmin':  thresh_dict['rhrmin'] * 1000, 'rmax':  thresh_dict['rhrmax'] * 1000,
              'azmin': thresh_dict['rhazmin'], 'azmax': thresh_dict['rhazmax'],
	      'elmin': thresh_dict['rhelmin'], 'elmax': thresh_dict['rhelmax'],
              'rh_sec': thresh_dict['rh_sec']}

    sector_wipeout = np.ma.ones((radar.nrays, radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            radar.azimuth['data'] < sector['azmin'],
            radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
    
    rh = radar.fields['RH']['data'].copy()
    sector_r = np.ones(rh.shape)
    rh_sec = sector['rh_sec']
    rh_lt = np.ma.where(rh < rh_sec , 1, 0)
    sec_f = np.logical_and(rh_lt == 1 , sector_wipeout == 1)
    sector_r[sec_f] = 0

    sector_rh = sector_r
    gu.add_field_to_radar_object(sector_rh, radar, field_name='SECRH', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='Sector RH Mask', 
                                 standard_name='Sector RH Mask', 
                                 dz_field='CZ')
     
    return radar

# ***************************************************************************************

def sd_sector(radar, thresh_dict):

    """
    filter out any data inside the region of interest that is < sd_sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    thresh_dict : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_sd : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    sector = {'hmin': thresh_dict['sdhmin'], 'hmax': thresh_dict['sdhmax'],
	      'rmin':  thresh_dict['sdrmin'] * 1000, 'rmax':  thresh_dict['sdrmax'] * 1000,
              'azmin': thresh_dict['sdazmin'], 'azmax': thresh_dict['sdazmax'],
	      'elmin': thresh_dict['sdelmin'], 'elmax': thresh_dict['sdelmax'],
              'sd_sec': thresh_dict['sd_sec']}

    sector_wipeout = np.ma.ones((radar.nrays, radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            radar.azimuth['data'] < sector['azmin'],
            radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
    
    sd = radar.fields['SD']['data'].copy()
    sector_s = np.ones(sd.shape)
    sd_sec = sector['sd_sec']
    sd_lt = np.ma.where(sd > sd_sec , 1, 0)
    sec_f = np.logical_and(sd_lt == 1 , sector_wipeout == 1)
    sector_s[sec_f] = 0

    sector_sd = sector_s
    gu.add_field_to_radar_object(sector_sd, radar, field_name='SECSD', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='Sector SD Mask', 
                                 standard_name='Sector SD Mask', 
                                 dz_field='CZ')

    return radar

# ***************************************************************************************

def ph_sector(radar, thresh_dict):
    
    """
    filter out any data inside the region of interest that is < ph_sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    thresh_dict : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_ph : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': thresh_dict['phhmin'], 'hmax': thresh_dict['phhmax'],
	      'rmin':  thresh_dict['phrmin'] * 1000, 'rmax':  thresh_dict['phrmax'] * 1000,
              'azmin': thresh_dict['phazmin'], 'azmax': thresh_dict['phazmax'],
	      'elmin': thresh_dict['phelmin'], 'elmax': thresh_dict['phelmax'],
              'ph_sec': thresh_dict['ph_sec']}

    sector_wipeout = np.ma.ones((radar.nrays, radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            radar.azimuth['data'] < sector['azmin'],
            radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[radar.azimuth['data'] > sector['azmax'], :] = 0
    
    ph = radar.fields['PHM']['data'].copy()
    sector_p = np.ones(ph.shape)
    ph_sec = sector['ph_sec']
    ph_lt = np.ma.where(ph < ph_sec , 1, 0)
    sec_f = np.logical_and(ph_lt == 1 , sector_wipeout == 1)
    sector_p[sec_f] = 0

    sector_ph = sector_p
    gu.add_field_to_radar_object(sector_ph, radar, field_name='SECPH', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='Sector PH Mask', 
                                 standard_name='Sector PH Mask', 
                                 dz_field='CZ')

    return radar

# ***************************************************************************************
def csu_filters(radar, thresh_dict):

    dz = gu.extract_unmasked_data(radar, 'CZ')
    zdr = gu.extract_unmasked_data(radar, 'DR')

    if thresh_dict['do_insect'] == True:
        insect_mask = csu_misc.insect_filter(dz, zdr)
        bad = -32767.0

        for fld in radar.fields:
            nf = gu.extract_unmasked_data(radar, fld)
            nf_insect = 1.0 * nf
            nf_insect[insect_mask] = bad 
            gu.add_field_to_radar_object(nf_insect, radar, field_name=fld, units='', 
                                     long_name=fld,
                                     standard_name=fld, 
                                     dz_field='CZ')
    
    if thresh_dict['do_despeckle'] == True:
        mask_ds = csu_misc.despeckle(dz, ngates=4)
        bad = -32767.0

        for fld in radar.fields:
            nf = gu.extract_unmasked_data(radar, fld)
            nf_ds = 1.0 * nf
            nf_ds[mask_ds] = bad 
            gu.add_field_to_radar_object(nf_ds, radar, field_name=fld, units='', 
                                         long_name=fld,
                                         standard_name=fld, 
                                         dz_field='CZ')

    return radar
    

# ***************************************************************************************

def threshold_qc_dpfields(radar, thresh_dict):

    """
    Use gatefilter to apply QC by looking at various thresholds of field values.
    Written by: Jason L. Pippitt, NASA/GSFC/SSAI

    Parameters
    ----------
    radar : radar object
            the radar object where the data is

    Thresholds for qc'ing data: dbz_thresh, rh_thresh,  dr_min, dr_max, kdp_min,
                                 kdp_max, sq_thresh, sd_thresh, sec, cos

    Returns
    -------
    radar: QC'd radar with gatefilters applied.

    """
    
    print("Begin Quality Control:  ")
    print("    Thresholding DP Fields...")

    # Declare thresholds for DP fields
    dbz_thresh = thresh_dict['dbz_thresh']
    rh_thresh  = thresh_dict['rh_thresh']
    dr_min     = thresh_dict['dr_min']
    dr_max     = thresh_dict['dr_max']
    sq_thresh  = thresh_dict['sq_thresh']
    sec = 1
    cos = 0

    # Create a pyart gatefilters from radar
    dbzfilter = pyart.filters.GateFilter(radar)
    gatefilter = pyart.filters.GateFilter(radar)

    # Apply dbz, sector, and SQI thresholds regardless of Temp 
    if thresh_dict['do_dbz'] == True: dbzfilter.exclude_below('CZ', dbz_thresh)
    if thresh_dict['do_sector'] == True: dbzfilter.exclude_not_equal('SEC', cos)
    if thresh_dict['do_rh_sector'] == True: dbzfilter.exclude_not_equal('SECRH', sec) 
    if thresh_dict['do_cos'] == True: dbzfilter.exclude_not_equal('COS', cos)
    if thresh_dict['do_sq'] == True: dbzfilter.exclude_below('SQ', sq_thresh)
    if radar.metadata['instrument_name'] == 'WSR-88D': dbzfilter.exclude_not_equal('WSR', cos)

    # Apply gate filters to radar
    for fld in radar.fields:
        nf = deepcopy(radar.fields[fld])
        nf['data'] = np.ma.masked_where(dbzfilter.gate_excluded, nf['data'])
        radar.add_field(fld, nf, replace_existing=True) 

    # Create AP filter variables
    if thresh_dict['do_ap'] == True:
        dz = radar.fields['CZ']['data'].copy()
        dr = radar.fields['DR']['data'].copy()
        ap = np.ones(dz.shape)
        dz_lt = np.ma.where(dz <= 45 , 1, 0)
        dr_lt = np.ma.where(dr >= 3 , 1, 0)
        ap_t = np.logical_and(dz_lt == 1 , dr_lt == 1)
        ap[ap_t] = 0
        gu.add_field_to_radar_object(ap, radar, field_name='AP', 
                                     units='0 = Z < 0, 1 = Z >= 0',
                                     long_name='AP Mask', 
                                     standard_name='AP Mask', 
                                     dz_field='CZ')

    # Call gatefliters for each field based on temperature or beam height
    if thresh_dict['use_qc_height'] == True or thresh_dict['use_sounding'] == False:
        thresh_dict.setdefault('qc_height', 4.4)
        qc_height = thresh_dict['qc_height'] * 1000
        gatefilter.exclude_all()
        gatefilter.include_below('HEIGHT', qc_height)
        if thresh_dict['do_rh'] == True: gatefilter.exclude_below('RH', rh_thresh)
        if thresh_dict['do_zdr'] == True: gatefilter.exclude_outside('DR', dr_min, dr_max)
        if thresh_dict['do_ap'] == True: gatefilter.exclude_not_equal('AP', sec)
        gatefilter.include_above('HEIGHT', qc_height)
    elif thresh_dict['use_sounding'] == True:
        gatefilter.exclude_all()
        gatefilter.include_above('TEMP', 3.0)
        if thresh_dict['do_rh'] == True: gatefilter.exclude_below('RH', rh_thresh)
        if thresh_dict['do_zdr'] == True: gatefilter.exclude_outside('DR', dr_min, dr_max)
        if thresh_dict['do_ap'] == True: gatefilter.exclude_not_equal('AP', sec)
        gatefilter.include_below('TEMP', 3.1)   

    # Apply gate filters to radar
    for fld in radar.fields:
        nf = deepcopy(radar.fields[fld])
        nf['data'] = np.ma.masked_where(gatefilter.gate_excluded, nf['data'])
        radar.add_field(fld, nf, replace_existing=True)
    return radar



# ***************************************************************************************
def threshold_qc_calfields(radar, thresh_dict):

    """
    Use gatefilter to apply QC by looking at thresholds of calculated field values.
    Written by: Jason L. Pippitt, NASA/GSFC/SSAI

    Parameters
    ----------
    radar : radar object
            the radar object where the data is

    Thresholds for qc'ing data: kdp_min, kdp_max, sd_thresh

    Returns
    -------
    radar: QC'd radar with gatefilters applied.

    """
    
    print("    Thresholding Cal Fields...")

    # Declare thresholds for DP fields

    kdp_min    = thresh_dict['kdp_min']
    kdp_max    = thresh_dict['kdp_max']
    sd_thresh  = thresh_dict['sd_thresh']
    ph_thresh = thresh_dict['ph_thresh']
    sec = 1
    cos = 0

    # Create a pyart gatefilter from radar
    secfilter = pyart.filters.GateFilter(radar)
    gatefilter_cal = pyart.filters.GateFilter(radar)

    # Apply sector thresholds regardless of temp 
    if thresh_dict['do_sd_sector'] == True: secfilter.exclude_not_equal('SECSD', sec)
    if thresh_dict['do_ph_sector'] == True: secfilter.exclude_not_equal('SECPH', sec)
    
    # Apply gate filters to radar
    for fld in radar.fields:
        nf = deepcopy(radar.fields[fld])
        nf['data'] = np.ma.masked_where(secfilter.gate_excluded, nf['data'])
        radar.add_field(fld, nf, replace_existing=True)

    # Call gatefliters for calculated fields based on temperature or beam height
    if thresh_dict['use_qc_height'] == True or thresh_dict['use_sounding'] == False:
        thresh_dict.setdefault('qc_height', 4.4)
        qc_height = thresh_dict['qc_height'] * 1000
        gatefilter_cal.exclude_all()
        gatefilter_cal.include_below('HEIGHT', qc_height)
        if thresh_dict['do_sd'] == True: gatefilter_cal.exclude_above('SD', sd_thresh)
        if thresh_dict['do_kdp'] == True: gatefilter_cal.exclude_outside('KD', kdp_min, kdp_max)
        if thresh_dict['do_ph'] == True: gatefilter_cal.exclude_below('PH', ph_thresh)
        gatefilter_cal.include_above('HEIGHT', qc_height)
    elif thresh_dict['use_sounding'] == True:
        gatefilter_cal.exclude_all()
        gatefilter_cal.include_above('TEMP', 3.0)
        if thresh_dict['do_sd'] == True: gatefilter_cal.exclude_above('SD', sd_thresh)
        if thresh_dict['do_kdp'] == True: gatefilter_cal.exclude_outside('KD', kdp_min, kdp_max)
        if thresh_dict['do_ph'] == True: gatefilter_cal.exclude_below('PH', ph_thresh)
        gatefilter_cal.include_below('TEMP', 3.1)    

    # Apply gate filters to radar
    for fld in radar.fields:
        nf = deepcopy(radar.fields[fld])
        nf['data'] = np.ma.masked_where(gatefilter_cal.gate_excluded, nf['data'])
        radar.add_field(fld, nf, replace_existing=True)

    print("QC Complete")
    print()
    return radar

# ***************************************************************************************

def rename_fields_in_radar(radar):

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
    print() 
    print("Renaming radar fields...")
    if 'PHIDP2' in radar.fields.keys():
        old_fields = ['DBZ2', 'VEL2', 'WIDTH2', 'ZDR2', 'KDP2', 'PHIDP2', 'RHOHV2', 'SQI2']
        new_fields = ['DZ',     'VR',    'SW',   'DR',   'KD',   'PH',     'RH',     'SQ']
    elif 'DBZ' in radar.fields.keys():
        old_fields = ['DBZ', 'VEL', 'WIDTH', 'ZDR', 'KDP', 'PHIDP', 'SQI', 'RHOHV']
        new_fields = ['DZ',  'VR',   'SW',   'DR',  'KD',   'PH',    'SQ',    'RH']
    elif 'DBZ2' in radar.fields.keys():
        old_fields = ['DBZ2', 'VEL2', 'WIDTH2', 'ZDR2', 'KDP2', 'PHIDP2', 'RHOHV2',  'SQI2']
        new_fields = ['DZ',   'VR',   'SW',     'DR',   'KD',   'PH',     'RH',      'SQ']
    elif 'reflectivity' in radar.fields.keys():
        old_fields = ['differential_phase', 'velocity', 'spectrum_width', 'reflectivity', 'differential_reflectivity', 'cross_correlation_ratio']
        new_fields = ['PH',     'VR',     'SW',   'DZ',   'DR', 'RH']
    elif 'REF' in radar.fields.keys():
        old_fields = ['SW', 'PHI', 'ZDR', 'REF', 'VEL', 'RHO']
        new_fields = ['SW', 'PH' , 'DR' , 'DZ' , 'VR' , 'RH' ]
    elif 'DZ' in radar.fields.keys():
        old_fields = []
        new_fields = []

    # Change names of old fields to new fields using pop
    nl = len(old_fields)
    for i in range(0,nl):
        old_field = old_fields[i]
        new_field = new_fields[i]
        radar.fields[new_field] = radar.fields.pop(old_field)
        i += 1  

    # Add Corrected Reflectivity field
    cz = radar.fields['DZ']['data'].copy()
    gu.add_field_to_radar_object(cz, radar, field_name='CZ', 
                                 units=' ',
                                 long_name='Corrected Reflectivity', 
                                 standard_name='Corrected Reflectivity', 
                                 dz_field='DZ') 
    return radar

# ***************************************************************************************

def remove_fields_from_radar(radar):

    """
    Remove fields from radar that are not needed.
    Written by: David B. Wolff, NASA/WFF

    Parameters:
    -----------
    radar: pyart radar object
    drop_fields: List of fields to drop from radar object

    Return:
    -------
    radar: Pruned radar

    """    
    drop_fields = []
    if 'TEMP' in radar.fields.keys(): drop_fields.extend(['TEMP'])
    if 'HEIGHT' in radar.fields.keys(): drop_fields.extend(['HEIGHT'])
    if 'PHIDPB' in radar.fields.keys(): drop_fields.extend(['PHIDPB'])
    if 'DBTV16' in radar.fields.keys(): drop_fields.extend(['DBTV16'])
    if 'DBZV16' in radar.fields.keys(): drop_fields.extend(['DBZV16'])
    if 'SNR16' in radar.fields.keys(): drop_fields.extend(['SNR16'])
    if 'DBT2' in radar.fields.keys(): drop_fields.extend(['DBT2'])
    if 'COS' in radar.fields.keys(): drop_fields.extend(['COS'])
    if 'SEC' in radar.fields.keys(): drop_fields.extend(['SEC'])
    if 'SECRH' in radar.fields.keys(): drop_fields.extend(['SECRH'])
    if 'SECSD' in radar.fields.keys(): drop_fields.extend(['SECSD'])
    if 'SECPH' in radar.fields.keys(): drop_fields.extend(['SECPH'])
    if 'AP' in radar.fields.keys(): drop_fields.extend(['AP'])
    if 'WSR' in radar.fields.keys(): drop_fields.extend(['WSR'])
    
    # Remove fields we no longer need.
    for field in drop_fields:
        radar.fields.pop(field)
    return radar

# ***************************************************************************************

def output_cf(radar, thresh_dict):

    # Outputs CF radial file

    # Get datetime and site from radar

    if 'site_name' in radar.metadata.keys():
        site = radar.metadata['site_name'].upper()
    elif 'instrument_name' in radar.metadata.keys():
        if isinstance(radar.metadata['instrument_name'], bytes):
            site = radar.metadata['instrument_name'].decode().upper()
        else:
            site = radar.metadata['instrument_name'].upper()
    else:
        site=''

    radar_DT = pyart.util.datetime_from_radar(radar)   
    month = str(radar_DT.month).zfill(2)
    day = str(radar_DT.day).zfill(2)
    year = str(radar_DT.year).zfill(4)
    hh = str(radar_DT.hour).zfill(2)
    mm = str(radar_DT.minute).zfill(2)
    ss = str(radar_DT.second).zfill(2)

    #Declare output dir
    scantype = radar.scan_type.upper()
    out_dir = thresh_dict['cf_dir']
    os.makedirs(out_dir, exist_ok=True)
   
    out_file = out_dir + '/' + site + '_' + year + '_' + month + day + '_' + hh + mm + ss + '_' + scantype + '.cf'
    
    pyart.io.write_cfradial(out_file,radar)
    
    print('Output cfRadial --> ' + out_file)
    print()

# ***************************************************************************************

def get_vcp(radar):

    """ Return a list of elevation angles representative of each sweep.
    These are the median of the elevation angles in each sweep, which are
    more likely to be identical than the mean due to change of elevation angle
    at the beginning and end of each sweep.

    Written by:  Eric Bruning
    """
    vcp = [np.median(el_this_sweep) for el_this_sweep in radar.iter_elevation()]
    return np.asarray(vcp, dtype=radar.elevation['data'].dtype)

# ***************************************************************************************

def unique_sweeps_by_elevation_angle(radar, tol=0.09):
    """ Returns the sweep indices that correspond to unique
    elevation angles, for use in extract_sweeps.
    
    The default is a tolerance of 0.05 deg. 

    Written by:  Eric Bruning
    """
    vcp = get_vcp(radar)
    close_enough = (vcp/tol).astype('int32')
    print(close_enough)
    unq_el, unq_el_idx, o_idx = np.unique(close_enough, return_index=True, return_inverse=True)
    u,indices = np.unique(close_enough, return_inverse = True)
    print(indices)
    print(unq_el)
    print(unq_el_idx)
    print(o_idx)
    return unq_el_idx

# ***************************************************************************************

def convert_to_cf(radar, file):

    print()
    print(radar.metadata['original_container'])
    print('    Converting data to cfRadial to organize split cuts and remove MRLE scans.')
    print()
    subprocess.call(shlex.split(f"./convert_to_cf.csh {file}"))
    cf_file = file + '.cf'
    radar = pyart.io.read(cf_file, file_field_names=True)
    os.remove(cf_file)

    return radar

# ***************************************************************************************

def get_default_thresh_dict():

    default_thresh_dict = {'do_dbz': True, 'dbz_thresh': 5.0,
                           'do_rh': True, 'rh_thresh': 0.72,
                           'do_zdr': True, 'dr_min': -6.0, 'dr_max': 4.0, 
                           'do_kdp': False, 'kdp_min': -2.0, 'kdp_max': 7.0, 
                           'do_sq': False, 'sq_thresh': 0.45, 
                           'do_sd': True, 'sd_thresh': 25.0, 
                           'do_ph': False, 'ph_thresh': 80.0, 
                           'do_ap': False, 'ap_dbz': 45, 'ap_zdr': 3, 
                           'do_insect': False, 
                           'do_despeckle': True, 
                           'do_cos': False, 'coshmin': 0, 'coshmax': None,
                           'cosrmin': 0, 'cosrmax': 5, 
                           'cosazmin': 0, 'cosazmax': 360, 
                           'coselmin': 0, 'coselmax': 20.0, 
                           'do_sector': False, 'sechmin': 0, 'sechmax': None, 
                           'secrmin': 0, 'secrmax': 150, 
                           'secazmin': 160, 'secazmax': 165, 
                           'secelmin': 0, 'secelmax': 20.0, 
                           'do_rh_sector': False, 'rhhmin': 0, 'rhhmax': None,
                           'rhrmin': 0, 'rhrmax': 20, 
                           'rhazmin': 0, 'rhazmax': 360, 
                           'rhelmin': 0, 'rhelmax': 7.0, 'rh_sec': 0.92, 
                           'do_sd_sector': False, 'sdhmin': 0, 'sdhmax': None, 
                           'sdrmin': 0, 'sdrmax': 20, 
                           'sdazmin': 0, 'sdazmax': 360, 
                           'sdelmin': 0, 'sdelmax': 7.0, 'sd_sec': 8.0, 
                           'do_ph_sector': False, 'phhmin': 0, 'phhmax': None,
                           'phrmin': 0, 'phrmax': 150, 
                           'phazmin': 160, 'phazmax': 165, 
                           'phelmin': 0, 'phelmax': 20.0, 'ph_sec': 80.0, 
                           'apply_cal': False, 'ref_cal': 0.7, 'zdr_cal': 0.1, 
                           'use_qc_height': True, 'qc_height': 4.4, 
                           'cf_dir': './cf',
                           'plot_images': True, 'max_range': 200, 'max_height': 15,
                           'sweeps_to_plot': [0], 
                           'plot_single': False, 
                           'fields_to_plot': ['DZ', 'CZ', 'VR', 'DR', 'KD', 'PH', 'RH', 'SD'], 
                           'plot_dir': './plots', 
                           'use_sounding': False, 'sounding_dir': './sounding'}

    return default_thresh_dict

# ***************************************************************************************

def get_default_product_dict():

    default_product_dict = {'cf_dir': './cf/',
                            'max_range': 150, 
                            'max_height': 10,
                            'sweeps_to_plot': [0],
                            'output_cf': True,
                            'output_fields': ['DZ', 'CZ', 'VR', 'DR', 'KD', 'PH', 'RH', 'SD', 'SQ', 'FH', 'RC', 'DM', 'NW'],
                            'plot_images': True,
                            'plot_single': False,
                            'fields_to_plot': ['CZ','RC','FH', 'DM', 'NW'],
                            'plot_dir': './plots/',
                            'use_sounding': True,
                            'sounding_dir': './soundings/'}

    return default_product_dict

# ***************************************************************************************
