# ***************************************************************************************
'''
Python based Quality Control utilizing PyArt 

Developed by the NASA GPM-GV group
V0.3 - 02/19/2021 - update by Jason Pippitt NASA/GSFC/SSAI
'''
# ***************************************************************************************

import numpy as np
from copy import deepcopy
import pyart
import os
import subprocess
import shlex
from gvradar import common as cm
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain,
                            csu_dsd, csu_kdp, csu_misc, fundamentals)

# ***************************************************************************************

def calibrate(self):

    """
    Applies calibration adjustments to DBZ and ZDR fields.

    returns:  radar with calibrated fields.

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    print('', 'Calibrating Reflectivity and ZDR...', sep='\n')

    ref_cal = self.ref_cal
    zdr_cal = self.zdr_cal

    fill_value = -32767.0
    #Calibrate Reflectivity field
    ref_field = self.radar.fields[self.ref_field_name]['data'].copy()
    corr_dbz = pyart.correct.correct_bias(self.radar, bias=ref_cal, field_name=self.ref_field_name)
    corz_dict = {'data': corr_dbz['data'], 'units': '', 'long_name': 'CZ',
                 '_FillValue': fill_value, 'standard_name': 'CZ'}
    self.radar.add_field('CZ', corz_dict, replace_existing=True)
    
    #Calibrate ZDR field
    zdr_field = self.radar.fields[self.zdr_field_name]['data'].copy()
    corr_zdr = pyart.correct.correct_bias(self.radar, bias=zdr_cal, field_name=self.zdr_field_name)
    corzdr_dict = {'data': corr_zdr['data'], 'units': '', 'long_name': 'DR',
                 '_FillValue': fill_value, 'standard_name': 'DR'}
    self.radar.add_field('DR', corzdr_dict, replace_existing=True)

    return self.radar

# ***************************************************************************************

def csu_filters(self):

    dz = cm.extract_unmasked_data(self.radar, self.ref_field_name)
    zdr = cm.extract_unmasked_data(self.radar, self.zdr_field_name)

    if self.do_insect == True:
        insect_mask = csu_misc.insect_filter(dz, zdr)
        bad = -32767.0

        for fld in self.radar.fields:
            nf = cm.extract_unmasked_data(self.radar, fld)
            nf_insect = 1.0 * nf
            nf_insect[insect_mask] = bad 
            cm.add_field_to_radar_object(nf_insect, self.radar, field_name=fld, units='', 
                                     long_name=fld,
                                     standard_name=fld, 
                                     dz_field=self.ref_field_name)
    
    if self.do_despeckle == True:
        mask_ds = csu_misc.despeckle(dz, ngates=4)
        bad = -32767.0

        for fld in self.radar.fields:
            nf = cm.extract_unmasked_data(self.radar, fld)
            nf_ds = 1.0 * nf
            nf_ds[mask_ds] = bad 
            cm.add_field_to_radar_object(nf_ds, self.radar, field_name=fld, units='', 
                                         long_name=fld,
                                         standard_name=fld, 
                                         dz_field=self.ref_field_name)

    return self.radar  

# ***************************************************************************************

def threshold_qc_dpfields(self):

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
    
    print("Begin Quality Control:  ", "    Thresholding DP Fields...", sep='\n')

    # Declare thresholds for DP fields
    sec = 1
    cos = 0

    # Create a pyart gatefilters from radar
    dbzfilter = pyart.filters.GateFilter(self.radar)
    gatefilter = pyart.filters.GateFilter(self.radar)

    # Apply dbz, sector, and SQI thresholds regardless of Temp 
    if self.do_dbz == True: dbzfilter.exclude_below('CZ', self.dbz_thresh)
    if self.do_sector == True: dbzfilter.exclude_not_equal('SEC', cos)
    if self.do_rh_sector == True: dbzfilter.exclude_not_equal('SECRH', sec) 
    if self.do_cos == True: dbzfilter.exclude_not_equal('COS', cos)
    if self.do_sq == True: dbzfilter.exclude_below('SQ', self.sq_thresh)
    if self.radar.metadata['instrument_name'] == 'WSR-88D': dbzfilter.exclude_not_equal('WSR', cos)

    # Apply gate filters to radar
    for fld in self.radar.fields:
        nf = deepcopy(self.radar.fields[fld])
        nf['data'] = np.ma.masked_where(dbzfilter.gate_excluded, nf['data'])
        self.radar.add_field(fld, nf, replace_existing=True) 

    # Create AP filter variables
    if self.do_ap == True:
        dz = self.radar.fields['CZ']['data'].copy()
        dr = self.radar.fields['DR']['data'].copy()
        ap = np.ones(dz.shape)
        dz_lt = np.ma.where(dz <= 45 , 1, 0)
        dr_lt = np.ma.where(dr >= 3 , 1, 0)
        ap_t = np.logical_and(dz_lt == 1 , dr_lt == 1)
        ap[ap_t] = 0
        cm.add_field_to_radar_object(ap, self.radar, field_name='AP', 
                                     units='0 = Z < 0, 1 = Z >= 0',
                                     long_name='AP Mask', 
                                     standard_name='AP Mask', 
                                     dz_field=self.ref_field_name)

    # Call gatefliters for each field based on temperature or beam height
    if self.use_qc_height == True or self.use_sounding == False:
        qc_height = self.qc_height * 1000
        gatefilter.exclude_all()
        gatefilter.include_below('HEIGHT', qc_height)
        if self.do_rh == True: gatefilter.exclude_below('RH', self.rh_thresh)
        if self.do_zdr == True: gatefilter.exclude_outside('DR', self.dr_min, self.dr_max)
        if self.do_ap == True: gatefilter.exclude_not_equal('AP', sec)
        gatefilter.include_above('HEIGHT', qc_height)
    elif self.use_sounding == True:
        gatefilter.exclude_all()
        gatefilter.include_above('TEMP', 3.0)
        if self.do_rh == True: gatefilter.exclude_below('RH', self.rh_thresh)
        if self.do_zdr == True: gatefilter.exclude_outside('DR', self.dr_min, self.dr_max)
        if self.do_ap == True: gatefilter.exclude_not_equal('AP', sec)
        gatefilter.include_below('TEMP', 3.1)   

    # Apply gate filters to radar
    for fld in self.radar.fields:
        nf = deepcopy(self.radar.fields[fld])
        nf['data'] = np.ma.masked_where(gatefilter.gate_excluded, nf['data'])
        self.radar.add_field(fld, nf, replace_existing=True)
    return self.radar



# ***************************************************************************************
def threshold_qc_calfields(self):

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
    sec = 1

    # Create a pyart gatefilter from radar
    secfilter = pyart.filters.GateFilter(self.radar)
    gatefilter_cal = pyart.filters.GateFilter(self.radar)

    # Apply sector thresholds regardless of temp 
    if self.do_sd_sector == True: secfilter.exclude_not_equal('SECSD', sec)
    if self.do_ph_sector == True: secfilter.exclude_not_equal('SECPH', sec)
    
    # Apply gate filters to radar
    for fld in self.radar.fields:
        nf = deepcopy(self.radar.fields[fld])
        nf['data'] = np.ma.masked_where(secfilter.gate_excluded, nf['data'])
        self.radar.add_field(fld, nf, replace_existing=True)

    # Call gatefliters for calculated fields based on temperature or beam height
    if self.use_qc_height == True or self.use_sounding == False:
        qc_height = self.qc_height * 1000
        gatefilter_cal.exclude_all()
        gatefilter_cal.include_below('HEIGHT', qc_height)
        if self.do_sd == True: gatefilter_cal.exclude_above('SD', self.sd_thresh)
        if self.do_kdp == True: gatefilter_cal.exclude_outside('KD', self.kdp_min, self.kdp_max)
        if self.do_ph == True: gatefilter_cal.exclude_below('PH', self.ph_thresh)
        gatefilter_cal.include_above('HEIGHT', qc_height)
    elif self.use_sounding == True:
        gatefilter_cal.exclude_all()
        gatefilter_cal.include_above('TEMP', 3.0)
        if self.do_sd == True: gatefilter_cal.exclude_above('SD', self.sd_thresh)
        if self.do_kdp == True: gatefilter_cal.exclude_outside('KD', self.kdp_min, self.kdp_max)
        if self.do_ph == True: gatefilter_cal.exclude_below('PH', self.ph_thresh)
        gatefilter_cal.include_below('TEMP', 3.1)    

    # Apply gate filters to radar
    for fld in self.radar.fields:
        nf = deepcopy(self.radar.fields[fld])
        nf['data'] = np.ma.masked_where(gatefilter_cal.gate_excluded, nf['data'])
        self.radar.add_field(fld, nf, replace_existing=True)

    print("QC Complete.", '', sep='\n')

    return self.radar

# ***************************************************************************************

def mask_cone_of_silence(self):

    """
    filter out any data inside the cone of silence

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    self : dict
            a dictionary defining the region of interest

    Returns
    -------
    cos_flag : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': self.coshmin, 'hmax': self.coshmax,
              'rmin': self.cosrmin * 1000, 'rmax':  self.cosrmax * 1000,
              'azmin': self.cosazmin, 'azmax': self.cosazmax,
              'elmin': self.coselmin, 'elmax': self.coselmax}
    
    cos_flag = np.ma.ones((self.radar.nrays, self.radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        cos_flag[self.radar.gate_altitude['data'] < sector['hmin']] = 0
    if sector['hmax'] is not None:
        cos_flag[self.radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits

    if sector['rmin'] is not None:
        cos_flag[:, self.radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        cos_flag[:, self.radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        cos_flag[self.radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        cos_flag[self.radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            cos_flag[self.radar.azimuth['data'] < sector['azmin'], :] = 0
            cos_flag[self.radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            cos_flag[np.logical_and(
            self.radar.azimuth['data'] < sector['azmin'],
            self.radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        cos_flag[self.radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        cos_flag[self.radar.azimuth['data'] > sector['azmax'], :] = 0

    cos_field = cos_flag
    cm.add_field_to_radar_object(cos_field, self.radar, field_name='COS', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='COS',
                                 standard_name='COS', 
                                 dz_field=self.ref_field_name)
    return self.radar

# ***************************************************************************************

def mask_88D_200(self):

    """
    filter out any data outside 200 KM

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    self : dict
            a dictionary defining the region of interest

    Returns
    -------
    WSR_flag : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': 0, 'hmax': None,
              'rmin': 200 * 1000, 'rmax':  300 * 1000,
              'azmin': 0, 'azmax': 360,
              'elmin': 0, 'elmax': None}
    
    WSR_flag = np.ma.ones((self.radar.nrays, self.radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        WSR_flag[self.radar.gate_altitude['data'] < sector['hmin']] = 0
    if sector['hmax'] is not None:
        WSR_flag[self.radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits

    if sector['rmin'] is not None:
        WSR_flag[:, self.radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        WSR_flag[:, self.radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        WSR_flag[self.radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        WSR_flag[self.radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            WSR_flag[self.radar.azimuth['data'] < sector['azmin'], :] = 0
            WSR_flag[self.radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            WSR_flag[np.logical_and(
            self.radar.azimuth['data'] < sector['azmin'],
            self.radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        WSR_flag[self.radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        WSR_flag[self.radar.azimuth['data'] > sector['azmax'], :] = 0

    WSR_field = WSR_flag
    cm.add_field_to_radar_object(WSR_field, self.radar, field_name='WSR', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='WSR',
                                 standard_name='WSR', 
                                 dz_field=self.ref_field_name)
    return self.radar

# ***************************************************************************************

def sector_wipeout(self):
    
    """
    filter out any data inside the region of interest defined by sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    self : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_wipeout : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': self.sechmin, 'hmax': self.sechmax,
	      'rmin':  self.secrmin * 1000, 'rmax':  self.secrmax * 1000,
              'azmin': self.secazmin, 'azmax': self.secazmax,
	      'elmin': self.secelmin, 'elmax': self.secelmax}

    sector_wipeout = np.ma.ones((self.radar.nrays, self.radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[self.radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[self.radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, self.radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, self.radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[self.radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[self.radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[self.radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[self.radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            self.radar.azimuth['data'] < sector['azmin'],
            self.radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[self.radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[self.radar.azimuth['data'] > sector['azmax'], :] = 0

    sector_field = sector_wipeout
    cm.add_field_to_radar_object(sector_field, self.radar, field_name='SEC', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='Sector Mask', 
                                 standard_name='Sector Mask', 
                                 dz_field=self.ref_field_name)

    return self.radar

# ***************************************************************************************

def rh_sector(self):
    
    """
    filter out any data inside the region of interest that is < rh_sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    self : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_rh : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': self.rhhmin, 'hmax': self.rhhmax,
	      'rmin':  self.rhrmin * 1000, 'rmax':  self.rhrmax * 1000,
              'azmin': self.rhazmin, 'azmax': self.rhazmax,
	      'elmin': self.rhelmin, 'elmax': self.rhelmax,
              'rh_sec': self.rh_sec}

    sector_wipeout = np.ma.ones((self.radar.nrays, self.radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[self.radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[self.radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, self.radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, self.radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[self.radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[self.radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[self.radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[self.radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            self.radar.azimuth['data'] < sector['azmin'],
            self.radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[self.radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[self.radar.azimuth['data'] > sector['azmax'], :] = 0
    
    rh = self.radar.fields['RH']['data'].copy()
    sector_r = np.ones(rh.shape)
    rh_sec = sector['rh_sec']
    rh_lt = np.ma.where(rh < rh_sec , 1, 0)
    sec_f = np.logical_and(rh_lt == 1 , sector_wipeout == 1)
    sector_r[sec_f] = 0

    sector_rh = sector_r
    cm.add_field_to_radar_object(sector_rh, self.radar, field_name='SECRH', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='Sector RH Mask', 
                                 standard_name='Sector RH Mask', 
                                 dz_field=self.ref_field_name)
     
    return self.radar

# ***************************************************************************************

def sd_sector(self):

    """
    filter out any data inside the region of interest that is < sd_sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    self : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_sd : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    sector = {'hmin': self.sdhmin, 'hmax': self.sdhmax,
	      'rmin':  self.sdrmin * 1000, 'rmax':  self.sdrmax * 1000,
              'azmin': self.sdazmin, 'azmax': self.sdazmax,
	      'elmin': self.sdelmin, 'elmax': self.sdelmax,
              'sd_sec': self.sd_sec}

    sector_wipeout = np.ma.ones((self.radar.nrays, self.radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[self.radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[self.radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, self.radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, self.radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[self.radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[self.radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[self.radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[self.radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            self.radar.azimuth['data'] < sector['azmin'],
            self.radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[self.radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[self.radar.azimuth['data'] > sector['azmax'], :] = 0
    
    sd = self.radar.fields['SD']['data'].copy()
    sector_s = np.ones(sd.shape)
    sd_sec = sector['sd_sec']
    sd_lt = np.ma.where(sd > sd_sec , 1, 0)
    sec_f = np.logical_and(sd_lt == 1 , sector_wipeout == 1)
    sector_s[sec_f] = 0

    sector_sd = sector_s
    cm.add_field_to_radar_object(sector_sd, self.radar, field_name='SECSD', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='Sector SD Mask', 
                                 standard_name='Sector SD Mask', 
                                 dz_field=self.ref_field_name)

    return self.radar

# ***************************************************************************************

def ph_sector(self):
    
    """
    filter out any data inside the region of interest that is < ph_sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    self : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_ph : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': self.phhmin, 'hmax': self.phhmax,
	      'rmin':  self.phrmin * 1000, 'rmax':  self.phrmax * 1000,
              'azmin': self.phazmin, 'azmax': self.phazmax,
	      'elmin': self.phelmin, 'elmax': self.phelmax,
              'ph_sec': self.ph_sec}

    sector_wipeout = np.ma.ones((self.radar.nrays, self.radar.ngates), dtype=int)

    # check for altitude limits
    if sector['hmin'] is not None:
        sector_wipeout[self.radar.gate_altitude['data'] < sector['hmin']] = 0

    if sector['hmax'] is not None:
        sector_wipeout[self.radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits
    if sector['rmin'] is not None:
        sector_wipeout[:, self.radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        sector_wipeout[:, self.radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        sector_wipeout[self.radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        sector_wipeout[self.radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            sector_wipeout[self.radar.azimuth['data'] < sector['azmin'], :] = 0
            sector_wipeout[self.radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            sector_wipeout[np.logical_and(
            self.radar.azimuth['data'] < sector['azmin'],
            self.radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        sector_wipeout[self.radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        sector_wipeout[self.radar.azimuth['data'] > sector['azmax'], :] = 0
    
    ph = self.radar.fields['PHM']['data'].copy()
    sector_p = np.ones(ph.shape)
    ph_sec = sector['ph_sec']
    ph_lt = np.ma.where(ph < ph_sec , 1, 0)
    sec_f = np.logical_and(ph_lt == 1 , sector_wipeout == 1)
    sector_p[sec_f] = 0

    sector_ph = sector_p
    cm.add_field_to_radar_object(sector_ph, self.radar, field_name='SECPH', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='Sector PH Mask', 
                                 standard_name='Sector PH Mask', 
                                 dz_field=self.ref_field_name)

    return self.radar

# ***************************************************************************************

def unfold_phidp(self):

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

    # Copy current PhiDP field to phm_field
    phm_field = self.radar.fields[self.phi_field_name]['data'].copy()

    # Get gate spacing info and determine start gate for unfolding
    # Start at 5 km from radar to avoid clutter gates with bad phase 
    gate_spacing = self.radar.range['meters_between_gates']
    start_gate = int(FIRST_GATE / gate_spacing)
    nsweeps = self.radar.nsweeps
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
    self.radar = cm.add_field_to_radar_object(phm_field, self.radar, field_name='PH', 
		units='deg',
		long_name=' Differential Phase (Marks)',
		standard_name='Specific Differential Phase (Marks)',
		dz_field=self.ref_field_name)
    
    return self.radar

# ***************************************************************************************

def calculate_kdp(self):

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

#    DZ = cm.extract_unmasked_data(self.radar, self.ref_field_name)
#    DP = cm.extract_unmasked_data(self.radar, self.phi_field_name)
    DZ = self.radar.fields[self.ref_field_name]['data'].copy()
    DP = self.radar.fields[self.phi_field_name]['data'].copy()

    # Range needs to be supplied as a variable, with same shape as DZ
    rng2d, az2d = np.meshgrid(self.radar.range['data'], self.radar.azimuth['data'])
    gate_spacing = self.radar.range['meters_between_gates']

    KDPB, PHIDPB, STDPHIB = csu_kdp.calc_kdp_bringi(dp=DP, dz=DZ, rng=rng2d/1000.0, 
                                                    thsd=25, gs=gate_spacing, window=5)

    self.radar = cm.add_field_to_radar_object(KDPB, self.radar, field_name='KD', 
		units='deg/km',
		long_name='Specific Differential Phase (Bringi)',
		standard_name='Specific Differential Phase (Bringi)',
		dz_field=self.ref_field_name)

    self.radar = cm.add_field_to_radar_object(PHIDPB, self.radar, 
		field_name='PHIDPB', units='deg',
		long_name='Differential Phase (Bringi)',
		standard_name='Differential Phase (Bringi)',
		dz_field=self.ref_field_name)

    self.radar = cm.add_field_to_radar_object(STDPHIB, self.radar, 
		field_name='SD', units='deg',
		long_name='STD Differential Phase (Bringi)',
		standard_name='STD Differential Phase (Bringi)',
		dz_field=self.ref_field_name)

    return self.radar

# ***************************************************************************************

def get_vcp(self):

    """ Return a list of elevation angles representative of each sweep.
    These are the median of the elevation angles in each sweep, which are
    more likely to be identical than the mean due to change of elevation angle
    at the beginning and end of each sweep.

    Written by:  Eric Bruning
    """
    vcp = [np.median(el_this_sweep) for el_this_sweep in self.radar.iter_elevation()]
    return np.asarray(vcp, dtype=self.radar.elevation['data'].dtype)

# ***************************************************************************************

def unique_sweeps_by_elevation_angle(self):
    """ Returns the sweep indices that correspond to unique
    elevation angles, for use in extract_sweeps.
    
    The default is a tolerance of 0.05 deg. 

    Written by:  Eric Bruning
    """
    tol=0.09
    vcp = get_vcp(self.radar)
    close_enough = (vcp/tol).astype('int32')
    unq_el, unq_el_idx, o_idx = np.unique(close_enough, return_index=True, return_inverse=True)
    u,indices = np.unique(close_enough, return_inverse = True)
    return unq_el_idx

# ***************************************************************************************

def convert_to_cf(self):

    print('', self.radar.metadata['original_container'], 
          '    Converting data to cfRadial to organize split cuts and remove MRLE scans.',
          '', sep='\n')
    subprocess.call(shlex.split(f"./convert_to_cf.csh {self.file}"))
    cf_file = self.file + '.cf'
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
                           'apply_cal': False, 'ref_cal': 0.2, 'zdr_cal': 0.0, 
                           'use_qc_height': True, 'qc_height': 4.4, 
                           'cf_dir': './cf',
                           'plot_images': True, 'max_range': 150, 'max_height': 15,
                           'sweeps_to_plot': [0], 
                           'plot_single': False, 
                           'fields_to_plot': ['DZ', 'CZ', 'VR', 'DR', 'KD', 'PH', 'RH', 'SD'],
                           'plot_dir': './plots/', 'add_logos': True,
                           'use_sounding': False, 'sounding_type': 'ruc_archive', 'sounding_dir': './sounding/'}

    return default_thresh_dict

# ***************************************************************************************

def remove_fields_from_radar(self):

    """
    Remove fields from radar that are not needed.  

    """    

    print("Removing unwanted fields...")

    drop_fields = []
    if 'TEMP' in self.radar.fields.keys(): drop_fields.extend(['TEMP'])
    if 'HEIGHT' in self.radar.fields.keys(): drop_fields.extend(['HEIGHT'])
    if 'PHIDPB' in self.radar.fields.keys(): drop_fields.extend(['PHIDPB'])
    if 'DBTV16' in self.radar.fields.keys(): drop_fields.extend(['DBTV16'])
    if 'DBZV16' in self.radar.fields.keys(): drop_fields.extend(['DBZV16'])
    if 'SNR16' in self.radar.fields.keys(): drop_fields.extend(['SNR16'])
    if 'DBT2' in self.radar.fields.keys(): drop_fields.extend(['DBT2'])
    if 'COS' in self.radar.fields.keys(): drop_fields.extend(['COS'])
    if 'SEC' in self.radar.fields.keys(): drop_fields.extend(['SEC'])
    if 'SECRH' in self.radar.fields.keys(): drop_fields.extend(['SECRH'])
    if 'SECSD' in self.radar.fields.keys(): drop_fields.extend(['SECSD'])
    if 'SECPH' in self.radar.fields.keys(): drop_fields.extend(['SECPH'])
    if 'AP' in self.radar.fields.keys(): drop_fields.extend(['AP'])
    if 'WSR' in self.radar.fields.keys(): drop_fields.extend(['WSR'])
    
    # Remove fields we no longer need.
    for field in drop_fields:
        self.radar.fields.pop(field)

    print('', "FINAL FIELDS -->  ", self.radar.fields.keys(), '', sep='\n')

    return self.radar

# ***************************************************************************************