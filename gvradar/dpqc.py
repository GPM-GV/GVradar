# ***************************************************************************************
'''
Python based Quality Control utilizing PyArt 

Developed by the NASA GPM-GV group
V0.5 - 12/06/2021 - update by Jason Pippitt NASA/GSFC/SSAI
V1.0 - 11/01/2022 - update by Jason Pippitt NASA/GSFC/SSAI
V1.5 - 02/02/2024 - update by Jason Pippitt NASA/GSFC/SSAI
'''
# ***************************************************************************************

import numpy as np
from scipy import ndimage
from copy import deepcopy
import pyart
import traceback
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
    print('Zcal:  ',self.ref_cal, 'ZDRcal:  ',self.zdr_cal)

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
        insect_mask = csu_misc.insect_filter(dz, zdr, height=self.qc_height, bad = -32767.0)
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
        size = self.speck_size
        if self.pyart_speck:
            print("    pyART despeckle...")
            speckle = pyart.correct.despeckle_field(self.radar, 'CZ', label_dict=None, threshold=0, size=size, gatefilter=None, delta=5.0)

            # Apply gate filters to radar
            for fld in self.radar.fields:
                nf = deepcopy(self.radar.fields[fld])
                nf['data'] = np.ma.masked_where(speckle.gate_excluded, nf['data'])
                self.radar.add_field(fld, nf, replace_existing=True)  
        else:
            print("    CSU despeckle...")
            mask_ds = csu_misc.despeckle(dz, bad = -32767.0, ngates=size)
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

def apply_SW_mask(self):

    self.radar.fields['CZ']['data'] = np.ma.masked_where(
        self.radar.fields['SW']['data'].mask, self.radar.fields["CZ"]['data'])
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
    
    print("", "Begin Quality Control:  ", "    Thresholding DP Fields...", sep='\n')

    # Declare thresholds for DP fields
    sec = 1
    cos = 0
    
    # Create a pyart gatefilters from radar
    dbzfilter = pyart.filters.GateFilter(self.radar)
    gatefilter = pyart.filters.GateFilter(self.radar)
    gatefilter_sq = pyart.filters.GateFilter(self.radar)
    
    # Apply dbz and sector regardless of Temp 
    if self.do_dbz == True:
        if self.dbz_max:
            dbzfilter.exclude_outside('CZ', self.dbz_thresh, self.dbz_max)
        else:    
            dbzfilter.exclude_below('CZ', self.dbz_thresh)
    if self.do_sector == True: dbzfilter.exclude_not_equal('SEC', cos)
    if self.do_rh_sector == True: dbzfilter.exclude_not_equal('SECRH', sec)
    if self.do_kd_sector == True: dbzfilter.exclude_not_equal('SECKD', sec)
    if self.do_sw_sector == True: dbzfilter.exclude_not_equal('SECSW', sec) 
    if self.do_sq_sector == True: dbzfilter.exclude_not_equal('SECSQ', sec) 
    if self.do_cos == True: dbzfilter.exclude_not_equal('COS', cos)
    #if self.do_sq == True: dbzfilter.exclude_below('SQ', self.sq_thresh)
    if self.radar.metadata['original_container'] == 'NEXRAD Level II' or\
       self.radar.metadata['original_container'] == 'UF' or\
       self.radar.metadata['original_container'] == 'odim_h5': dbzfilter.exclude_not_equal('WSR', cos)
    
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
        dz_lt = np.ma.where(dz <= self.ap_dbz , 1, 0)
        dr_lt = np.ma.where(dr >= self.ap_zdr , 1, 0)
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
        sq_height = self.sq_height * 1000
        gatefilter.exclude_all()
        gatefilter_sq.exclude_all()
        gatefilter.include_below('HEIGHT', qc_height)
        gatefilter_sq.include_below('HEIGHT', sq_height)
        if self.do_rh == True: gatefilter.exclude_below('RH', self.rh_thresh)
        if self.do_zdr == True: gatefilter.exclude_outside('DR', self.dr_min, self.dr_max)
        if self.do_ap == True: gatefilter.exclude_not_equal('AP', sec)
        if self.do_sq == True: gatefilter_sq.exclude_below('SQ', self.sq_thresh)
        gatefilter.include_above('HEIGHT', qc_height)
        gatefilter_sq.include_above('HEIGHT', sq_height)
    elif self.use_sounding == True:
        gatefilter.exclude_all()
        gatefilter.include_above('TEMP', 3.0)
        if self.do_rh == True: gatefilter.exclude_below('RH', self.rh_thresh)
        if self.do_zdr == True: gatefilter.exclude_outside('DR', self.dr_min, self.dr_max)
        if self.do_ap == True: gatefilter.exclude_not_equal('AP', sec)
        if self.do_sq == True: gatefilter_sq.exclude_below('SQ', self.sq_thresh)
        gatefilter.include_below('TEMP', 3.1)   
    
    # Apply gate filters to radar
    for fld in self.radar.fields:
        nf = deepcopy(self.radar.fields[fld])
        nf['data'] = np.ma.masked_where(gatefilter.gate_excluded, nf['data'])
        self.radar.add_field(fld, nf, replace_existing=True)
    for fld in self.radar.fields:
        nf = deepcopy(self.radar.fields[fld])
        nf['data'] = np.ma.masked_where(gatefilter_sq.gate_excluded, nf['data'])
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
    gatefilter_sd = pyart.filters.GateFilter(self.radar)

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
        sd_height = self.sd_height * 1000
        gatefilter_cal.exclude_all()
        gatefilter_sd.exclude_all()
        gatefilter_cal.include_below('HEIGHT', qc_height)
        gatefilter_sd.include_below('HEIGHT', sd_height)
        if self.sd_thresh_max == 0:
            if self.do_sd == True: gatefilter_sd.exclude_above('SD', self.sd_thresh)
        else:
            if self.do_sd == True: gatefilter_sd.include_outside('SD', self.sd_thresh,self.sd_thresh_max)
        if self.do_kdp == True: gatefilter_cal.exclude_outside('KD', self.kdp_min, self.kdp_max)
        if self.do_ph == True: gatefilter_cal.exclude_above('PH', self.ph_thresh)
        gatefilter_cal.include_above('HEIGHT', qc_height)
        gatefilter_sd.include_above('HEIGHT', sd_height)
    elif self.use_sounding == True:
        gatefilter_cal.exclude_all()
        gatefilter_cal.include_above('TEMP', 3.0)
        if self.do_sd == True: gatefilter_cal.exclude_above('SD', self.sd_thresh)
        if self.do_kdp == True: gatefilter_cal.exclude_outside('KD', self.kdp_min, self.kdp_max)
        if self.do_ph == True: gatefilter_cal.exclude_above('PH', self.ph_thresh)
        gatefilter_cal.include_below('TEMP', 3.1)    

    # Apply gate filters to radar
    for fld in self.radar.fields:
        nf = deepcopy(self.radar.fields[fld])
        nf['data'] = np.ma.masked_where(gatefilter_cal.gate_excluded, nf['data'])
        self.radar.add_field(fld, nf, replace_existing=True)
    if self.use_qc_height == True or self.use_sounding == False:
        for fld in self.radar.fields:
            nf = deepcopy(self.radar.fields[fld])
            nf['data'] = np.ma.masked_where(gatefilter_sd.gate_excluded, nf['data'])
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

    sector = {'hmin': None, 'hmax': None,
              'rmin': 200 * 1000, 'rmax':  None,
              'azmin': None, 'azmax': None,
              'elmin': None, 'elmax': None}
    
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

def kd_sector(self):
    
    """
    filter out any data inside the region of interest that is < kd_sector

    Parameters
    ----------
    radar : radar object
            the radar object where the data is
    self : dict
            a dictionary defining the region of interest

    Returns
    -------
    sector_kd : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': self.kdhmin, 'hmax': self.kdhmax,
	          'rmin':  self.kdrmin * 1000, 'rmax':  self.kdrmax * 1000,
              'azmin': self.kdazmin, 'azmax': self.kdazmax,
	          'elmin': self.kdelmin, 'elmax': self.kdelmax}

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
    
    kd = self.radar.fields['KD']['data'].copy()
    sector_k = np.ones(kd.shape)
    kd_nan = np.ma.where(np.ma.getmaskarray(kd), 1, 0)
    sec_f = np.logical_and(kd_nan == 1 , sector_wipeout == 1)
    sector_k[sec_f] = 0

    sector_kd = sector_k
    cm.add_field_to_radar_object(sector_kd, self.radar, field_name='SECKD', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='Sector KD Mask', 
                                 standard_name='Sector KD Mask', 
                                 dz_field=self.ref_field_name)
     
    return self.radar

# ***************************************************************************************

def sw_sector(self):
    
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
    sector_sw : ndarray
    a field array with ones in gates that are in the Region of Interest

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    sector = {'hmin': self.swhmin, 'hmax': self.swhmax,
	          'rmin':  self.swrmin * 1000, 'rmax':  self.swrmax * 1000,
              'azmin': self.swazmin, 'azmax': self.swazmax,
	          'elmin': self.swelmin, 'elmax': self.swelmax,
              'sw_sec': self.sw_sec}

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
    
    sw = self.radar.fields['SW']['data'].copy()
    sector_w = np.ones(sw.shape)
    sw_sec = sector['sw_sec']
    sw_lt = np.ma.where(sw < sw_sec , 1, 0)
    sec_f = np.logical_and(sw_lt == 1 , sector_wipeout == 1)
    sector_w[sec_f] = 0

    sector_sw = sector_w
    cm.add_field_to_radar_object(sector_sw, self.radar, field_name='SECSW', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='Sector SW Mask', 
                                 standard_name='Sector SW Mask', 
                                 dz_field=self.ref_field_name)
     
    return self.radar

# ***************************************************************************************

def sq_sector(self):
    
    """
    filter out any data inside the region of interest that is < sq_sector

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

    sector = {'hmin': self.sqhmin, 'hmax': self.sqhmax,
	      'rmin':  self.sqrmin * 1000, 'rmax':  self.sqrmax * 1000,
              'azmin': self.sqazmin, 'azmax': self.sqazmax,
	      'elmin': self.sqelmin, 'elmax': self.sqelmax,
              'sq_sec': self.sq_sec}

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
    
    sq = self.radar.fields['SQ']['data'].copy()
    sector_r = np.ones(sq.shape)
    sq_sec = sector['sq_sec']
    sq_lt = np.ma.where(sq < sq_sec , 1, 0)
    sec_f = np.logical_and(sq_lt == 1 , sector_wipeout == 1)
    sector_r[sec_f] = 0

    sector_sq = sector_r
    cm.add_field_to_radar_object(sector_sq, self.radar, field_name='SECSQ', 
                                 units='0 = Z < 0, 1 = Z >= 0',
                                 long_name='Sector SQ Mask', 
                                 standard_name='Sector SQ Mask', 
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
    
    ph = self.radar.fields['PH']['data'].copy()
    sector_p = np.ones(ph.shape)
    ph_sec = sector['ph_sec']
    ph_lt = np.ma.where(ph > ph_sec , 1, 0)
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
    MAX_PHIDP_DIFF = self.max_phidp_diff          # Set maximum phidp gate-to-gate difference allowed

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
            if(diff < 0.0):
                if abs(diff) >= MAX_PHIDP_DIFF:
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
		long_name='Unfolded Differential Phase (Marks)',
		standard_name='Differential Phase',
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

#    DZ = cm.extract_unmasked_data(self.radar, self.ref_field_name)
#    DP = cm.extract_unmasked_data(self.radar, self.phi_field_name)
#    DZ = self.radar.fields[self.ref_field_name]['data'].copy()
#    DP = self.radar.fields[self.phi_field_name]['data'].copy()

    if not self.noKDP:
        print('    Getting new Kdp...')
        std_list  = ['AL1','JG1','MC1','NT1','PE1','SF1','ST1','SV1','TM1']
        if self.site in std_list:
            try:
                DZ = cm.extract_unmasked_data(self.radar, self.ref_field_name)
                DP = cm.extract_unmasked_data(self.radar, self.phi_field_name)
            except:  
                DZ = self.radar.fields['DZ']['data'].copy()
                DP = self.radar.fields['PH']['data'].copy()
            window=4
            std_gate=15
            nfilter=1
        else:
            try:
                DZ = cm.extract_unmasked_data(self.radar, self.ref_field_name)
                DP = cm.extract_unmasked_data(self.radar, self.phi_field_name)
            except:
                DZ = self.radar.fields['DZ']['data'].copy()
                DP = self.radar.fields['PH']['data'].copy()
            window=4
            std_gate=15
            nfilter=1

        # Range needs to be supplied as a variable, with same shape as DZ
        rng2d, az2d = np.meshgrid(self.radar.range['data'], self.radar.azimuth['data'])
        gate_spacing = self.radar.range['meters_between_gates']

        try:
            KDPB, PHIDPB, STDPHIB = csu_kdp.calc_kdp_bringi(dp=DP, dz=DZ, rng=rng2d/1000.0, 
                                                        thsd=25, gs=gate_spacing, 
                                                        window=window, nfilter=nfilter, 
                                                        std_gate=std_gate)                                                 
        except Exception as e:
            print("An error occurred:", e)
            traceback.print_exc()
            print('    CSU Radar Tools could not retrieve Kdp...')
            KDPB = np.zeros((self.radar.nrays, self.radar.ngates), dtype=float) - 32767.0
            PHIDPB = np.zeros((self.radar.nrays, self.radar.ngates), dtype=float) - 32767.0
            STDPHIB = np.zeros((self.radar.nrays, self.radar.ngates), dtype=float) - 32767.0

        if 'KD' not in self.radar.fields.keys():
            self.radar = cm.add_field_to_radar_object(KDPB, self.radar, field_name='KD', 
		        units='deg/km',
		        long_name='Specific Differential Phase (Bringi)',
		        standard_name='Specific Differential Phase (Bringi)',
		        dz_field=self.ref_field_name)

    if self.unfold_phidp == False:
        print('    Retrieving CSU PH')
        self.radar = cm.add_field_to_radar_object(PHIDPB, self.radar, 
		    field_name='PHIDPB', units='deg',
		    long_name='Differential Phase (Bringi)',
		    standard_name='Differential Phase (Bringi)',
		    dz_field=self.ref_field_name)
    
    if self.get_GV_SD:
        print('    Retrieving GPMGV SD')
        self.radar = get_SD(self)
    else:
        print('    Retrieving CSU SD')
        self.radar = cm.add_field_to_radar_object(STDPHIB, self.radar, 
		    field_name='SD', units='deg',
		    long_name='STD Differential Phase (Bringi)',
		    standard_name='STD Differential Phase (Bringi)',
		    dz_field=self.ref_field_name)
    
    return self.radar

# ***************************************************************************************

def get_SD(self):
    
    BAD_DATA       = -32767.0
    ws = self.SD_window
    
    ws_h = ws//2

    # Copy current PhiDP field to phm_field
    ph_field = self.radar.fields['PH']['data'].copy()
    sd_field = self.radar.fields['PH']['data'].copy() * 0
    dz_field = self.radar.fields['CZ']['data'].copy()
    nrays = ph_field.data.shape[0]
    gate_data = ph_field.data[0]
    ngates = gate_data.shape[0]

    for iray in range(0, nrays-1):
        ph_gate_data = ph_field.data[iray]
        sd_gate_data = sd_field.data[iray]
        dz_gate_data = dz_field.data[iray]
    
        for igate in range(0, ws_h-1):
            phbox=ph_gate_data[0:ws]
            dzbox=dz_gate_data[0:ws]
            x = np.logical_and(phbox != BAD_DATA, dzbox != BAD_DATA)
            if sum(x) >= 5:
                sd_gate_data[igate] = np.std(phbox[x])
            else:
                sd_gate_data[igate] = BAD_DATA
            
        for igate in range(ws_h,ngates-(ws_h+1)):
            phbox=ph_gate_data[igate-ws_h:igate+ws_h]
            dzbox=dz_gate_data[igate-ws_h:igate+ws_h]
            x = np.logical_and(phbox != BAD_DATA, dzbox != BAD_DATA)
            if sum(x) >= 5:
                sd_gate_data[igate] = np.std(phbox[x])
            else:
                sd_gate_data[igate] = BAD_DATA
         
        for igate in range(ngates-ws_h,ngates-1):
            phbox=ph_gate_data[igate-ws:igate-1]
            dzbox=dz_gate_data[igate-ws:igate-1]
            x = np.logical_and(phbox != BAD_DATA, dzbox != BAD_DATA)
            if sum(x) >= 5:
                sd_gate_data[igate] = np.std(phbox[x])
            else:
                sd_gate_data[igate] = BAD_DATA
         
        sd_field.data[iray] = sd_gate_data
    
    sd_dict = {"data": sd_field, "units": "Std(PhiDP)",
               "long_name": "STD Differential Phase (GPM-GV)", "_FillValue": -32767.0,
               "standard_name": "STD Differential Phase (GPM-GV)",}
    self.radar.add_field("SD", sd_dict, replace_existing=True)

    return self.radar

# ***************************************************************************************

def remove_fields_from_radar(self):

    """
    Remove fields from radar that are not needed.  
    """    

    print("Removing QC threshold fields...", '', sep='\n')

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

    return self.radar

# ***************************************************************************************

def get_default_thresh_dict():

    default_thresh_dict = {'do_dbz': True, 'dbz_thresh': 5.0, 'dbz_max': None,
                           'do_rh': True, 'rh_thresh': 0.72,
                           'do_zdr': True, 'dr_min': -6.0, 'dr_max': 4.0, 
                           'do_kdp': False, 'kdp_min': -2.0, 'kdp_max': 7.0, 
                           'do_sq': False, 'sq_thresh': 0.45, 
                           'do_sd': True, 'sd_thresh': 18.0, 'sd_thresh_max': 0,
                           'do_ph': False, 'ph_thresh': 80.0, 'max_phidp_diff': 360,
                           'do_ap': True, 'ap_dbz': 45, 'ap_zdr': 3,
                           'get_GV_SD':  False, 'SD_window': 15,
                           'noKDP':  False,
                           'unfold_phidp': True,
                           'merge_sp': True,
                           'dealias_velocity': False,
                           'do_sw_mask': False,
                           'do_insect': False, 
                           'do_despeckle': True, 'speck_size': 10, 'pyart_speck': True,
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
                           'do_sw_sector': False, 'swhmin': 0, 'swhmax': None,
                           'swrmin': 0, 'swrmax': 20, 
                           'swazmin': 0, 'swazmax': 360, 
                           'swelmin': 0, 'swelmax': 7.0, 'sw_sec': 2.0, 
                           'do_kd_sector': False, 'kdhmin': 0, 'kdhmax': None,
                           'kdrmin': 0, 'kdrmax': 20, 
                           'kdazmin': 0, 'kdazmax': 360, 
                           'kdelmin': 0, 'kdelmax': 7.0,
                           'do_sq_sector': False, 'sqhmin': 0, 'sqhmax': None,
                           'sqrmin': 0, 'sqrmax': 20, 
                           'sqazmin': 0, 'sqazmax': 360, 
                           'sqelmin': 0, 'sqelmax': 7.0, 'sq_sec': 0.35,
                           'do_sd_sector': False, 'sdhmin': 0, 'sdhmax': None, 
                           'sdrmin': 0, 'sdrmax': 20, 
                           'sdazmin': 0, 'sdazmax': 360, 
                           'sdelmin': 0, 'sdelmax': 7.0, 'sd_sec': 8.0, 
                           'do_ph_sector': False, 'phhmin': 0, 'phhmax': None,
                           'phrmin': 0, 'phrmax': 150, 
                           'phazmin': 160, 'phazmax': 165, 
                           'phelmin': 0, 'phelmax': 20.0, 'ph_sec': 80.0,
                           'get_cal_file': False, 'cal_dir': './cal_files/', 
                           'apply_cal': False, 'ref_cal': 0.1, 'zdr_cal': 0.0, 
                           'use_qc_height': True, 'qc_height': 4.4,
                           'sd_height': 4.4,
                           'sq_height': 4.4,
                           'output_cf': False,
                           'output_grid': False,
                           'output_fields': ['DZ', 'CZ', 'VR', 'DR', 'KD', 'PH', 'RH', 'SD'],
                           'cf_dir': './cf',
                           'grid_dir': './grid',
                           'plot_raw_images': False,
                           'png': True,
                           'plot_images': True, 'max_range': 200, 'max_height': 10,
                           'sweeps_to_plot': [0], 
                           'plot_single': True,
                           'plot_multi': False,
                           'plot_fast': False,
                           'mask_outside': True,
                           'fields_to_plot': ['CZ', 'DR', 'KD', 'PH', 'RH', 'VR'],
                           'plot_dir': './plots/', 'add_logos': True,
                           'use_sounding': False, 'sounding_type': 'ruc_archive', 'sounding_dir': './sounding/'}

    return default_thresh_dict

# ***************************************************************************************

def velocity_dealias(self):
    
    # Set the nyquist
    nyq = self.radar.instrument_parameters['nyquist_velocity']['data'][0]
    
    '''
    # Testing VR gatefliter
    # Calculate the velocity texture
    vel_texture = pyart.retrieve.calculate_velocity_texture(self.radar, 
                  vel_field='VR', wind_size=8, nyq=nyq)
    self.radar.add_field('VT', vel_texture, replace_existing=True)
      
    # Set up the gatefilter to be based on the velocity texture.
    vr_gatefilter = pyart.filters.GateFilter(self.radar)
    vr_gatefilter.exclude_above('VT', 3)
        
    vt = deepcopy(self.radar.fields['VT'])
    vr = deepcopy(self.radar.fields['VR'])
    # Apply gate filters to radar
    for fld in self.radar.fields:
        nf = deepcopy(self.radar.fields[fld])
        nf['data'] = np.ma.masked_where(vr_gatefilter.gate_excluded, nf['data'])
        self.radar.add_field(fld, nf, replace_existing=True)  
        
    self.radar.add_field('VT', vt, replace_existing=True)
    self.radar.add_field('VR', vr, replace_existing=True)  
    ''' 
    
    # Dealiasing Velocity                     
    print('    Dealiasing Velocity...')
    
    velocity_dealiased = pyart.correct.dealias_region_based(self.radar, vel_field='VR',
                                                            nyquist_vel=nyq, centered=True)
    self.radar.add_field('VR', velocity_dealiased, replace_existing=True)   
        
    return self.radar
    
# ***************************************************************************************

def filter_D3R_el(self, elevation_range=(0, 90), 
                                azimuth_range=None, 
                                copy_radar=True,
                                verbose=True):
    """
    Filter radar object to include only elevations from 0-90 degrees
    
    Parameters:
    -----------
    radar : pyart.core.Radar
        Input radar object
    elevation_range : tuple, optional
        (min_elevation, max_elevation) in degrees. Default: (0, 90)
    azimuth_range : tuple, optional
        (min_azimuth, max_azimuth) in degrees. If None, all azimuths included
    copy_radar : bool, optional
        If True, create copy of radar object. If False, modify in place
    verbose : bool, optional
        Print filtering statistics
        
    Returns:
    --------
    filtered_radar : pyart.core.Radar
        Filtered radar object
    """
    
    # Get elevation data and correct for 360° scans
    elevation_deg = np.array(self.radar.elevation['data'])
    elevation_corrected = elevation_deg.copy()
    elevation_corrected[elevation_corrected > 180] = elevation_corrected[elevation_corrected > 180] - 360
    
    # Get azimuth data if filtering by azimuth
    azimuth_deg = np.array(self.radar.azimuth['data']) if azimuth_range is not None else None
    
    # Create elevation mask
    min_elev, max_elev = elevation_range
    elevation_mask = (elevation_corrected >= min_elev) & (elevation_corrected <= max_elev)
    
    # Create azimuth mask if specified
    if azimuth_range is not None:
        min_az, max_az = azimuth_range
        if min_az <= max_az:
            # Normal case: e.g., 90° to 180°
            azimuth_mask = (azimuth_deg >= min_az) & (azimuth_deg <= max_az)
        else:
            # Wrap-around case: e.g., 270° to 90° (crosses 0°)
            azimuth_mask = (azimuth_deg >= min_az) | (azimuth_deg <= max_az)
    else:
        azimuth_mask = np.ones(len(elevation_deg), dtype=bool)
    
    # Combine masks
    combined_mask = elevation_mask & azimuth_mask
    
    # Check if any rays remain
    if not np.any(combined_mask):
        raise ValueError(f"No rays found in specified range. "
                        f"Elevation: {elevation_range}, Azimuth: {azimuth_range}")
    
    # Print statistics if verbose
    if verbose:
        print("=== Radar Forward Filtering Results ===")
        print(f"Original rays: {len(elevation_deg)}")
        print(f"Elevation range: {min_elev}° to {max_elev}°")
        if azimuth_range:
            print(f"Azimuth range: {azimuth_range[0]}° to {azimuth_range[1]}°")
        print(f"Rays passing elevation filter: {np.sum(elevation_mask)} ({100*np.sum(elevation_mask)/len(elevation_deg):.1f}%)")
        if azimuth_range:
            print(f"Rays passing azimuth filter: {np.sum(azimuth_mask)} ({100*np.sum(azimuth_mask)/len(elevation_deg):.1f}%)")
        print(f"Final filtered rays: {np.sum(combined_mask)} ({100*np.sum(combined_mask)/len(elevation_deg):.1f}%)")
        
        if np.any(combined_mask):
            final_elevations = elevation_corrected[combined_mask]
            print(f"Filtered elevation range: {final_elevations.min():.1f}° to {final_elevations.max():.1f}°")
    
    # Use PyART's extract_sweeps and manual construction
    if copy_radar:
        # Create a new radar object with filtered data
        filtered_radar = create_filtered_radar_object(self.radar, combined_mask, elevation_corrected)
    else:
        # Modify in place - this is more complex and risky, so we'll create a new one anyway
        filtered_radar = create_filtered_radar_object(self.radar, combined_mask, elevation_corrected)
        if verbose:
            print("Note: Created new radar object instead of in-place modification for safety")
    
    # Update metadata
    if 'comment' in filtered_radar.metadata:
        filtered_radar.metadata['comment'] += f" | Forward filtered: {elevation_range}°"
    else:
        filtered_radar.metadata['comment'] = f"Forward filtered: elevation {elevation_range[0]}-{elevation_range[1]}°"
        
    if azimuth_range:
        filtered_radar.metadata['comment'] += f", azimuth {azimuth_range[0]}-{azimuth_range[1]}°"
    
    return filtered_radar

# ***************************************************************************************

def create_filtered_radar_object(original_radar, mask, elevation_corrected):
  
    """
    Create a new filtered radar object using PyART's safe methods
    """
    
    # Get filtered indices
    ray_indices = np.where(mask)[0]
    
    # Use PyART's extract_sweeps
    try:
        # Start with the first sweep
        filtered_radar = original_radar.extract_sweeps([0])
    except:
        # If extract_sweeps fails, create manually
        filtered_radar = _create_radar_manually(original_radar, mask, elevation_corrected)
        return filtered_radar
    
    # Get the original sweep limits
    sweep_start = original_radar.sweep_start_ray_index['data'][0]
    sweep_end = original_radar.sweep_end_ray_index['data'][0]
    
    # Filter coordinate data
    original_elevation = np.array(original_radar.elevation['data'])
    original_azimuth = np.array(original_radar.azimuth['data'])
    
    # Update basic radar parameters
    filtered_radar.nrays = np.sum(mask)
    filtered_radar.sweep_end_ray_index['data'][0] = np.sum(mask) - 1
    
    # Update coordinate arrays
    filtered_radar.elevation['data'] = elevation_corrected[mask]
    filtered_radar.azimuth['data'] = original_azimuth[mask]
    
    # Keep the same range array
    filtered_radar.range = original_radar.range
    filtered_radar.ngates = original_radar.ngates
    
    # Update all field data
    for field_name in original_radar.fields:
        if field_name in filtered_radar.fields:
            original_field_data = original_radar.fields[field_name]['data']
            
            # Handle different data shapes
            if len(original_field_data.shape) == 1:
                if len(original_field_data) == len(mask):
                    filtered_radar.fields[field_name]['data'] = original_field_data[mask]
            elif len(original_field_data.shape) == 2:
                if original_field_data.shape[0] == len(mask):
                    filtered_radar.fields[field_name]['data'] = original_field_data[mask, :]
                elif original_field_data.shape[0] > len(mask):
                    # Handle case where we have more data rows than mask elements
                    filtered_radar.fields[field_name]['data'] = original_field_data[sweep_start:sweep_end+1, :][mask, :]
    
    # Copy other important attributes
    filtered_radar.metadata = original_radar.metadata.copy()
    filtered_radar.scan_type = original_radar.scan_type
    
    return filtered_radar 

# ***************************************************************************************

def boundary_artifact_removal(self, boundary_km=None, extend_boundary=0.05, 
                                         verbose=True):
    """
    Apply full zone smooth transition boundary artifact removal and return a radar object
    """

    if verbose:
        print("=== Boundary Artifact Removal ===")
    
    # Use PyART's extract_sweeps to create a workable copy
    try:
        filtered_radar = self.radar.extract_sweeps([0])  # Extract first sweep
    except:
        # Fallback: work with the original radar (modify in place)
        filtered_radar = self.radar
        if verbose:
            print("Working with original radar object (no copy created)")
    
    # Get range information
    range_km = np.array(filtered_radar.range['data']) / 1000.0
    
    # Determine boundary location
    if boundary_km is None:
        # Auto-detect and round up to nearest tenth
        detected_boundary, _ = detect_processing_boundary(filtered_radar, verbose=False)
        if detected_boundary is not None:
            boundary_km = np.ceil(detected_boundary * 10) / 10.0  # Round up to nearest 0.1 km
            if verbose:
                print(f"Auto-detected boundary: {detected_boundary:.2f} km")
                print(f"Rounded up to: {boundary_km:.1f} km")
        else:
            boundary_km = 4.4  # Default fallback
            if verbose:
                print(f"Auto-detection failed, using default: {boundary_km} km")
    else:
        if verbose:
            print(f"Using manual boundary: {boundary_km} km")
    
    # Add extension beyond boundary
    processing_boundary = boundary_km + extend_boundary
    boundary_idx = np.argmin(np.abs(range_km - processing_boundary))
    actual_boundary_km = range_km[boundary_idx]
    
    if verbose:
        print(f"Processing boundary: {processing_boundary:.2f} km (gate {boundary_idx})")
        print(f"Actual boundary used: {actual_boundary_km:.2f} km")
        print(f"Will process inner zone: 0 to {actual_boundary_km:.2f} km ({boundary_idx} gates)")
    
    # Apply smooth transition to radar fields
    fields_processed = []
    processing_stats = {}
    
    for field_name in filtered_radar.fields:
        if verbose:
            print(f"Processing field: {field_name}")
        
        try:
            # Get field data - convert to regular numpy array
            original_field_data = np.array(filtered_radar.fields[field_name]['data'])
            
            # Apply smooth transition
            filtered_field_data = apply_smooth_transition_to_field(
                original_field_data, boundary_idx, range_km, field_name, verbose=False
            )
            
            # Update the radar object - direct assignment to the data array
            filtered_radar.fields[field_name]['data'][:] = filtered_field_data
            
            # Calculate statistics
            original_std = np.nanstd(original_field_data[:, :boundary_idx])
            filtered_std = np.nanstd(filtered_field_data[:, :boundary_idx])
            
            processing_stats[field_name] = {
                'original_std': original_std,
                'filtered_std': filtered_std,
                'improvement_ratio': filtered_std / original_std if original_std > 0 else np.nan
            }
            
            fields_processed.append(field_name)
            
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Could not process {field_name}: {e}")
    
    # Update radar metadata
    if hasattr(filtered_radar, 'metadata'):
        try:
            if 'comment' in filtered_radar.metadata:
                filtered_radar.metadata['comment'] += f" | Boundary artifact removed (0-{actual_boundary_km:.1f}km)"
            else:
                filtered_radar.metadata['comment'] = f"Boundary artifact removed (0-{actual_boundary_km:.1f}km)"
        except:
            pass  # Skip metadata update if it fails
    
    # Prepare processing information
    processing_info = {
        'boundary_km': actual_boundary_km,
        'boundary_idx': boundary_idx,
        'fields_processed': fields_processed,
        'processing_stats': processing_stats,
        'method': 'smooth_transition_full_zone',
        'inner_zone_gates': boundary_idx,
        'success': len(fields_processed) > 0
    }
    
    if verbose:
        print(f"\n✅ Processing complete!")
        print(f"Fields processed: {len(fields_processed)}")
        print(f"Inner zone replaced: 0 to {actual_boundary_km:.2f} km")
        
        # Show improvement for key fields
        key_fields = ['Reflectivity', 'Velocity', 'SpectralWidth']
        for field in key_fields:
            if field in processing_stats:
                stats = processing_stats[field]
                if not np.isnan(stats['improvement_ratio']):
                    print(f"  {field}: std {stats['original_std']:.2f} -> {stats['filtered_std']:.2f} ({stats['improvement_ratio']:.1f}x)")
    
    return filtered_radar, processing_info

# ***************************************************************************************

def create_clean_radar_simple(radar, boundary_km=4.4, verbose=True):
    """
    Simple function to create a clean radar with boundary artifact removed
    """
    
    if verbose:
        print(f"Applying boundary artifact removal at {boundary_km} km...")
    
    # Apply the filter
    clean_radar, info = apply_boundary_artifact_removal_fixed(
        radar, 
        boundary_km=boundary_km,
        extend_boundary=0.0,
        verbose=verbose
    )
    
    if info['success']:
        if verbose:
            print("✅ Success! Boundary artifact removed.")
            print(f"Processed fields: {', '.join(info['fields_processed'])}")
        return clean_radar
    else:
        if verbose:
            print("❌ Failed to remove boundary artifact.")
        return radar  # Return original if processing failed

# ***************************************************************************************

def apply_smooth_transition_to_field(field_data, boundary_idx, range_km, field_name, verbose=True):
    """
    Apply smooth transition to a specific radar field
    """
    
    # Handle different field characteristics
    field_characteristics = {
        'DZ': {'realistic_std': 10.0, 'typical_range': (-20, 50)},
        'CZ': {'realistic_std': 10.0, 'typical_range': (-20, 50)},
        'VR': {'realistic_std': 5.0, 'typical_range': (-30, 30)},
        'SW': {'realistic_std': 1.0, 'typical_range': (0, 8)},
        'DR': {'realistic_std': 1.0, 'typical_range': (-2, 5)},
        'RH': {'realistic_std': 0.1, 'typical_range': (0.5, 1.0)},
        'PH': {'realistic_std': 20.0, 'typical_range': (-180, 180)}
    }
    
    # Get field characteristics or use defaults
    field_char = field_characteristics.get(field_name, {'realistic_std': 5.0, 'typical_range': (-50, 50)})
    
    filtered_data = field_data.copy()
    
    # Define zones
    inner_zone = slice(0, boundary_idx)
    outer_start = boundary_idx + 2
    outer_zone = slice(outer_start, min(field_data.shape[1], outer_start + 20))
    
    if outer_zone.start >= outer_zone.stop:
        return field_data  # Can't process without outer zone reference
    
    # Get outer zone statistics
    outer_data = field_data[:, outer_zone]
    target_mean = np.nanmean(outer_data)
    target_std = np.nanstd(outer_data)
    
    # Use realistic defaults if outer zone is problematic
    if np.isnan(target_mean):
        target_mean = np.nanmean(field_char['typical_range'])
    if np.isnan(target_std) or target_std <= 0:
        target_std = field_char['realistic_std']
    
    # Process each ray
    for ray_idx in range(field_data.shape[0]):
        # Get reference data from outer zone for this ray
        ray_outer_data = field_data[ray_idx, outer_zone]
        ray_outer_valid = ray_outer_data[~np.isnan(ray_outer_data)]
        
        if len(ray_outer_valid) > 3:
            # Use this ray's characteristics
            ray_mean = np.nanmean(ray_outer_valid)
            ray_std = np.nanstd(ray_outer_valid) if len(ray_outer_valid) > 1 else target_std
            
            # Create realistic inner zone data
            if len(ray_outer_valid) >= boundary_idx:
                # Sample from outer zone with replacement
                sampled_indices = np.random.choice(len(ray_outer_valid), size=boundary_idx, replace=True)
                synthetic_data = ray_outer_valid[sampled_indices]
                
                # Add spatial correlation by smoothing
                if boundary_idx > 2:
                    kernel = np.array([0.25, 0.5, 0.25])
                    synthetic_data = np.convolve(synthetic_data, kernel, mode='same')
                    
            else:
                # Generate synthetic data
                synthetic_data = np.random.normal(ray_mean, ray_std, boundary_idx)
                
                # Clip to reasonable range for this field
                min_val, max_val = field_char['typical_range']
                synthetic_data = np.clip(synthetic_data, min_val, max_val)
            
            # Apply to inner zone
            filtered_data[ray_idx, inner_zone] = synthetic_data
            
        else:
            # Fallback: use global statistics
            synthetic_data = np.random.normal(target_mean, target_std, boundary_idx)
            min_val, max_val = field_char['typical_range']
            synthetic_data = np.clip(synthetic_data, min_val, max_val)
            filtered_data[ray_idx, inner_zone] = synthetic_data
    
    return filtered_data

# ***************************************************************************************

def detect_processing_boundary(radar, min_boundary_km=3.0, 
                                      max_boundary_km=6.0, verbose=True):
    """
    Automatically detect processing boundary by looking for sharp std deviation increase
    """
    
    range_km = np.array(radar.range['data']) / 1000.0
    ref_data = np.array(radar.fields['Reflectivity']['data'])
    
    # Calculate standard deviation for each range gate
    std_by_range = np.nanstd(ref_data, axis=0)
    
    # Smooth the std profile to reduce noise
    window_size = 3
    std_smoothed = ndimage.uniform_filter1d(std_by_range, size=window_size, mode='nearest')
    
    # Find the gradient (rate of change) in standard deviation
    std_gradient = np.gradient(std_smoothed)
    
    # Look for the transition from LOW std to HIGH std
    # Based on our analysis: inner zone has ~2 dBZ std, outer zone has ~10 dBZ std
    
    search_mask = (range_km >= min_boundary_km) & (range_km <= max_boundary_km)
    search_indices = np.where(search_mask)[0]
    
    if len(search_indices) == 0:
        return None, {'error': 'No valid search range'}
    
    # Find the location where std makes the biggest jump from low to high
    best_boundary_idx = None
    best_score = 0
    
    for idx in search_indices:
        # Check window before this point (should be low std)
        before_window = slice(max(0, idx-5), idx)
        # Check window after this point (should be high std)  
        after_window = slice(idx, min(len(std_smoothed), idx+5))
        
        if before_window.stop > before_window.start and after_window.stop > after_window.start:
            std_before = np.nanmean(std_smoothed[before_window])
            std_after = np.nanmean(std_smoothed[after_window])
            
            # Score based on:
            # 1. Large absolute increase in std
            # 2. Low std before (processed data characteristic)
            # 3. High std after (natural data characteristic)
            
            if std_before > 0:
                std_ratio = std_after / std_before
                std_increase = std_after - std_before
                
                # Composite score favoring:
                # - Large ratio increase (std_after >> std_before)
                # - Large absolute increase 
                # - Low initial std (< 5 dBZ suggests processed data)
                score = std_ratio * std_increase * (1.0 if std_before < 5.0 else 0.5)
                
                if score > best_score:
                    best_score = score
                    best_boundary_idx = idx
    
    if best_boundary_idx is None:
        if verbose:
            print("No significant processing boundary detected")
        return None, {'boundary_detected': False}
    
    boundary_km = range_km[best_boundary_idx]
    
    # Get final statistics for the detected boundary
    before_window = slice(max(0, best_boundary_idx-5), best_boundary_idx)
    after_window = slice(best_boundary_idx, min(len(std_smoothed), best_boundary_idx+5))
    
    std_before = np.nanmean(std_smoothed[before_window])
    std_after = np.nanmean(std_smoothed[after_window])
    std_ratio = std_after / std_before if std_before > 0 else np.inf
    
    detection_info = {
        'boundary_detected': True,
        'boundary_km': boundary_km,
        'boundary_idx': best_boundary_idx,
        'std_before': std_before,
        'std_after': std_after,
        'std_ratio': std_ratio,
        'score': best_score,
        'confidence': 'high' if std_ratio > 4 and std_before < 3 else 'medium' if std_ratio > 2 else 'low'
    }
    
    if verbose:
        print(f"Improved boundary detection results:")
        print(f"  Location: {boundary_km:.2f} km (gate {best_boundary_idx})")
        print(f"  Std before: {std_before:.2f} dBZ")
        print(f"  Std after: {std_after:.2f} dBZ") 
        print(f"  Ratio: {std_ratio:.1f}x increase")
        print(f"  Score: {best_score:.2f}")
        print(f"  Confidence: {detection_info['confidence']}")
    
    return boundary_km, detection_info    
