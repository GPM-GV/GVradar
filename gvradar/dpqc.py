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
        if self.site == 'NPOL':
            if self.sd_thresh_max == 0:
                if self.do_sd == True: gatefilter_sd.exclude_above('SD', self.sd_thresh)
            else:
                if self.do_sd == True: gatefilter_sd.include_outside('SD', self.sd_thresh,self.sd_thresh_max)
        else:
            if self.do_sd == True: gatefilter_sd.exclude_above('SD', self.sd_thresh)
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

    if noKDP:
        continue
    else:
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
                           'do_sd': True, 'sd_thresh': 18.0, 'sd_thresh_max': 60,
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
    
