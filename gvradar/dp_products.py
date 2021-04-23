# ***************************************************************************************
'''
Rainfall product generation from Dual Pol data, utilizing PyArt.

Developed by the NASA GPM-GV group
V0.4 - 04/22/2021 - update by Jason Pippitt NASA/GSFC/SSAI
'''
# ***************************************************************************************

from __future__ import print_function
import numpy as np
from gvradar import common as cm
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain, 
                            csu_dsd, csu_kdp, csu_misc, fundamentals)

# ***************************************************************************************

def dbz_to_zlin(dz):
    """dz = Reflectivity (dBZ), returns Z (mm^6 m^-3)"""
    return 10.0**(dz / 10.0)

# ***************************************************************************************

def zlin_to_dbz(Z):
    """Z (mm^6 m^-3), returns dbz = Reflectivity (dBZ) """
    return 10.0 * np.log10(Z)

# ***************************************************************************************

def add_csu_fhc(radar, dz, dr, rh, kd, radar_band, radar_T):
    
    print('    Add HID field to radar...')

    scores = csu_fhc.csu_fhc_summer(dz=dz, zdr=dr, rho=rh, kdp=kd, use_temp=True, 
                                band=radar_band,
                                T=radar_T)

    fh = np.argmax(scores, axis=0) + 1
    radar = cm.add_field_to_radar_object(fh, radar, field_name = 'FH',
                             units='Unitless',
                             long_name='Hydrometeor ID', 
                             standard_name='Hydrometeor ID',
                             dz_field='CZ')
    return radar, fh

# ***************************************************************************************

def add_csu_liquid_ice_mass(radar, dz, dr, radar_z, radar_T):

    print('    Calculating water and ice mass...')

    mw, mi = csu_liquid_ice_mass.calc_liquid_ice_mass(dz, dr, radar_z, T=radar_T)
    radar = cm.add_field_to_radar_object(mw, radar, field_name='MW', units='g m-3',
                                 long_name='Liquid Water Mass',
                                 standard_name='Liquid Water Mass',
                                 dz_field='CZ')

    radar = cm.add_field_to_radar_object(mi, radar, field_name='MI', units='g m-3',
                                 long_name='Ice Water Mass',
                                 standard_name='Ice Water Mass',
                                 dz_field='CZ')
    return radar

# ***************************************************************************************

def add_csu_blended_rain(radar, dz, dr, kd, fh):

    print('    Calculating blended rainfall field...')

    rain, method = csu_blended_rain.csu_hidro_rain(dz=dz, zdr=dr, kdp=kd, fhc=fh)

    radar = cm.add_field_to_radar_object(rain, radar, field_name='RC', units='mm/h',
                                 long_name='HIDRO Rainfall Rate', 
                                 standard_name='Rainfall Rate',
                                 dz_field='CZ')

    radar = cm.add_field_to_radar_object(method, radar, field_name='MRC', units='',
                                 long_name='HIDRO Rainfall Method', 
                                 standard_name='Rainfall Method',
                                 dz_field='CZ')
    return radar

# ***************************************************************************************

def add_calc_dsd_sband_tokay_2020(radar, dz, dr, location):

    print('    Calculating Drop-Size Distribution...')

    dm, nw = calc_dsd_sband_tokay_2020(dz, dr, loc=location, d0_n2=False)

    radar = cm.add_field_to_radar_object(dm, radar, field_name='DM', units='mm',
                              long_name='Mass-weighted mean diameter',
                              standard_name='Mass-weighted mean diameter',
                              dz_field='CZ')
    radar = cm.add_field_to_radar_object(nw, radar, field_name='NW', units='[Log Nw, m^-3 mm^-1]',
                              long_name='Normalized intercept parameter',
                              standard_name='Normalized intercept parameter',
                              dz_field='CZ')     

    return radar

# ***************************************************************************************

def calc_dsd_sband_tokay_2020(dz, zdr, loc='wff', d0_n2=False):

    """
    Compute dm and nw or (d0 and n2) following the methodology of Tokay et al. 2020
    Works for S-band radars only
    Written by: Charanjit S. Pabla, NASA/WFF/SSAI

    Parameters:
    -----------
    dz: Reflectivity (numpy 2d array)
    zdr: Differential Reflectivity (numpy 2d array)
    
    Keywords:
    -----------
    loc: wff (default, string); region or field campaign name (DSD depends on environment)
         user options: wff, alabama, ifloods, iphex, mc3e, olympex
    d0_n2: False (default, bool)
        if true then function will return d0 and n2

    Return:
    -------
    dm and nw (default, numpy array)
    if d0_n2 set to True then return d0 and n2 (numpy array)
    """ 
    missing = -32767.0 
    dm = np.zeros(dz.shape)
    nw = np.zeros(dz.shape)
    dz_lin = dbz_to_zlin(dz)
    
    #force input string to lower case
    loc = loc.lower()
    
    if not d0_n2:
        
        #compute dm
        if loc == 'wff':
            high = zdr > 3.5
            low = zdr <= 3.5
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.0990 * zdr[low]**3 - 0.6141 * zdr[low]**2 + 1.8364 * zdr[low] + 0.4559
        elif loc == 'alabama':
            high = zdr > 3.8
            low = zdr <= 3.8
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.0453 * zdr[low]**3 - 0.3236 * zdr[low]**2 + 1.2939 * zdr[low] + 0.7065
        elif loc == 'ifloods':
            high = zdr > 3.1
            low = zdr <= 3.1
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.1988 * zdr[low]**3 - 1.0747 * zdr[low]**2 + 2.3786 * zdr[low] + 0.3623
        elif loc == 'iphex':
            high = zdr > 2.9
            low = zdr <= 2.9
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.1887 * zdr[low]**3 - 1.0024 * zdr[low]**2 + 2.3153 * zdr[low] + 0.3834
        elif loc == 'mc3e':
            high = zdr > 3.1
            low = zdr <= 3.1
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.1861 * zdr[low]**3 - 1.0453 * zdr[low]**2 + 2.3804 * zdr[low] + 0.3561
        elif loc == 'olpymex':
            high = zdr > 2.7
            low = zdr <= 2.7
            dm[high] = 0.0188 * zdr[high]**3 - 0.1963 * zdr[high]**2 + 1.1796 * zdr[high] + 0.7183
            dm[low] = 0.2209 * zdr[low]**3 - 1.1577 * zdr[low]**2 + 2.3162 * zdr[low] + 0.3486
    
        #compute nw
        nw = np.log10(35.43 * dz_lin * dm**-7.192)
    
        #set dm and nw missing based on acceptable zdr range
        zdr_bad = np.logical_or(zdr <= 0.0, zdr > 4.0)
        dm[zdr_bad] = missing
        nw[zdr_bad] = missing
    
        #set dm and nw missing based on acceptable dm range
        dm_bad = np.logical_or(dm < 0.5, dm > 4.0)
        dm[dm_bad] = missing
        nw[dm_bad] = missing
    
        #set dm and nw missing based on acceptable nw range
        bad_nw = np.logical_or(nw < 0.5, nw > 6.0)
        nw[bad_nw] = missing
        dm[bad_nw] = missing
        
        return dm, nw
    else:
        #user request d0 and n2
        
        d0 = dm
        n2 = nw
        
        d0 = 0.0215 * zdr**3 - 0.0836 * zdr**2 + 0.7898 * zdr + 0.8019
        n2 = np.log10(20.957 * dz_lin * d0**-7.7)
        
        #set d0 and n2 missing
        d0_bad = d0 <= 0
        n2_bad = n2 <= 0
        d0[d0_bad] = missing
        n2[n2_bad] = missing
        
        return d0, n2

# ***************************************************************************************

def get_kdp(self):

    '''If no KDP field, we need to calculate one.'''
    
    print('', '    Calculating Kdp...', '', sep='\n')
    
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

    return self.radar

# ***************************************************************************************        

def get_default_product_dict():

    default_product_dict = {'cf_dir': './cf/',
                            'do_HID': True,
                            'do_mass': True,
                            'do_RC': True,
                            'do_tokay_DSD': True,
                            'max_range': 150, 
                            'max_height': 10,
                            'sweeps_to_plot': [0],
                            'output_cf': True,
                            'output_fields': ['DZ', 'CZ', 'VR', 'DR', 'KD', 
                                              'PH', 'RH', 'SD', 'SQ', 'FH',
                                              'RC', 'DM', 'NW', 'MW', 'MI'],
                            'plot_images': True,
                            'plot_single': False,
                            'fields_to_plot': ['RC','FH', 'DM', 'NW', 'MW', 'MI'],
                            'plot_dir': './plots/', 'add_logos': True,
                            'use_sounding': True,
                            'sounding_type': 'ruc_archive',
                            'sounding_dir': './sounding/'}

    return default_product_dict

# ***************************************************************************************
