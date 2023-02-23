# ***************************************************************************************
'''
Rainfall product generation from Dual Pol data, utilizing PyArt.

Developed by the NASA GPM-GV group
V0.5 - 12/06/2021 - update by Jason Pippitt NASA/GSFC/SSAI
V1.0 - 11/01/2022 - update by Jason Pippitt NASA/GSFC/SSAI
'''
# ***************************************************************************************

from __future__ import print_function
import numpy as np
import pyart
from gvradar import (common as cm, plot_images as pi)

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

def add_csu_fhc(self):
    
    #Run Summer HID

    if self.do_HID_summer:
        print('    Add Summer HID field to radar...')
        fh = csu_fhc.csu_fhc_summer(dz=self.dz, zdr=self.dr, rho=self.rh, kdp=self.kd, use_temp=True,
                                    T=self.radar_T, band=self.radar_band, verbose=False,
                                    use_trap=False, method='hybrid')
    
        self.radar = cm.add_field_to_radar_object(fh, self.radar, field_name = 'FS',
                                                  units='Unitless', long_name='Summer Hydrometeor ID', 
                                                  standard_name='Summer Hydrometeor ID', dz_field='CZ')

        self.fh = self.radar.fields['FS']['data']  

    #Run Winter HID
    if self.do_HID_winter:
        print('    Add Winter HID field to radar...')

        if self.scan_type == 'PPI':
            azimuths = self.radar.azimuth['data']
            sn = pyart.retrieve.simple_moment_calculations.calculate_snr_from_reflectivity(self.radar,refl_field='CZ',toa=15000.0)
            sndat = sn['data'][:]
            radar = cm.add_field_to_radar_object(sndat,self.radar,field_name='SN',dz_field='CZ')
            sndat = np.ma.masked_array(sndat)
            nsect = 36
        
        rheights = self.radar_z/1000.

        minRH = 0.5
        if self.scan_type == 'RHI':
           fw = csu_fhc.run_winter(dz=self.dz, zdr=self.dr, kdp=self.kd, rho=self.rh, 
                                   expected_ML=self.expected_ML, sn = None, T = self.radar_T, 
                                   heights = rheights, scan_type = self.scan_type, verbose = False,
                                   use_temp = True, band = self.radar_band, minRH=minRH,
                                   return_scores = False)
        else:
            fw = csu_fhc.run_winter(dz=self.dz, zdr=self.dr, kdp=self.kd, rho=self.rh, azimuths=azimuths,
                                    expected_ML=self.expected_ML, T = self.radar_T, heights = rheights, 
                                    nsect=nsect, scan_type = self.scan_type, verbose = False, 
                                    use_temp = True, band=self.radar_band, minRH=minRH,
                                    return_scores=False ,sn_thresh=self.snthresh, sn=sndat)

        self.radar = cm.add_field_to_radar_object(fw, self.radar, field_name = 'FW',
                                                  units='Unitless', long_name='Winter Hydrometeor ID',
                                                  standard_name='Winter Hydrometeor ID',
                                                  dz_field='CZ')                            

        self.fw = self.radar.fields['FW']['data']

    return self.radar
# ***************************************************************************************

def add_csu_liquid_ice_mass(self):

    print('    Calculating water and ice mass...')

    mw, mi = csu_liquid_ice_mass.calc_liquid_ice_mass(self.dz, self.dr, self.radar_z/1000.0, T=self.radar_T)
    self.radar = cm.add_field_to_radar_object(mw, self.radar, field_name='MW', units='g m-3',
                                 long_name='Liquid Water Mass',
                                 standard_name='Liquid Water Mass',
                                 dz_field='CZ')

    self.radar = cm.add_field_to_radar_object(mi, self.radar, field_name='MI', units='g m-3',
                                 long_name='Ice Water Mass',
                                 standard_name='Ice Water Mass',
                                 dz_field='CZ')
    return self.radar

# ***************************************************************************************

def add_csu_blended_rain(self):

    print('    Calculating blended rainfall field...')

    rain, method = csu_blended_rain.csu_hidro_rain(dz=self.dz, zdr=self.dr, kdp=self.kd, fhc=self.fh)

    self.radar = cm.add_field_to_radar_object(rain, self.radar, field_name='RC', units='mm/h',
                                 long_name='HIDRO Rainfall Rate', 
                                 standard_name='Rainfall Rate',
                                 dz_field='CZ')

    self.radar = cm.add_field_to_radar_object(method, self.radar, field_name='MRC', units='',
                                 long_name='HIDRO Rainfall Method', 
                                 standard_name='Rainfall Method',
                                 dz_field='CZ')
    return self.radar

# ***************************************************************************************

def add_calc_dsd_sband_tokay_2020(self):

    print('    Calculating Drop-Size Distribution...')

    dm, nw = calc_dsd_sband_tokay_2020(self.dz, self.dr, loc=self.dsd_loc, d0_n2=False)

    self.radar = cm.add_field_to_radar_object(dm, self.radar, field_name='DM', units='mm',
                              long_name='Mass-weighted mean diameter',
                              standard_name='Mass-weighted mean diameter',
                              dz_field='CZ')
    self.radar = cm.add_field_to_radar_object(nw, self.radar, field_name='NW', units='[Log Nw, m^-3 mm^-1]',
                              long_name='Normalized intercept parameter',
                              standard_name='Normalized intercept parameter',
                              dz_field='CZ')     

    return self.radar

# ***************************************************************************************

def calc_dsd_sband_tokay_2020(dz, zdr, loc='all', d0_n2=False):

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
    loc: all (default, string); region or field campaign name (DSD depends on environment)
         user options: wff, alabama, ifloods, iphex, mc3e, olympex, all
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
        print('    DSD equation:  ',loc)
        if loc == 'wff':
            high = zdr > 3.5
            low = zdr <= 3.5
            dm[high] = 0.0138 * zdr[high]**3 - 0.1696 * zdr[high]**2 + 1.1592 * zdr[high] + 0.7215
            dm[low] = 0.0990 * zdr[low]**3 - 0.6141 * zdr[low]**2 + 1.8364 * zdr[low] + 0.4559
        elif loc == 'alabama':
            high = zdr > 3.1
            low = zdr <= 3.1
            dm[high] = 0.0138 * zdr[high]**3 - 0.1696 * zdr[high]**2 + 1.1592 * zdr[high] + 0.7215
            dm[low] = 0.0782 * zdr[low]**3 - 0.4679 * zdr[low]**2 + 1.5355 * zdr[low] + 0.6377
        elif loc == 'ifloods':
            high = zdr > 3.1
            low = zdr <= 3.1
            dm[high] = 0.0138 * zdr[high]**3 - 0.1696 * zdr[high]**2 + 1.1592 * zdr[high] + 0.7215
            dm[low] = 0.1988 * zdr[low]**3 - 1.0747 * zdr[low]**2 + 2.3786 * zdr[low] + 0.3623
        elif loc == 'iphex':
            high = zdr > 2.9
            low = zdr <= 2.9
            dm[high] = 0.0138 * zdr[high]**3 - 0.1696 * zdr[high]**2 + 1.1592 * zdr[high] + 0.7215
            dm[low] = 0.1887 * zdr[low]**3 - 1.0024 * zdr[low]**2 + 2.3153 * zdr[low] + 0.3834
        elif loc == 'mc3e':
            high = zdr > 3.1
            low = zdr <= 3.1
            dm[high] = 0.0138 * zdr[high]**3 - 0.1696 * zdr[high]**2 + 1.1592 * zdr[high] + 0.7215
            dm[low] = 0.1861 * zdr[low]**3 - 1.0453 * zdr[low]**2 + 2.3804 * zdr[low] + 0.3561
        elif loc == 'olpymex':
            high = zdr > 2.7
            low = zdr <= 2.7
            dm[high] = 0.0138 * zdr[high]**3 - 0.1696 * zdr[high]**2 + 1.1592 * zdr[high] + 0.7215
            dm[low] = 0.2209 * zdr[low]**3 - 1.1577 * zdr[low]**2 + 2.3162 * zdr[low] + 0.3486
        elif loc == 'all':
            dm = 0.0138 * zdr**3 - 0.1696 * zdr**2 + 1.1592 * zdr + 0.7215

        #compute nw
        nw = np.log10(35.43 * dz_lin * dm**-7.192)
    
        #set dm and nw missing based on acceptable zdr range
        #zdr_bad = np.logical_or(zdr <= 0.0, zdr > 4.0)
        zdr_bad = np.logical_and(np.logical_or(zdr <= 0.0, zdr > 4.0),np.abs(dm)>0)
        dm[zdr_bad] = missing
        nw[zdr_bad] = missing
    
        #set dm and nw missing based on acceptable dm range
        #dm_bad = np.logical_or(dm < 0.5, dm > 4.0)
        dm_bad = np.logical_and(np.logical_or(dm < 0.5, dm > 4.0),np.abs(dm)>0)
        dm[dm_bad] = missing
        nw[dm_bad] = missing
    
        #set dm and nw missing based on acceptable nw range
        #bad_nw = np.logical_or(nw < 0.5, nw > 6.0)
        bad_nw = np.logical_and(np.logical_or(nw < 0.5, nw > 6.0),np.abs(dm)>0)
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
    
    DZ = cm.extract_unmasked_data(self.radar, self.ref_field_name)
    DP = cm.extract_unmasked_data(self.radar, self.phi_field_name)
#    DZ = self.radar.fields[self.ref_field_name]['data'].copy()
#    DP = self.radar.fields[self.phi_field_name]['data'].copy()

    # Range needs to be supplied as a variable, with same shape as DZ
    rng2d, az2d = np.meshgrid(self.radar.range['data'], self.radar.azimuth['data'])
    gate_spacing = self.radar.range['meters_between_gates']

    if self.site == 'KWAJ':
        window=4
    else:
        window=5

    KDPB, PHIDPB, STDPHIB = csu_kdp.calc_kdp_bringi(dp=DP, dz=DZ, rng=rng2d/1000.0, 
                                                    thsd=25, gs=gate_spacing, 
                                                    window=window, nfilter=1, std_gate=15)

    self.radar = cm.add_field_to_radar_object(KDPB, self.radar, field_name='KD', 
		units='deg/km',
		long_name='Specific Differential Phase (Bringi)',
		standard_name='Specific Differential Phase (Bringi)',
		dz_field=self.ref_field_name)

    return self.radar
# ***************************************************************************************  

def get_conv_strat(self):
    
    # interpolate to grid
    cs_grid = pyart.map.grid_from_radars(
    (self.radar,), grid_shape=(1, 201, 201),
    grid_limits=((0, 10000), (-200000.0, 200000.0), (-200000.0, 200000.0)),
    fields=['CZ'])

    # get dx dy
    dx = cs_grid.x['data'][1] - cs_grid.x['data'][0]
    dy = cs_grid.y['data'][1] - cs_grid.y['data'][0]

    # convective stratiform classification
    convsf_dict = pyart.retrieve.conv_strat_yuter(cs_grid, dx, dy, refl_field='CZ', always_core_thres=40,
                                              bkg_rad_km=20, use_cosine=True, max_diff=5, zero_diff_cos_val=55,
                                              weak_echo_thres=10, max_conv_rad_km=2)

    # add to grid object
    # mask zero values (no surface echo)
    convsf_masked = np.ma.masked_equal(convsf_dict['convsf']['data'], 0)
    # mask 3 values (weak echo)
    convsf_masked = np.ma.masked_equal(convsf_masked, 3)
    # add dimension to array to add to grid object
    convsf_dict['convsf']['data'] = convsf_masked[None,:,:]
    # add field
    cs_grid.add_field('convsf', convsf_dict['convsf'], replace_existing=True)

    self.ds = cs_grid.to_xarray()
    self.cs_grid = cs_grid

    if self.plot_conv_strat == True:
        pi.plot_conv_strat(self)

    

# ***************************************************************************************        

def get_default_product_dict():

    default_product_dict = {'cf_dir': './cf/',
                            'do_HID_summer': True,
                            'do_HID_winter':  False,
                            'radar_band': 'S',
                            'snthresh': -30,
                            'do_mass': True,
                            'do_RC': True,
                            'do_tokay_DSD': True,
                            'dsd_loc': 'all',
                            'max_range': 200, 
                            'max_height': 10,
                            'sweeps_to_plot': [0],
                            'output_cf': False,
                            'output_grid': False,
                            'cf_dir': './cf',
                            'grid_dir': './grid',
                            'output_fields': ['DZ', 'CZ', 'VR', 'DR', 'KD', 
                                              'PH', 'RH', 'SD', 'SQ', 'FH',
                                              'RC', 'DM', 'NW', 'MW', 'MI'],
                            'plot_images': True,
                            'plot_single': True,
                            'plot_multi': False,
                            'fields_to_plot': ['CZ', 'DR', 'KD', 'RH', 'RC', 'DM', 'NW', 'FH'],
                            'plot_dir': './plots/', 'add_logos': True,
                            'use_sounding': True,
                            'sounding_type': 'ruc_archive',
                            'sounding_dir': './sounding/'}

    return default_product_dict

# ***************************************************************************************
