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
import datetime
from copy import deepcopy
from scipy.special import gamma
from gvradar import (common as cm, plot_images as pi)
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain, 
                            csu_dsd, csu_kdp, csu_misc, fundamentals)

# ***************************************************************************************

def dbz_to_zlin(dz):
    """dz = Reflectivity (dBZ), returns Z (mm^6 m^-3)"""
    return 10.**(np.asarray(dz)/10.)

# ***************************************************************************************

def zlin_to_dbz(Z):
    """Z (mm^6 m^-3), returns dbz = Reflectivity (dBZ) """
    return 10. * np.log10(np.asarray(zlin))

# ***************************************************************************************

def add_csu_fhc(self):
    
    # Run Summer HID

    if self.do_HID_summer:
        print('    Add Summer HID field to radar...')
        fh = csu_fhc.csu_fhc_summer(dz=self.dz, zdr=self.dr, rho=self.rh, kdp=self.kd, use_temp=True,
                                    T=self.radar_T, band=self.radar_band, verbose=False,
                                    use_trap=False, method='hybrid')
    
        self.radar = cm.add_field_to_radar_object(fh, self.radar, field_name = 'FH',
                                                  units='Unitless', long_name='Summer Hydrometeor ID', 
                                                  standard_name='Summer Hydrometeor ID', dz_field='CZ') 

    # Run Winter HID
    if self.do_HID_winter:
        print('    Add Winter HID field to radar...')

        if self.scan_type == 'PPI':
            azimuths = self.radar.azimuth['data']
            sn = pyart.retrieve.simple_moment_calculations.calculate_snr_from_reflectivity(self.radar,refl_field='CZ',toa=15000.0)
            sndat = sn['data'][:]
            self.radar = cm.add_field_to_radar_object(sndat,self.radar,field_name='SN',dz_field='CZ')
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
            if self.no_temp:
                fnt = csu_fhc.run_winter(dz=self.dz, zdr=self.dr, kdp=self.kd, rho=self.rh, azimuths=azimuths,
                                    T = None, heights = rheights, 
                                    nsect=nsect, scan_type = self.scan_type, verbose = False, 
                                    use_temp = False, band=self.radar_band, minRH=minRH,
                                    return_scores=False ,sn_thresh=self.snthresh, sn=sndat)
                self.radar = cm.add_field_to_radar_object(fnt, self.radar, field_name = 'NT',
                                                  units='Unitless', long_name='No TEMP Winter Hydrometeor ID',
                                                  standard_name='no TEMP Winter Hydrometeor ID',
                                                  dz_field='CZ')

        self.radar = cm.add_field_to_radar_object(fw, self.radar, field_name = 'FW',
                                                  units='Unitless', long_name='Winter Hydrometeor ID',
                                                  standard_name='Winter Hydrometeor ID',
                                                  dz_field='CZ')                            

    return self.radar
# ***************************************************************************************

def add_csu_liquid_ice_mass(self):

    print('    Calculating water and ice mass...')

    try:
        mw, mi = csu_liquid_ice_mass.calc_liquid_ice_mass(self.dz, self.dr, self.radar_z/1000.0, T=self.radar_T)
    except:
        print(' ',"No precip MW and MI will be -32767.0", '', sep='\n')
        mw = self.radar.fields['CZ']['data'].copy()
        mi = self.radar.fields['CZ']['data'].copy()

    # HID ice threshold
    #mw = remove_ice(mw, self.fh)
    #mi = remove_ice(mi, self.fh)

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

    rain, method = csu_blended_rain.csu_hidro_rain(dz=self.dz, zdr=self.dr, 
                                                   kdp=self.kd, fhc=self.fh,
                                                   band=self.radar_band)

    # Set fill to zero
    rain = np.ma.filled(rain, fill_value=0.0)

    # Max rain rate test
    rc_max = np.greater(rain,300)
    rain[rc_max] = rain[rc_max] * -1.0

    # HID ice threshold
    rain = remove_ice(rain, self.fh)

    rc_dict = {"data": rain, "units": "mm/h",
                "long_name": "HIDRO Rainfall Rate", "_FillValue": -32767.0,
                "standard_name": "HIDRO Rainfall Rate",}
    self.radar.add_field("RC", rc_dict, replace_existing=True)
    
    return self.radar
# ***************************************************************************************

def add_polZR_rr(self):

    rp = np.zeros((self.radar.nrays, self.radar.ngates), dtype=float)
    #rp = 0.0 * self.dz

    use_nw = False
    if use_nw:
        # If NW exsits use to compute RP
        print('    Calculating PolZR rain rate with Ali NW')
        nw = self.radar.fields['NW']['data']
        rp = get_bringi_rainrate_nw(rp,self.dz,self.dr,self.kd,self.rh,nw,self.fh)
    else:
        # IF no NW compute with equations
        print('    Calculating PolZR rain rate with computed NW')
        rp, nw = get_bringi_rainrate(self,rp,self.dz,self.dr,self.kd,self.rh,self.fh)

    # Set fill to zero
    rp = np.ma.filled(rp, fill_value=0.0)

    # Max rain rate test
    rp_max = np.greater(rp,300)
    rp[rp_max] = rp[rp_max] * -1.0

    # HID ice threshold
    rp = remove_ice(rp, self.fh)

    rp_dict = {"data": rp, "units": "mm/h",
               "long_name": "Polzr_Rain_Rate", "_FillValue": -32767.0,
               "standard_name": "Polzr_Rain_Rate",}
    self.radar.add_field("RP", rp_dict, replace_existing=True)

    return self.radar                                     

# ***************************************************************************************

def add_calc_dsd_sband_tokay_2020(self):

    print('    Calculating Drop-Size Distribution...')

    if self.site == 'NPOL':
        radar_DT = pyart.util.datetime_from_radar(self.radar)
        DT_beg_IPHEX = datetime.datetime(*map(int, ['2014','04','01','0','0','0']))
        DT_end_IPHEX = datetime.datetime(*map(int, ['2014','06','30','23','59','59']))
        DT_beg_OLYMPEX = datetime.datetime(*map(int, ['2015','11','01','0','0','0']))
        DT_end_OLYMPEX = datetime.datetime(*map(int, ['2016','01','31','23','59','59']))
        if (radar_DT >= DT_beg_IPHEX) & (radar_DT <= DT_end_IPHEX):  
            self.dsd_loc = 'iphex'
        elif (radar_DT >= DT_beg_OLYMPEX) & (radar_DT <= DT_end_OLYMPEX): 
            self.dsd_loc = 'iphex'
        else:
            self.dsd_loc = 'wff'

    dm, nw = calc_dsd_sband_tokay_2020(self, self.dz, self.dr, loc=self.dsd_loc)

    # Set fill to zero
    dm = np.ma.filled(dm, fill_value=0.0)
    nw = np.ma.filled(nw, fill_value=0.0)

    # HID ice threshold
    dm = remove_ice(dm, self.fh)
    nw = remove_ice(nw, self.fh)

    dm_dict = {"data": dm, "units": "DM [mm]",
                "long_name": "Mass-weighted mean diameter", "_FillValue": -32767.0,
                "standard_name": "Mass-weighted mean diameter",}
    self.radar.add_field("DM", dm_dict, replace_existing=True)
    
    nw_dict = {"data": nw, "units": "Log[Nw, m^-3 mm^-1]",
                "long_name": "Normalized intercept parameter", "_FillValue": -32767.0,
                "standard_name": "Normalized intercept parameter",}
    self.radar.add_field("NW", nw_dict, replace_existing=True)

    return self.radar

# ***************************************************************************************

def calc_dsd_sband_tokay_2020(self, dz, zdr, loc='all'):

    """
    Compute dm and nw or (d0 and n2) following the methodology of Tokay et al. 2020
    Works for S-band radars only

    Parameters:
    -----------
    dz: Reflectivity (numpy 2d array)
    zdr: Differential Reflectivity (numpy 2d array)
    
    Keywords:
    -----------
    loc: all (default, string); region or field campaign name (DSD depends on environment)
         user options: wff, alabama, ifloods, iphex, mc3e, olympex, all

    Return:
    -------
    dm and nw (default, numpy array)
    """
    dm = np.zeros((self.radar.nrays, self.radar.ngates), dtype=float)
    nw = np.zeros((self.radar.nrays, self.radar.ngates), dtype=float)
    #dm = 0.0 * dz
    #nw = 0.0 * dz
    dz_lin = dbz_to_zlin(dz)
    
    # Force input string to lower case
    loc = loc.lower()
    
    # Compute dm for valid ZDR
    print('    DSD equation:  ',loc)
    if loc == 'wff':
        high = np.logical_and(zdr > 3.5, zdr <= 4.0)
        dm[high] = get_dm(zdr[high],0.0138,0.1696,1.1592,0.7215)
        low = np.logical_and(zdr <= 3.5, zdr > 0.0)
        dm[low] = get_dm(zdr[low],0.0990,0.6141,1.8364,0.4559)
    elif loc == 'alabama':
        high = np.logical_and(zdr > 3.1, zdr <= 4.0)
        dm[high] = get_dm(zdr[high],0.0138,0.1696,1.1592,0.7215)
        low = np.logical_and(zdr <= 3.1, zdr > 0.0)
        dm[low] = get_dm(zdr[low],0.0782,0.4679,1.5355,0.6377)
    elif loc == 'ifloods':
        high = np.logical_and(zdr > 3.1, zdr <= 4.0)
        dm[high] = get_dm(zdr[high],0.0138,0.1696,1.1592,0.7215)
        low = np.logical_and(zdr <= 3.1, zdr > 0.0)
        dm[low] = get_dm(zdr[low],0.1988,1.0747,2.3786,0.3623)
    elif loc == 'iphex':
        high = np.logical_and(zdr > 2.9, zdr <= 4.0)
        dm[high] = get_dm(zdr[high],0.0138,0.1696,1.1592,0.7215)
        low = np.logical_and(zdr <= 2.9, zdr > 0.0)
        dm[low] = get_dm(zdr[low],0.1887,1.0024,2.3153,0.3834)
    elif loc == 'mc3e':
        high = np.logical_and(zdr > 3.1, zdr <= 4.0)
        dm[high] = get_dm(zdr[high],0.0138,0.1696,1.1592,0.7215)
        low = np.logical_and(zdr <= 3.1, zdr > 0.0)
        dm[low] = get_dm(zdr[low],0.1861,1.0453,2.3804,0.3561)
    elif loc == 'olpymex':
        high = np.logical_and(zdr > 2.7, zdr <= 4.0)
        dm[high] = get_dm(zdr[high],0.0138,0.1696,1.1592,0.7215)
        low = np.logical_and(zdr <= 2.7, zdr > 0.0)
        dm[low] = get_dm(zdr[low],0.2209,1.1577,2.3162,0.3486)
    elif loc == 'all':
        good_zr = np.logical_and(zdr > 0.0, zdr <= 4.0)
        dm[good_zr] = get_dm(zdr[good_zr],0.0138,0.1696,1.1592,0.7215)
        
    # Compute nw for valid dm and log(nw)
    dm_range = np.logical_and(dm >= 0.5, dm <= 4.0)
    nw[dm_range] = 35.43 * dz_lin[dm_range] * dm[dm_range]**-7.192
    nw[dm_range] = np.log10(nw[dm_range])
    
    # Set dm and nw based on acceptable dm range
    bad_dm = np.less(dm,0.5)
    nw[bad_dm] = -1.0 * nw[bad_dm]
    dm[bad_dm] = -1.0 * dm[bad_dm]
    dm4 = np.greater(dm,4)
    nw[dm4] = -1.0 * nw[dm4]
    dm[dm4] = -1.0 * dm[dm4]
    
    # Set nw based on acceptable nw range
    bad_nw = np.less(nw,0.5)
    nw[bad_nw] = -1.0 * nw[bad_nw]
    nw6 = np.greater(nw,6)
    nw[nw6] = -1.0 * nw[nw6]
    
    return dm, nw

# ***************************************************************************************

def get_dm(zdr,a,b,c,d):
    
    return a * zdr**3 - b * zdr**2 + c * zdr + d

# ***************************************************************************************

def get_bringi_rainrate(self,rp,dbz,zdr,kdp,rhv,hid):

    #Calculates DSD fields to assign a rain rate.  

    #Disdrometer-based Z-R coefficients
    #a1 = 0.024285, b1 = 0.6895, Z = 219 Z^1.45
    #a1 = 0.017, b1 = 0.7143, Z = 300 Z^1.4 
    #a1 = 0.036. b1 = 0.6250, Z = 200 R^1.6 
    #a_dsd = (1.0/a1)**(1./b1)
    #b_dsd = (1.0/b1)

    #Check to see if DP variables can be used to calculate 
    #the rain rate. We need D0, Nw and mu to calculate the 
    #Polarimetric ZR

    zh = 10.**(0.1*dbz)     
    xi_dr = 10.**(0.1*zdr)
    d0 = np.zeros((self.radar.nrays, self.radar.ngates), dtype=float)
    dm = np.zeros((self.radar.nrays, self.radar.ngates), dtype=float)
    nw = np.zeros((self.radar.nrays, self.radar.ngates), dtype=float)
    logNw = np.zeros((self.radar.nrays, self.radar.ngates), dtype=float)
    mu = np.ones((self.radar.nrays, self.radar.ngates), dtype=float) * 3.0
    beta = 0

    # Light rain rates with noisy Zd
    light_rain = np.logical_and(zdr >= -0.5, zdr < 0.2)
    nw[light_rain] = get_nw_light(zh[light_rain],xi_dr[light_rain])
    
    # Light rain rates with modest kdp
    modest_kdp = np.logical_and(kdp < 0.3, zdr >= 0.2, zdr < 0.5)
    nw[modest_kdp] = get_nw_modest(zdr[modest_kdp],zh[modest_kdp],xi_dr[modest_kdp])
    
    # Moderate rain rates
    moderate_rr = np.logical_and(kdp < 0.3, zdr >= 0.5)
    nw[moderate_rr] = get_moderate_nw(zdr[moderate_rr],zh[moderate_rr])
    
    # Heavy rain rates
    heavy_rr = np.logical_and(dbz >= 35, kdp >= 0.3, zdr >= 0.2)
    nw[heavy_rr], mu[heavy_rr] = get_heavy_nw(zdr[heavy_rr],kdp[heavy_rr],zh[heavy_rr],xi_dr[heavy_rr])
           
    # Calculate the coefficient a' in Z = a' * R^1.5 using the DSD
    # parameters. First, calculate f(mu)
    rp = get_polzr_rainrate(dbz,nw,mu)
    
    nw = np.log10(nw)
    
    return rp, nw

# ***************************************************************************************

def get_nw_light(zh,xi_dr):

    d0 = 0.6096*(zh**0.0516)*(xi_dr**3.111)
    nw = (21*zh)/d0**7.3529
    mu = 3.0
    dm = d0 * ((4+mu)/(3.67+mu))
    
    return nw

# ***************************************************************************************

def get_nw_modest(zdr,zh,xi_dr):

    x1 = ((zdr - 0.2)/0.3) * 1.81 * zdr**(0.486)
    x2 = ((0.5-zdr)/0.3) * 0.6096*zh**(0.0516) * xi_dr**(3.111)
    d0 = x1 + x2
    nw = (21*zh)/d0**7.3529
    mu = 3.0
    dm = d0 * ((4+mu)/(3.67+mu))
    
    return nw

# ***************************************************************************************

def get_moderate_nw(zdr,zh):

    d0 = 1.81*zdr**0.486
    nw = (21*zh)/d0**7.3529
    mu = 3.0
    dm = d0 * ((4+mu)/(3.67+mu))
    
    return nw

# ***************************************************************************************

def get_heavy_nw(zdr,kdp,zh,xi_dr):

    beta = 2.08*zh**(-0.365) * kdp**(0.38) * xi_dr**(0.965)
    a1 = 0.56
    b1 = 0.064
    c1 = 0.024*beta**(-1.42)
    d0 = a1 * zh**b1 * xi_dr**c1
    a2 = 3.29
    b2 = 0.058
    c2 = -0.023 * beta**(-1.389)
    logNw = a2 * zh**b2 * xi_dr**c2
    nw = 10**(logNw)
    a3 = 203. * beta**(1.89)
    b3 = 2.23 * beta**(0.0388)
    c3 = 3.16 * beta**(-0.0463)
    d3 = 0.374 * beta**(-0.355)
    x1 = a3 * (d0**b3)/(xi_dr-1)
    x2 = c3 * (xi_dr**d3)
    mu = x1 - x2
    dm = d0 * ((4+mu)/(3.67+mu))    
    
    return nw, mu

# ***************************************************************************************

def get_bringi_rainrate_nw(rp,dbz,zdr,kdp,rhv,nw,fh):

    # Calculate the coefficient a' in Z = a' * R^1.5 using the DSD
    # parameters. First, calculate f(mu)
    mu = 3.0
    nw = 10**nw
    rp = get_polzr_rainrate(dbz,nw,mu)
    
    return rp

# ***************************************************************************************

def get_polzr_rainrate(dbz,nw,mu):
    
    # From eq. A.25 in Bring et al. 2004
    x1 = 6.0 * (3.67 + mu)**(4+mu)
    x2 = (3.67**4) * gamma(mu+4)
    f_mu = x1/x2

    # From eq. A.23 in Bring et al. 2004
    x3 = f_mu * gamma(7+mu)
    x4 = (3.67+mu)**(7+mu)
    fz_mu = x3/x4

    # From eq. A.24 in Bring et al. 2004
    x5  = np.pi * 0.0006 * 3.78 * f_mu
    x6  = gamma(4.67 + mu) / (3.67+mu)**(4.67+mu)

    fr_mu = x5 * x6
    
    a_mu = fz_mu / (fr_mu**(1.5))

    # Now, the new A parameter for the Z-R relation is given by:
    # a' = a_mu/(nw^1.5) 

    a_prime = a_mu/(nw**0.5)
    b = 1.5
    rp =  get_zr_rain(dbz, a_prime, b)

    return rp

# ***************************************************************************************

def get_zr_rain(dbz, a, b):
    
    # Set max_dbz dB as maximum reasonable reflectivity
    max_dbz = 55
    dbz_max = np.greater_equal(dbz,max_dbz)
    dbz[dbz_max] = max_dbz

    # Now calculate the rain rate from the pass dBZ value
    zh = 10**(dbz/10.0)
    rp = (zh/a)**(1.0/b)
    
    return rp

# ***************************************************************************************

def remove_ice(fl,hid):
    
    hid_ice = [0, 3, 4, 5, 6, 7, 8, 9]
    for xice in hid_ice:
        ice = np.equal(hid, xice)
        fl[ice] = -999
        
    return fl

# ***************************************************************************************

def set_low_dbz(fl,zz):

    #Use raw reflectivity since corrected reflectivity masks values <= 5
    low_dbz = np.logical_and(zz > 0, zz <= 5)
    fl[low_dbz] = 0.0
    
    return fl

# ***************************************************************************************

def mask_beyond_150(self):

    """
    Filter out any data outside 150 KM set to -32767.0
    """
    
    sector = {'hmin': 0, 'hmax': None,
              'rmin': 150 * 1000, 'rmax':  400 * 1000,
              'azmin': 0, 'azmax': 360,
              'elmin': 0, 'elmax': None}
    
    beyond_flag = np.ma.ones((self.radar.nrays, self.radar.ngates), dtype=int)
    
    # check for altitude limits
    if sector['hmin'] is not None:
        beyond_flag[self.radar.gate_altitude['data'] < sector['hmin']] = 0
    if sector['hmax'] is not None:
        beyond_flag[self.radar.gate_altitude['data'] > sector['hmax']] = 0

    # check for range limits

    if sector['rmin'] is not None:
        beyond_flag[:, self.radar.range['data'] < sector['rmin']] = 0

    if sector['rmax'] is not None:
        beyond_flag[:, self.radar.range['data'] > sector['rmax']] = 0

    # check elevation angle limits
    if sector['elmin'] is not None:
        beyond_flag[self.radar.elevation['data'] < sector['elmin'], :] = 0

    if sector['elmax'] is not None:
        beyond_flag[self.radar.elevation['data'] > sector['elmax'], :] = 0

    # check min and max azimuth angle
    if sector['azmin'] is not None and sector['azmax'] is not None:
        if sector['azmin'] <= sector['azmax']:
            beyond_flag[self.radar.azimuth['data'] < sector['azmin'], :] = 0
            beyond_flag[self.radar.azimuth['data'] > sector['azmax'], :] = 0
        if sector['azmin'] > sector['azmax']:
            beyond_flag[np.logical_and(
            self.radar.azimuth['data'] < sector['azmin'],
            self.radar.azimuth['data'] > sector['azmax']), :] = 0
    elif sector['azmin'] is not None:
        beyond_flag[self.radar.azimuth['data'] < sector['azmin'], :] = 0
    elif sector['azmax'] is not None:
        beyond_flag[self.radar.azimuth['data'] > sector['azmax'], :] = 0

    beyond_field = beyond_flag
    apply_beyond = np.equal(beyond_field,1)

    fields = []
    cm_fields = ['FH','FW','RC','RP','MW','MI','DM','NW']
    for fld in cm_fields:
        if fld in self.radar.fields.keys():
            fields.append(fld)
    for fld in fields:
        nf = self.radar.fields[fld]['data']
        nf[apply_beyond] = -32767.0
        self.radar.add_field_like(fld,fld,nf,replace_existing=True)

    return self.radar

# ***************************************************************************************

def set_blockage(self, sector_dict):

    """
    Set known blockages to -888
    """

    for k in range(len(sector_dict)):
        
        sector = sector_dict[k]
        
        block_flag = np.ma.ones((self.radar.nrays, self.radar.ngates), dtype=int)
    
        # check for altitude limits
        if sector['hmin'] is not None:
            block_flag[self.radar.gate_altitude['data'] < sector['hmin']] = 0
        if sector['hmax'] is not None:
            block_flag[self.radar.gate_altitude['data'] > sector['hmax']] = 0

        # check for range limits

        if sector['rmin'] is not None:
            block_flag[:, self.radar.range['data'] < sector['rmin']] = 0

        if sector['rmax'] is not None:
            block_flag[:, self.radar.range['data'] > sector['rmax']] = 0

        # check elevation angle limits
        if sector['elmin'] is not None:
            block_flag[self.radar.elevation['data'] < sector['elmin'], :] = 0

        if sector['elmax'] is not None:
            block_flag[self.radar.elevation['data'] > sector['elmax'], :] = 0

        # check min and max azimuth angle
        if sector['azmin'] is not None and sector['azmax'] is not None:
            if sector['azmin'] <= sector['azmax']:
                block_flag[self.radar.azimuth['data'] < sector['azmin'], :] = 0
                block_flag[self.radar.azimuth['data'] > sector['azmax'], :] = 0
            if sector['azmin'] > sector['azmax']:
                block_flag[np.logical_and(
                self.radar.azimuth['data'] < sector['azmin'],
                self.radar.azimuth['data'] > sector['azmax']), :] = 0
        elif sector['azmin'] is not None:
            block_flag[self.radar.azimuth['data'] < sector['azmin'], :] = 0
        elif sector['azmax'] is not None:
            block_flag[self.radar.azimuth['data'] > sector['azmax'], :] = 0
         
        globals()['block_flag_%s' % k] = block_flag 
    
    block_flag = np.ma.zeros((self.radar.nrays, self.radar.ngates), dtype=int)
    for k in range(len(sector_dict)):
        globals()['apply_block_%s' % k] = np.equal(globals()['block_flag_%s' % k],1)
        block_flag[globals()['apply_block_%s' % k]] = 1
    
    block_field = block_flag
    apply_block = np.equal(block_flag,1)
    
    fields = []
    all_fields = ['CZ','DR','KD','PH','RH','SD','SW','VR','FS','FW','RC','RP','MW','MI','DM','NW']
    for fld in all_fields:
        if fld in self.radar.fields.keys():
            fields.append(fld)
    for fld in fields:
        nf = self.radar.fields[fld]['data']
        nf[apply_block] = -888
        self.radar.add_field_like(fld,fld,nf,replace_existing=True)
    
    block_dict = {"data": block_field, "units": "0: False, 1: True",
                  "long_name": "BLOCK", "_FillValue": -32767.0,
                  "standard_name": "BLOCK",}
    self.radar.add_field("BLOCK", block_dict, replace_existing=False)
    
    return self.radar

# *************************************************************************************** 

def get_kdp(self):

    '''If no KDP field, we need to calculate one.'''
    
    print('', '    Calculating Kdp...', '', sep='\n')
    
    DZ = cm.extract_unmasked_data(self.radar, self.ref_field_name)
    DP = cm.extract_unmasked_data(self.radar, self.phi_field_name)

    # Range needs to be supplied as a variable, with same shape as DZ
    rng2d, az2d = np.meshgrid(self.radar.range['data'], self.radar.azimuth['data'])
    gate_spacing = self.radar.range['meters_between_gates']

    if self.site == 'KWAJ':
        window=4
    else:
        window=4

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
                            'no_temp': False,
                            'snthresh': -30,
                            'do_mass': True,
                            'do_RC': True,
                            'do_RP': True,
                            'do_tokay_DSD': True,
                            'dsd_loc': 'all',
                            'do_150_mask': True,
                            'do_block_mask': False,
                            'get_Bringi_kdp': False,
                            'max_range': 200, 
                            'max_height': 10,
                            'sweeps_to_plot': [0],
                            'output_cf': False,
                            'output_grid': False,
                            'cf_dir': './cf',
                            'grid_dir': './grid',
                            'output_fields': ['DZ', 'CZ', 'VR', 'DR', 'KD', 
                                              'PH', 'RH', 'SD', 'SQ', 'FH',
                                              'RC', 'DM', 'NW', 'MW', 'MI',
                                              'RP', 'FW'],
                            'plot_images': True,
                            'plot_fast': False,
                            'mask_outside': True,
                            'plot_single': True,
                            'plot_multi': False,
                            'fields_to_plot': ['CZ', 'DR', 'KD', 'RH', 'RC', 'DM', 'NW', 'FH', 'RP'],
                            'plot_dir': './plots/', 'add_logos': True,
                            'use_sounding': True,
                            'sounding_type': 'ruc_archive',
                            'sounding_dir': './sounding/'}

    return default_product_dict

# ***************************************************************************************
