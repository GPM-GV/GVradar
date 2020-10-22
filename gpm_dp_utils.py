from __future__ import print_function
import numpy as np
import math
import os, sys, glob
import pyart
from skewt import SkewT
import datetime
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain, 
                            csu_dsd, csu_kdp, csu_misc, fundamentals)

# ******************************************************************************************************

def get_files_within_time_window(DP_product_dict):

    #Get full filelist for this day
    wc = DP_product_dict['in_dir'] + '*'
    all_files = sorted(glob.glob(wc))
    nf = len(all_files)
    if(nf == 0):
        print("No files found in " + wc)
        sys.exit("Bye.")

    #Get file start and end times and date    
    stime = DP_product_dict['StartTime']
    etime = DP_product_dict['EndTime']

    DT_beg = datetime.datetime(*map(int, stime))
    DT_end = datetime.datetime(*map(int, etime))

    # NPOL1_2020_0430_152502_PPI.cf np1200430164431.RAWAJB8.gz
    # ['NPOL1', '2020', '0430', '153919', 'PPI.cf']
    files = []
    for file in all_files:
        fileb = os.path.basename(file)
        y = fileb.split('.')
        cfy = (y[1])
        if cfy == 'cf':
            x = fileb.split('_')
            year  = int(x[1])
            month = int(x[2][0:2])
            day   = int(x[2][2:4])
            hour  = int(x[3][0:2])
            mint  = int(x[3][2:4])
            sec   = int(x[3][4:6])
        else:
            x = fileb.split('.')
            year  = int(x[0][3:5])
            year = year + 2000
            month = int(x[0][5:7])
            day   = int(x[0][7:9])
            hour  = int(x[0][9:11])
            mint  = int(x[0][11:13])
            sec   = int(x[0][13:15])

        DT = datetime.datetime(year, month, day, hour, mint, sec)
        if (DT >= DT_beg) & (DT < DT_end):
            files.append(file)
    return files

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

def calc_field_diff(radar, field1, field2):

    """
    Compute the difference between two fields.
    Written by: Charanjit S. Pabla, NASA/WFF/SSAI

    Parameters:
    -----------
    radar: pyart radar object
    field1: radar moment (str)
    field2: radar moment (str)

    Return:
    -------
    radar: pyart radar object with difference field included
    """
    
    #make sure fields are included in the radar object
    if field1 and field2 in radar.fields.keys():
    
        #copy fields
        f1 = radar.fields[field1]['data'].copy()
        f2 = radar.fields[field2]['data'].copy()
    
        #compute difference
        diff = f2 - f1
    
        #put back into radar objective
        radar.add_field_like(field, field+'_diff')
    
        return radar
    else:
        print(radar.fields.keys())
        raise Exception("{} {} fields are not in radar object".format(field1, field2))    

# ***************************************************************************************

def dbz_to_zlin(dz):
    """dz = Reflectivity (dBZ), returns Z (mm^6 m^-3)"""
    return 10.0**(dz / 10.0)

# ***************************************************************************************

def zlin_to_dbz(Z):
    """Z (mm^6 m^-3), returns dbz = Reflectivity (dBZ) """
    return 10.0 * np.log10(Z)

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

def add_csu_fhc(radar, dz, dr, rh, kd, radar_band, radar_T):
    
    scores = csu_fhc.csu_fhc_summer(dz=dz, zdr=dr, rho=rh, kdp=kd, use_temp=True, 
                                band=radar_band,
                                T=radar_T)

    fh = np.argmax(scores, axis=0) + 1
    radar = add_field_to_radar_object(fh, radar, field_name = 'FH',
                             units='Unitless',
                             long_name='Hydrometeor ID', 
                             standard_name='Hydrometeor ID',
                             dz_field='CZ')
    return radar, fh

# ***************************************************************************************

def add_csu_liquid_ice_mass(radar, dz, dr, radar_z, radar_T):
    mw, mi = csu_liquid_ice_mass.calc_liquid_ice_mass(dz, dr, radar_z, T=radar_T)
    radar = add_field_to_radar_object(mw, radar, field_name='MW', units='g m-3',
                                 long_name='Liquid Water Mass',
                                 standard_name='Liquid Water Mass',
                                 dz_field='CZ')

    radar = add_field_to_radar_object(mi, radar, field_name='MI', units='g m-3',
                                 long_name='Ice Water Mass',
                                 standard_name='Ice Water Mass',
                                 dz_field='CZ')
    return radar

# ***************************************************************************************

def add_csu_blended_rain(radar, dz, dr, kd, fh):
    rain, method = csu_blended_rain.csu_hidro_rain(dz=dz, zdr=dr, kdp=kd, fhc=fh)

    radar = add_field_to_radar_object(rain, radar, field_name='RC', units='mm/h',
                                 long_name='HIDRO Rainfall Rate', 
                                 standard_name='Rainfall Rate',
                                 dz_field='CZ')

    radar = add_field_to_radar_object(method, radar, field_name='MRC', units='',
                                 long_name='HIDRO Rainfall Method', 
                                 standard_name='Rainfall Method',
                                 dz_field='CZ')
    return radar


# ***************************************************************************************

def add_calc_dsd_sband_tokay_2020(radar, dz, dr, location):
    dm, nw = calc_dsd_sband_tokay_2020(dz, dr, loc=location, d0_n2=False)

    radar = add_field_to_radar_object(dm, radar, field_name='DM', units='mm',
                              long_name='Mass-weighted mean diameter',
                              standard_name='Mass-weighted mean diameter',
                              dz_field='CZ')
    radar = add_field_to_radar_object(nw, radar, field_name='NW', units='[Log Nw, m^-3 mm^-1]',
                              long_name='Normalized intercept parameter',
                              standard_name='Normalized intercept parameter',
                              dz_field='CZ') 
    return radar

# ***************************************************************************************

def remove_undesirable_fields(radar, DP_product_dict):
    print("Removing unwanted fields...")
    cf_fields = DP_product_dict['output_fields']
    drop_fields = [i for i in radar.fields.keys() if i not in cf_fields]
    for field in drop_fields:
        radar.fields.pop(field)
    print("CF FIELDS -->  ")
    print(radar.fields.keys())
    print()
  
    return radar
