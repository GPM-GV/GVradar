import pathlib
import numpy as np
from copy import deepcopy
import pyart
import os
import subprocess
import shlex
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import matplotlib.colors as colors
import copy
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import matplotlib.image as image
import time
from gvradar import common as cm
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain,
                            csu_dsd, csu_kdp, csu_misc, fundamentals)

# ***************************************************************************************
# Temporary script to ingest D3R data.
# ***************************************************************************************
def read_d3r(file):

    cfy = pathlib.Path(file).suffix
    if cfy == '.gz': 
        file = cm.unzip_file(file)
        file_unzip = file
        radar = pyart.aux_io.read_d3r_gcpex_nc(file, file_field_names=True)    
    else:
        radar = pyart.aux_io.read_d3r_gcpex_nc(file, file_field_names=True)   

    #site = 'D3R'
    return radar 

# ***************************************************************************************
def run_d3r(self):

    # Rename fields we want to keep with GPM, 2-letter IDs (e.g. CZ, DR, KD)
    print('', "Renaming radar fields...", sep='\n')

    old_fields = ['ReflectivityHV', 'Velocity', 'SpectralWidth', 
                      'DifferentialReflectivity', 'DifferentialPhase', 
                      'CopolarCorrelation']
    new_fields = ['DZ', 'VR', 'SW', 'DR', 'PH', 'RH']

    # Change names of old fields to new fields using pop
    nl = len(old_fields)
    for i in range(0,nl):
        old_field = old_fields[i]
        new_field = new_fields[i]
        self.radar.fields[new_field] = self.radar.fields.pop(old_field)
        i += 1 

    zz = deepcopy(self.radar.fields['DZ'])
    cz_dict = {'data': zz['data'], 'units': '', 'long_name': 'CZ',
               '_FillValue': -32767.0, 'standard_name': 'CZ'}
    self.radar.add_field('CZ', cz_dict, replace_existing=True)

    print(self.radar.fields.keys(), '', sep='\n')
    #exit('D3R radar support coming soon.')

# ***************************************************************************************

def plot_d3r_rhi(self):

    # Plot D3R radar 
    print('Plot D3R RHI')

    #
    # *** Get radar elevation, date, time
    # *** Rename fields to a standard for plotting purposes
    #
    sweep = 0
    png = False
    site, mydate, mytime, azi, year, month, day, hh, mm, ss, string_csweep = get_radar_info(self.radar, sweep)
    fields = ['DZ']
    #
    # *** Set bounding limits for plot
    #
    xlim=[0,40]
    ylim=[0,15]
    num_fields = len(fields)
    nrows = (num_fields + 1) // 2
    if num_fields < 2:
        width=12
        height=3.5
        ncols=1
    else:
        width=24
        height=3.5 * nrows
        ncols=2

    r_c = []
    for y in range(ncols):
        for x in range(nrows):
            r_c.append((x,y))

    #
    # *** Plotting Begins
    #
    set_plot_size_parms_rhi(num_fields)
    display = pyart.graph.RadarMapDisplay(self.radar)
    
    if num_fields < 2:
        fig = plt.figure(figsize=[width, height], constrained_layout=False)
    else:
        fig = plt.figure(figsize=[width, height], constrained_layout=True)
   
    spec = plt.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
        
    for index, field in enumerate(fields):

        units,vmin,vmax,cmap,title = get_field_info(self.radar, field)
        
        if num_fields < 2:
            title = '{} {} {} {} UTC RHI Azi: {:2.1f}'.format(site,field,mydate,mytime,azi)
        else:
            mytitle = '{} {} {} UTC RHI {:2.1f} Azi'.format(site,mydate,mytime,azi)
     
        ax = fig.add_subplot(spec[r_c[index]])
 
        display.plot_rhi(field, sweep, vmin=vmin, vmax=vmax, cmap=cmap,
                         title=title,
                         colorbar_label=units)
        display.set_limits(xlim, ylim, ax=ax)
        display.plot_grid_lines()
        
        #if add_logos:  annotate_plot_rhi(ax,fig,num_fields,nrows)
        
        if field == 'FH' or field == 'FH2': display.cbs[index] = adjust_fhc_colorbar_for_pyart(display.cbs[index])
        if field == 'MRC' or field == 'MRC2': display.cbs[index] = adjust_meth_colorbar_for_pyart(display.cbs[index])

    if num_fields >= 2:
        plt.suptitle(mytitle,fontsize = 36, weight ='bold')
    
    #
    # *** save plot
    #    
    if png:
        if num_fields == 1:
            png_file = '{}_{}_{}_{}_{}_{:2.1f}AZ_RHI.png'.format(site,year
                                                    ,month+day,hh+mm+ss,field,azi)
            outdir_daily = outdir + '/' + year + '/' + month + day + '/RHI/' + field + '/'
            os.makedirs(outdir_daily, exist_ok=True)
            fig.savefig(outdir_daily + '/' + png_file, dpi=1200, bbox_inches='tight')
            print('  --> ' + outdir_daily + '/' + png_file, '', sep='\n')
        elif num_fields >1:
            png_file = '{}_{}_{}_{}_{}panel_{:2.1f}AZ_RHI.png'.format(site,year
                                                    ,month+day,hh+mm+ss,num_fields,azi)
            outdir_multi = outdir + '/' + year + '/' + month + day + '/multi/'
            os.makedirs(outdir_multi, exist_ok=True)
            fig.savefig(outdir_multi + '/' + png_file, dpi=150, bbox_inches='tight')
            print('  --> ' + outdir_multi + '/' + png_file, '', sep='\n')
        if outdir == '':
            outdir = os.getcwd()
    else:
        plt.show()
    
    exit('D3R radar support coming soon.')
# ***************************************************************************************

def plot_d3r_ppi(self):

    # Plot D3R radar 
    print('Plot D3R PPI')

    #
    # *** Get radar elevation, date, time
    # *** Rename fields to a standard for plotting purposes
    #
    sweep = 0
    png = False
    #site, mydate, mytime, azi, year, month, day, hh, mm, ss, string_csweep = get_radar_info(self.radar, sweep)
    fields=self.fields_to_plot

    site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep = get_radar_info(self.radar, sweep) 

    #
    # *** Calculate bounding limits for map
    #
    max_range = 40
    radar_lat = self.radar.latitude['data'][0]
    radar_lon = self.radar.longitude['data'][0]
    dtor = math.pi/180.0
    maxrange_meters = max_range * 1000.
    meters_to_lat = 1. / 111177.
    meters_to_lon =  1. / (111177. * math.cos(radar_lat * dtor))

    min_lat = radar_lat - maxrange_meters * meters_to_lat
    max_lat = radar_lat + maxrange_meters * meters_to_lat
    min_lon = radar_lon - maxrange_meters * meters_to_lon
    max_lon = radar_lon + maxrange_meters * meters_to_lon
    min_lon_rn=round(min_lon,2)
    max_lon_rn=round(max_lon,2)
    min_lat_rn=round(min_lat,2)
    max_lat_rn=round(max_lat,2)

    lon_grid = np.arange(min_lon_rn - 1.00 , max_lon_rn + 1.00, 1.0)
    lat_grid = np.arange(min_lat_rn - 1.00 , max_lat_rn + 1.00, 1.0)
    
    projection = ccrs.LambertConformal(radar_lon, radar_lat)
    display = pyart.graph.RadarMapDisplay(self.radar)

    num_fields = len(fields)
    nrows = round((num_fields)//4)
    if nrows < 1 : nrows = 1
    if num_fields <= 4:
        width=num_fields * 6
        height = float((nrows)*4.5)
        ncols=num_fields
    elif num_fields > 4:
        width=24
        height = float((nrows)*4.5)
        ncols=4

    r_c = []
    for x in range(nrows):
        for y in range(ncols):
            r_c.append((x,y))
    #
    # *** Plotting Begins
    #
    set_plot_size_parms_ppi(num_fields)

    if num_fields < 2:
        fig = plt.figure(figsize=[width, height], constrained_layout=False)
        spec = plt.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    else:
        fig = plt.figure(figsize=[width, height], constrained_layout=False, dpi=240)
        spec = plt.GridSpec(ncols=ncols, nrows=nrows, figure=fig, left=0.0, right=1.0, top=1.0, bottom=0, wspace=0.000000009, hspace=0.15)
    
    for index, field in enumerate(fields):
        
        units,vmin,vmax,cmap,title = get_field_info(self.radar, field)
        
        if num_fields < 2:
            title = '{} {} {} {} UTC PPI Elev: {:2.1f} deg'.format(site,field,mydate,mytime,elv)
        else:
            mytitle = '{} {} {} UTC PPI {:2.1f} deg'.format(site,mydate,mytime,elv)
        
        ax = fig.add_subplot(spec[r_c[index]], projection=projection)
        display.plot_ppi_map(field, sweep, vmin=vmin, vmax=vmax,
                     resolution='10m',
                     title = title,
                     projection=projection, ax=ax,
                     cmap=cmap,colorbar_label=units,
                     min_lon=min_lon, max_lon=max_lon,
                     min_lat=min_lat, max_lat=max_lat,
                     lon_lines=lon_grid,lat_lines=lat_grid,
                     lat_0=radar_lat,
                     lon_0=radar_lon,
                     embellish = False)
        add_logos = False
        add_rings_radials(display, radar_lat, radar_lon, max_range, ax, add_logos, fig, num_fields, nrows, ncols)
        '''
        if index == num_fields-1:
            add_logo_ppi(display, radar_lat, radar_lon, max_range, ax, add_logos, fig, num_fields, nrows, ncols)
            if num_fields >= 2:
                plt.suptitle(mytitle, fontsize = 8*ncols, weight ='bold', y=(1.0+(ncols*0.05)))
        '''        
        if field == 'FH' or field == 'FH2': display.cbs[index] = adjust_fhc_colorbar_for_pyart(display.cbs[index])
        if field == 'MRC' or field == 'MRC2': display.cbs[index] = adjust_meth_colorbar_for_pyart(display.cbs[index])
        if field == 'FS' or field == 'FH2': display.cbs[index] = adjust_fhc_colorbar_for_pyart(display.cbs[index])
        if field == 'FW' or field == 'FH2': display.cbs[index] = adjust_fhw_colorbar_for_pyart(display.cbs[index])
    
    #
    # *** save plot
    #
    if png:
        if num_fields == 1:
            png_file='{}_{}_{}_{}_{}_sw{}_PPI.png'.format(site,year,month+day,hh+mm+ss,field,string_csweep)
            outdir_daily = outdir + '/' + year + '/' + month + day + '/PPI/' + field + '/'
            os.makedirs(outdir_daily, exist_ok=True)
            fig.savefig(outdir_daily + '/' + png_file, dpi=240, bbox_inches='tight')
            print('  --> ' + outdir_daily + '/' + png_file, '', sep='\n')
        elif num_fields >1:
            png_file='{}_{}_{}_{}_{}panel_sw{}_PPI.png'.format(site,year,month+day,hh+mm+ss,num_fields,string_csweep)
            outdir_multi = outdir + '/' + year + '/' + month + day + '/multi/' 
            os.makedirs(outdir_multi, exist_ok=True)
            fig.savefig(outdir_multi + '/' + png_file, dpi=240, bbox_inches='tight')
            print('  --> ' + outdir_multi + '/' + png_file, '', sep='\n')
        if outdir == '':
            outdir = os.getcwd()

    else:
        plt.show()

    exit('D3R radar support coming soon.')

# ***************************************************************************************

def get_radar_info(radar, sweep):
    #
    # *** get radar elevation, date, time
    #
    radar_DT = pyart.util.datetime_from_radar(radar)
    elv=radar.fixed_angle['data'][sweep]
    string_csweep = str(sweep).zfill(2)
    month = str(radar_DT.month).zfill(2)
    day = str(radar_DT.day).zfill(2)
    year = str(radar_DT.year).zfill(4)
    hh = str(radar_DT.hour).zfill(2)
    mm = str(radar_DT.minute).zfill(2)
    ss = str(radar_DT.second).zfill(2)
    mydate = month + '/' + day + '/' + year
    mytime = hh + ':' + mm + ':' + ss
    site = 'D3R'

    return site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep

# ***************************************************************************************

def set_plot_size_parms_rhi(num_fields):

    if num_fields < 2:
        SMALL_SIZE = 6
        MEDIUM_SIZE = 8
        BIGGER_SIZE = 10
        
        plt.rc('font', size=MEDIUM_SIZE, weight='bold') # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    else:
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 20

        plt.rc('font', size=SMALL_SIZE, weight='bold') # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# ****************************************************************************************

def get_field_info(radar, field):

    # Set up colorbars
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
                  'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    hid_colors_summer =  ['White','LightBlue','MediumBlue','Darkorange','LightPink',
                   'Cyan','DarkGray', 'Lime','Yellow','Red','Fuchsia']
    hid_colors_winter = ['White','Orange', 'Purple', 'Fuchsia', 'Pink', 'Cyan',
                         'LightBlue', 'Blue']
    cmaphidw = colors.ListedColormap(hid_colors_winter)
    cmaphid = colors.ListedColormap(hid_colors_summer)
    #cmaphid = colors.ListedColormap(hid_colors)
    cmapmeth = colors.ListedColormap(hid_colors[0:6])
    cmapmeth_trop = colors.ListedColormap(hid_colors[0:7])

    #get cmap, units, vmin, vmax
    if field == 'CZ':
        units='Zh [dBZ]'
        vmin=0
        vmax=65
        title = 'Corrected Reflectivity [dBZ]'
        cmap='jet'
    elif field == 'DZ':
        units='Zh [dBZ]'
        vmin=0
        vmax=65
        title = 'RAW Reflectivity [dBZ]'
        cmap='jet'
    elif field == 'RawPower_HV' or field == 'ReflectivityV':
        units='Zh [dBZ]'
        vmin=0
        vmax=65
        title = 'RAW Reflectivity [dBZ]'
        cmap='jet'    
    elif field == 'DR':
        units='Zdr [dB]'
        vmin=-1
        vmax=3
        title = 'Differential Reflectivity [dB]'
        cmap='pyart_RefDiff'
    elif field == 'VR':
        units='Velocity [m/s]'
        vmin=-20
        vmax=20
        title = 'Radial Velocity [m/s]'
        #cmap='pyart_NWSVel'
        cmap='pyart_balance'
    elif field == 'corrected_velocity':
        units='Velocity [m/s]'
        vmin=-20
        vmax=20
        title = 'Dealiased Radial Velocity [m/s]'
        #cmap='pyart_NWSVel'   
        cmap='pyart_balance'
    elif field == 'KD':
        units='Kdp [deg/km]'
        vmin=-2
        vmax=3
        title = 'Specific Differential Phase [deg/km]'
        cmap='pyart_NWSRef'
    elif field == 'KDPB':
        units='Kdp [deg/km]'
        vmin=-1
        vmax=3
        title = 'Specific Differential Phase [deg/km] (Bringi)'
        cmap='pyart_NWSRef'
    elif field == 'PH':
        units='PhiDP [deg]'
        vmin=0
        vmax=360
        title ='Differential Phase [deg]' 
        #cmap='pyart_LangRainbow12_r'
        cmap='pyart_Carbone42'
    elif field == 'PHM':
        units='PhiDP [deg]'
        vmin=0
        vmax=360
        title ='Differential Phase [deg] Marks' 
        cmap='pyart_LangRainbow12_r'
    elif field == 'RH':
        units='Correlation'
        vmin=0
        vmax=1
        title = 'Correlation Coefficient'
        cmap='jet'
        #cmap='pyart_Wild25'
    elif field == 'SD':
        units='Std(PhiDP)'
        vmin=0
        vmax=70
        title = 'Standard Deviation of PhiDP'
        cmap='pyart_NWSRef'
    elif field == 'SQ':
        units='SQI'
        vmin=0
        vmax=1
        title = 'Signal Quality Index'
        cmap='pyart_Bu10_r'
    elif field == 'FH':
        units='HID'
        vmin=0
        vmax=11
        title = 'Hydrometeor Identification'
        cmap=cmaphid
    elif field == 'FS':
        units='HID'
        vmin=0
        vmax=11
        title = 'Summer Hydrometeor Identification'
        cmap=cmaphid
    elif field == 'FW':
        units='HID'
        vmin=0
        vmax=8
        title = 'Winter Hydrometeor Identification'
        cmap=cmaphidw
    elif field == 'MW':
        units='Water Mass [g/m^3]'
        vmin=0
        vmax=10
        title = 'Water Mass [g/m^3]'
        cmap='pyart_BlueBrown10'
    elif field == 'MI':
        units='Ice Mass [g/m^3]'
        vmin=-1
        vmax=3
        title ='Ice Mass [g/m^3]'
        cmap='pyart_BlueBrown10'
    elif field == 'RC':
        units='HIDRO Rain Rate [mm/hr]'
        vmin=0
        vmax=80
        title ='HIDRO Rain Rate [mm/hr]'
        cmap='pyart_NWSRef'
    elif field == 'MRC':
        units='HIDRO Method'
        vmin=0
        vmax=5
        title ='HIDRO Method'
        cmap=cmapmeth
    elif field == 'DM':
        units='DM [mm]'
        vmin=0
        vmax=4
        title ='DM [mm]'
        cmap='pyart_BlueBrown10'
    elif field == 'NW':
        units='Log[Nw, m^-3 mm^-1]'
        vmin=0
        vmax=6
        title ='Log[Nw, m^-3 mm^-1]'
        cmap='pyart_BlueBrown10'

    return units,vmin,vmax,cmap,title

# ****************************************************************************************
def set_plot_size_parms_ppi(num_fields):

    if num_fields < 2:
        SMALL_SIZE = 6
        MEDIUM_SIZE = 8
        BIGGER_SIZE = 10

        plt.rc('font', size=MEDIUM_SIZE, weight='bold') # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    else:
        SMALL_SIZE = 8
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14

        plt.rc('font', size=SMALL_SIZE, weight='bold') # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
# ****************************************************************************************
def add_rings_radials(display, radar_lat, radar_lon, max_range, ax, add_logos, fig, num_fields, nrows, ncols):

    # NASA WFF instrument pad locations
    Pad_lon = -75.471
    Pad_lat =  37.934
    PCMK_lon = -75.515
    PCMK_lat =  38.078
    dtor = math.pi/180.0
    max_range = 40

    maxrange_meters = max_range * 1000.
    meters_to_lat = 1. / 111177.
    meters_to_lon =  1. / (111177. * math.cos(radar_lat * dtor))
    display.plot_cross_hair(10,npts=100)
    for rng in range(20,max_range+20,20):
        display.plot_range_ring(rng, line_style='k--', lw=0.5)

    for azi in range(0,360,30):
        azimuth = 90. - azi
        dazimuth = azimuth * dtor
        lon_maxrange = radar_lon + math.cos(dazimuth) * meters_to_lon * maxrange_meters
        lat_maxrange = radar_lat + math.sin(dazimuth) * meters_to_lat * maxrange_meters
        display.plot_line_geo([radar_lon, lon_maxrange], [radar_lat, lat_maxrange],
                              line_style='k--',lw=0.5)
    display.plot_cross_hair(10,npts=100)
    display.plot_point(Pad_lon, Pad_lat, symbol = 'kv', markersize=5)
    display.plot_point(PCMK_lon, PCMK_lat, symbol = 'kv', markersize=5)

    # Add state and countines to map
    states_provinces = cfeature.NaturalEarthFeature(
                                category='cultural',
                                name='admin_1_states_provinces_lines',
                                scale='10m',
                                facecolor='none')
    ax.add_feature(states_provinces, edgecolor='black', lw=0.5)
    county_dir = os.path.dirname(__file__)
    reader = shpreader.Reader(county_dir + '/countyl010g.shp')
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='black', lw=0.25)

    # Add cartopy grid lines
    grid_lines = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', x_inline=False)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    grid_lines.xformatter = LONGITUDE_FORMATTER
    grid_lines.yformatter = LATITUDE_FORMATTER
    grid_lines.xlabel_style = {'size': 6, 'color': 'black', 'rotation': 0, 'weight': 'bold', 'ha': 'center'}
    grid_lines.ylabel_style = {'size': 6, 'color': 'black', 'rotation': 90, 'weight': 'bold', 'va': 'bottom', 'ha': 'center'}

    return

# ****************************************************************************************
