import numpy as np
import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import pyart
import gpm_dpqc as qc
import copy
import os

# ****************************************************************************************

def plot_fields(radar, thresh_dict):

    """
    Calls plotting programs based on user defined dictionary parameters.

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """

    scantype = radar.scan_type
    if thresh_dict['sweeps_to_plot'] == 'all':
        sweepn = radar.sweep_number['data'][:][:]
    else:
        sweepn = thresh_dict['sweeps_to_plot']
    plot_single = thresh_dict['plot_single']
    fields_to_plot = thresh_dict['fields_to_plot']

    if scantype == 'rhi':
        print('Plotting RHI images...')
        max_range = thresh_dict['max_range']
        max_height = thresh_dict['max_height']
        if plot_single == True:
            for ifld in range(len(fields_to_plot)):
                print(fields_to_plot[ifld])
                field = fields_to_plot[ifld]
                plt_dir = thresh_dict['plot_dir'] + field + '/'
                os.makedirs(plt_dir, exist_ok=True)
                for isweeps in range(len(sweepn)):
                    sweep = sweepn[isweeps]
                    plot_fields_RHI(radar, sweep=sweep, fields=[field] , ymax=max_height, xmax=max_range, png=True, outdir=plt_dir)
        if plot_single == False:
            for isweeps in range(len(sweepn)):
                sweep = sweepn[isweeps]
                plt_dir = thresh_dict['plot_dir']
                os.makedirs(plt_dir, exist_ok=True)
                plot_fields_RHI(radar, sweep=sweep, fields=fields_to_plot , ymax=max_height, xmax=max_range, png=True, outdir=plt_dir)

    if scantype == 'ppi':
        print('Plotting PPI images...')
        max_range = thresh_dict['max_range']
        if plot_single == True:
            for ifld in range(len(fields_to_plot)):
                print(fields_to_plot[ifld])
                field = fields_to_plot[ifld]
                plt_dir = thresh_dict['plot_dir'] + field + '/'
                os.makedirs(plt_dir, exist_ok=True)
                for isweeps in range(len(sweepn)):
                    sweep = sweepn[isweeps]
                    plot_fields_PPI(radar, sweep=sweep, fields=[field] , max_range=max_range, png=True, outdir=plt_dir)
        if plot_single == False:
            for isweeps in range(len(sweepn)):
                sweep = sweepn[isweeps]
                plt_dir = thresh_dict['plot_dir']
                os.makedirs(plt_dir, exist_ok=True)
                plot_fields_PPI(radar, sweep=sweep, fields=fields_to_plot , max_range=max_range, png=True, outdir=plt_dir)
    
    #Clear python memory
    del radar_copy

# ****************************************************************************************

def plot_fields_PPI(radar, sweep=0, fields=['CZ'], max_range=150, png=False, outdir=''):

    #
    # *** Get radar elevation, date, time
    # *** Rename fields to a standard for plotting purposes
    #

    radar_copy, site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep = get_radar_info(radar, sweep) 

    #
    # *** Calculate bounding limits for map
    #
    radar_lat = radar_copy.latitude['data'][0]
    radar_lon = radar_copy.longitude['data'][0]
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
    display = pyart.graph.RadarMapDisplay(radar_copy)

    num_fields = len(fields)
    nrows = (num_fields + 1) // 2
    #ncols = (num_fields + 1) % 2 + 1
    if num_fields < 2:
        width=6
        ncols=1
    else:
        width=12
        ncols=2

    #
    # *** Plotting Begins
    #
    set_plot_size_parms(num_fields)
    fig = plt.figure(figsize=(width, float(nrows)*4.5))
    for index, field in enumerate(fields):
        
        units,vmin,vmax,cmap,title = get_field_info(field)
        
        if num_fields < 2:
            title = '{} {} {} {} UTC PPI Elev: {:2.1f} deg'.format(site,field,mydate,mytime,elv)
        else:
            mytitle = '{} {} {} UTC PPI {:2.1f} deg'.format(site,mydate,mytime,elv)
 
        ax = fig.add_subplot(nrows, ncols, index+1, projection=projection)
        display.plot_ppi_map(field, sweep, vmin=vmin, vmax=vmax,
                     resolution='10m',
                     title = title,
                     projection=projection, ax=ax,
                     cmap=cmap,colorbar_label=units,
                     min_lon=min_lon, max_lon=max_lon,
                     min_lat=min_lat, max_lat=max_lat,
                     lon_lines=lon_grid,lat_lines=lat_grid,
                     lat_0=radar_lat,
                     lon_0=radar_lon)
        annotate_plot(display, radar_lat, radar_lon, max_range)
        if field == 'FH' or field == 'FH2': display.cbs[index] = adjust_fhc_colorbar_for_pyart(display.cbs[index])
        if field == 'MRC' or field == 'MRC2': display.cbs[index] = adjust_meth_colorbar_for_pyart(display.cbs[index])

    if num_fields >= 2:
        plt.subplots_adjust(hspace=0.15, wspace=0.05)
        plt.suptitle(mytitle, fontsize = 14, va='top', y=0.97)

    #
    # *** save plot
    #
    if png:
        if num_fields == 1:
            png_file='{}_{}_{}_{}_{}_sw{}_PPI.png'.format(site,year,month+day,hh+mm+ss,field,string_csweep)
        elif num_fields >1:
            png_file='{}_{}_{}_{}_{}panel_sw{}_PPI.png'.format(site,year,month+day,hh+mm+ss,num_fields,string_csweep)
        if outdir == '':
            outdir = os.getcwd()
        fig.savefig(outdir+'/'+png_file, dpi=150, bbox_inches='tight')
        print('  --> ' + outdir+'/'+png_file)
        print()
    else:
        plt.show()

# ****************************************************************************************

def plot_fields_RHI(radar, sweep=0, fields=['CZ'], ymax=10, xmax=150, png=False, outdir=''):

    #
    # *** Get radar elevation, date, time
    # *** Rename fields to a standard for plotting purposes
    #
    
    radar_copy, site, mydate, mytime, azi, year, month, day, hh, mm, ss, string_csweep = get_radar_info(radar, sweep)

    #
    # *** Set bounding limits for plot
    #
    xlim=[0,xmax]
    ylim=[0,ymax]
    num_fields = len(fields)
    nrows = (num_fields + 1) // 2
    if num_fields < 2:
        width=8
        ncols=1
    else:
        width=14
        ncols=2

    #
    # *** Plotting Begins
    #
    set_plot_size_parms(num_fields)
    display = pyart.graph.RadarMapDisplay(radar_copy)
    fig = plt.figure(figsize=(width, float(nrows)*5))
    for index, field in enumerate(fields):
        
        units,vmin,vmax,cmap,title = get_field_info(field)
        
        if num_fields < 2:
            title = '{} {} {} {} UTC RHI Azi: {:2.1f}'.format(site,field,mydate,mytime,azi)
        else:
            mytitle = '{} {} {} UTC RHI {:2.1f} Azi'.format(site,mydate,mytime,azi)
     
        ax = fig.add_subplot(nrows, ncols, index+1)
        display.plot_rhi(field, sweep, vmin=vmin, vmax=vmax, cmap=cmap,
                         title=title,
                         colorbar_label=units)
        display.set_limits(xlim, ylim, ax=ax)
        display.plot_grid_lines()

        if field == 'FH' or field == 'FH2': display.cbs[index] = adjust_fhc_colorbar_for_pyart(display.cbs[index])
        if field == 'MRC' or field == 'MRC2': display.cbs[index] = adjust_meth_colorbar_for_pyart(display.cbs[index])

    if num_fields >= 2:
        plt.subplots_adjust(hspace=0.25, wspace=0.15)
        plt.suptitle(mytitle, fontsize = 18, va='top', y=0.93)
    
    #
    # *** save plot
    #
    if png:
        if num_fields == 1:
            png_file = '{}_{}_{}_{}_{}_{:2.1f}AZ_RHI.png'.format(site,year
                                                    ,month+day,hh+mm+ss,field,azi)
        elif num_fields >1:
            png_file = '{}_{}_{}_{}_{}panel_{:2.1f}AZ_RHI.png'.format(site,year
                                                    ,month+day,hh+mm+ss,num_fields,azi)
        if outdir == '':
            outdir = os.getcwd()
        fig.savefig(outdir+'/'+png_file, dpi=150,bbox_inches='tight')
        print('  --> ' + outdir+'/'+png_file)
        print()
    else:
        plt.show()

# ****************************************************************************************

def set_plot_size_parms(num_fields):

    if num_fields < 2:
        SMALL_SIZE = 6
        MEDIUM_SIZE = 8
        BIGGER_SIZE = 10
    else:
        SMALL_SIZE = 8
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14 

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

# ****************************************************************************************

def get_field_info(field):

    # Set up colorbars
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
                  'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)
    cmapmeth = colors.ListedColormap(hid_colors[0:6])
    cmapmeth_trop = colors.ListedColormap(hid_colors[0:7])

    #get cmap, units, vmin, vmax
    if field == 'DZ' or field == 'CZ':
        units='Zh [dBZ]'
        vmin=0
        vmax=65
        title = 'Corrected Reflectivity [dBZ]'
        cmap='pyart_NWSRef'
    elif field == 'ZZ':
        units='Zh [dBZ]'
        vmin=0
        vmax=65
        title = 'RAW Reflectivity [dBZ]'
        cmap='pyart_NWSRef'
    elif field == 'DR':
        units='Zdr [dB]'
        vmin=-1
        vmax=4
        title = 'Differential Reflectivity [dB]'
        cmap='pyart_RefDiff'
    elif field == 'VR':
        units='Velocity [m/s]'
        vmin=-20
        vmax=20
        title = 'Radial Velocity [m/s]'
        cmap='pyart_NWSVel'
    elif field == 'KD':
        units='Kdp [deg/km]'
        vmin=-1
        vmax=3
        title = 'Specific Differential Phase [deg/km]'
        cmap='pyart_NWSRef'
    elif field == 'PH':
        units='PhiDP [deg]'
        vmin=0
        vmax=360
        title ='Differential Phase [deg]' 
        cmap='pyart_LangRainbow12_r'
    elif field == 'RH':
        units='Correlation'
        vmin=0
        vmax=1
        title = 'Correlation Coefficient'
        cmap='pyart_Wild25'
    elif field == 'SD':
        units='Std(PhiDP)'
        vmin=0
        vmax=25
        title = 'Standard Deviation of PhiDP'
        cmap='pyart_NWSRef'
    elif field == 'SQ':
        units='SQI'
        vmin=0
        vmax=5
        title = 'Signal Quality Index'
        cmap='pyart_Bu10_r'
    elif field == 'FH':
        units='HID'
        vmin=0
        vmax=11
        title = 'Hydrometeor Identification'
        cmap=cmaphid
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
        vmax=100
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

def get_radar_info(radar, sweep):

    #
    # *** get radar elevation, date, time
    #

    radar_copy = copy.deepcopy(radar)
    radar_DT = pyart.util.datetime_from_radar(radar_copy)
    elv=radar_copy.fixed_angle['data'][sweep]
    string_csweep = str(sweep).zfill(2)
    month = str(radar_DT.month).zfill(2)
    day = str(radar_DT.day).zfill(2)
    year = str(radar_DT.year).zfill(4)
    hh = str(radar_DT.hour).zfill(2)
    mm = str(radar_DT.minute).zfill(2)
    ss = str(radar_DT.second).zfill(2)
    mydate = month + '/' + day + '/' + year
    mytime = hh + ':' + mm + ':' + ss
    if 'site_name' in radar.metadata.keys():
        site = radar.metadata['site_name'].upper()
    elif 'instrument_name' in radar.metadata.keys():
        if isinstance(radar.metadata['instrument_name'], bytes):
            site = radar.metadata['instrument_name'].decode().upper()
        else:
            site = radar.metadata['instrument_name'].upper()
    else:
        site=''

    #
    # *** rename fields to a standard for plotting purposes
    #
    if 'PHIDP2' in radar_copy.fields.keys():
        old_fields = ['DBZ2', 'VEL2', 'WIDTH2', 'ZDR2', 'KDP2', 'PHIDP2', 'RHOHV2', 'SQI2']
        new_fields = ['DZ',     'VR',    'SW',   'DR',   'KD',   'PH',     'RH',     'SQ']
    elif 'DBZ' in radar_copy.fields.keys():
        old_fields = ['DBZ', 'VEL', 'WIDTH', 'ZDR', 'KDP', 'PHIDP', 'SQI', 'RHOHV']
        new_fields = ['DZ',  'VR',   'SW',   'DR',  'KD',   'PH',    'SQ',    'RH']
    elif 'DBZ2' in radar_copy.fields.keys():
        old_fields = ['DBZ2', 'VEL2', 'WIDTH2', 'ZDR2', 'KDP2', 'PHM', 'RHOHV2', 'STDPHIB', 'SQI2']
        new_fields = ['DZ',     'VR',     'SW',   'DR',   'KD', 'PH',     'RH',     'SD',    'SQ']
    elif 'CZ' in radar_copy.fields.keys():
        #old_fields = ['CZ',   'VR',     'SW',   'DR',   'KD',  'PH',     'RH',      'SD',   'SQ']
        old_fields = []
        new_fields = []
    qc.rename_fields_in_radar(radar_copy, old_fields, new_fields)

    return radar_copy, site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep

# ****************************************************************************************

def annotate_plot(display, radar_lat, radar_lon, max_range):
    plt.yticks(rotation=90, va = 'center')

    # Annotate plot
    Pad_lon = -75.471
    Pad_lat =  37.934
    PCMK_lon = -75.515
    PCMK_lat =  38.078
    dtor = math.pi/180.0

    maxrange_meters = max_range * 1000.
    meters_to_lat = 1. / 111177.
    meters_to_lon =  1. / (111177. * math.cos(radar_lat * dtor))
    display.plot_cross_hair(10,npts=100)
    for rng in range(50,200,50):
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
    return

# ****************************************************************************************

def adjust_fhc_colorbar_for_pyart(cb):
    cb.set_ticks(np.arange(1.4, 10, 0.9))
    cb.ax.set_yticklabels(['Drizzle', 'Rain', 'Ice Crystals', 'Aggregates',
                           'Wet Snow', 'Vertical Ice', 'LD Graupel',
                           'HD Graupel', 'Hail', 'Big Drops'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

# ****************************************************************************************

def adjust_meth_colorbar_for_pyart(cb, tropical=False):
    if not tropical:
        cb.set_ticks(np.arange(1.25, 5, 0.833))
        cb.ax.set_yticklabels(['R(Kdp, Zdr)', 'R(Kdp)', 'R(Z, Zdr)', 'R(Z)', 'R(Zrain)'])
    else:
        cb.set_ticks(np.arange(1.3, 6, 0.85))
        cb.ax.set_yticklabels(['R(Kdp, Zdr)', 'R(Kdp)', 'R(Z, Zdr)', 'R(Z_all)', 'R(Z_c)', 'R(Z_s)'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

# ****************************************************************************************

def plot_field_diff(radar, field1, field2, sweep_num):

    pyradar_DT = pyart.util.datetime_from_radar(radar)
    elv=radar.fixed_angle['data'][sweep_num]
    mydate = str(radar_DT.month).zfill(2) + '/' + str(radar_DT.day).zfill(2) + '/' + \
         str(radar_DT.year).zfill(4)

    mytime = str(radar_DT.hour).zfill(2) + ':' + str(radar_DT.minute).zfill(2) + ':' + \
         str(radar_DT.second).zfill(2)
    mytitle = 'NPOL ' + mydate + ' ' + mytime + ' UTC {:2.1f} deg'.format(elv)

    fields=[field1, field2, field1+'_diff']

    num_fields = len(fields)
    nrows = (num_fields + 1) // 2
    ncols = (num_fields + 1) % 2 + 1
    azimuth=np.arange(0,360)
    fig = plt.figure(figsize=(14, float(nrows)*5))
    for index, field in enumerate(fields):

        #compute mean
        pymean = cmp_azimuth_mean(pyradar, sweep_num, field)
        idlmean = cmp_azimuth_mean(idlradar, sweep_num, field)
        #print(pymean.shape)
        #print(idlmean.shape)
        #print(x.shape)

        #plot
        ax = fig.add_subplot(nrows, 2, index+1)
        ax.plot(azimuth, pymean, color='red', label='pyQC')
        ax.plot(azimuth, idlmean, color='blue', label='idlQC')
        ax.set_xlabel('Azimuth (deg)')
        ax.set_ylabel(field + ' ('+units[index]+')')
        ax.grid(True)
        if index == 0: ax.legend(loc=0)
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle(mytitle,fontsize=20, va='top', y=0.975)
    plt.show()

# ****************************************************************************************


