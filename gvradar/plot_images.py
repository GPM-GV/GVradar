import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import matplotlib.colors as colors
import pyart
import copy
import os
import cartopy
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import matplotlib.image as image
import time

# ****************************************************************************************

def plot_fields(self):

    """
    Calls plotting programs based on user defined dictionary parameters.

    Written by: Jason L. Pippitt, NASA/GSFC/SSAI
    """
    start = time.time()
    plot_fast = True
    if self.sweeps_to_plot == 'all':
        sweepn = self.radar.sweep_number['data'][:][:]
        swn = []
        for sn in range(len(sweepn)):
            swn.append(sn)
        sweepn = swn
    else:
        sweepn = self.sweeps_to_plot

    if self.scan_type == 'RHI':
        print('Plotting RHI images...')
        if self.plot_multi == True:
            for isweeps in range(len(sweepn)):
                sweep = sweepn[isweeps]
                os.makedirs(self.plot_dir, exist_ok=True)
                plot_fields_RHI(self.radar, sweep=sweep, fields=self.fields_to_plot , ymax=self.max_height,
                                xmax=self.max_range, png=True, outdir=self.plot_dir, add_logos = self.add_logos)
        if self.plot_single == True:
            for ifld in range(len(self.fields_to_plot)):
                print(self.fields_to_plot[ifld])
                field = self.fields_to_plot[ifld]
                #plot_dir = self.plot_dir + '/' + field
                os.makedirs(self.plot_dir, exist_ok=True)
                for isweeps in range(len(sweepn)):
                    sweep = sweepn[isweeps]
                    plot_fields_RHI(self.radar, sweep=sweep, fields=[field] , ymax=self.max_height, 
	                            xmax=self.max_range, png=True, outdir=self.plot_dir, add_logos = self.add_logos)

    if self.scan_type == 'PPI':
        print('Plotting PPI images...')
        if self.plot_multi == True:
            for isweeps in range(len(sweepn)):
                sweep = sweepn[isweeps]
                os.makedirs(self.plot_dir, exist_ok=True)
                if plot_fast:
                    plot_fields_PPI_QC(self.radar, sweep=sweep, fields=self.fields_to_plot , max_range=self.max_range, png=True, outdir=self.plot_dir, add_logos = self.add_logos)
                else:
                    plot_fields_PPI(self.radar, sweep=sweep, fields=self.fields_to_plot , max_range=self.max_range, png=True, outdir=self.plot_dir, add_logos = self.add_logos)
        if self.plot_single == True:
            for ifld in range(len(self.fields_to_plot)):
                print(self.fields_to_plot[ifld])
                field = self.fields_to_plot[ifld]
                #plot_dir = self.plot_dir + '/' + field + '/'
                os.makedirs(self.plot_dir, exist_ok=True)
                for isweeps in range(len(sweepn)):
                    sweep = sweepn[isweeps]
                    if plot_fast:
                        plot_fields_PPI_QC(self.radar, sweep=sweep, fields=[field] , max_range=self.max_range, png=True, outdir=self.plot_dir, add_logos = self.add_logos)
                    else:
                        plot_fields_PPI(self.radar, sweep=sweep, fields=[field] , max_range=self.max_range, png=True, outdir=self.plot_dir, add_logos = self.add_logos)

    end = time.time()
    print('ploting time:  ', end - start)
# ****************************************************************************************

def plot_fields_PPI(radar, sweep=0, fields=['CZ'], max_range=150, png=False, outdir='', add_logos=True):

    #
    # *** Get radar elevation, date, time
    # *** Rename fields to a standard for plotting purposes
    #

    site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep = get_radar_info(radar, sweep) 

    #
    # *** Calculate bounding limits for map
    #
    radar_lat = radar.latitude['data'][0]
    radar_lon = radar.longitude['data'][0]
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
    
    #projection = ccrs.LambertConformal(radar_lon, radar_lat)
    projection = ccrs.Orthographic(radar_lon, radar_lat)
    display = pyart.graph.RadarMapDisplay(radar)

    num_fields = len(fields)
    nrows = math.ceil((num_fields)/4)
    if nrows < 1 : nrows = 1
    if num_fields <= 4:
        width=num_fields * 6
        height = float((nrows)*4.5)
        ncols=num_fields 
    elif num_fields > 4:
        width=24
        height = float((nrows)*4.5)
        ncols=round((num_fields)/2)

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
        fig = plt.figure(figsize=[width, height], constrained_layout=False)
        spec = plt.GridSpec(ncols=ncols, nrows=nrows, figure=fig, left=0.0, right=1.0, top=1.0, bottom=0, wspace=0.000000009, hspace=0.15)
    
    for index, field in enumerate(fields):
        
        units,vmin,vmax,cmap,title,Nbins = get_field_info(radar, field)
        
        if num_fields < 2:
            title = '{} {} {} {} UTC PPI Elev: {:2.1f} deg'.format(site,field,mydate,mytime,elv)
        else:
            mytitle = '{} {} {} UTC PPI {:2.1f} deg'.format(site,mydate,mytime,elv)
        
        if Nbins == 0:
            cmap=cmap
        else:
            cmap=discrete_cmap(Nbins, base_cmap=cmap)

        kwargs = {}
        kwargs.update({'transform_first': True})
        ax = fig.add_subplot(spec[r_c[index]], projection=projection)
        display.plot_ppi_map(field, sweep, vmin=vmin, vmax=vmax,
                     resolution='10m',
                     title = title,
                     projection=projection, ax=ax,
                     cmap=cmap,
                     colorbar_label=units,
                     min_lon=min_lon, max_lon=max_lon,
                     min_lat=min_lat, max_lat=max_lat,
                     lon_lines=lon_grid,lat_lines=lat_grid,
                     add_grid_lines=False,
                     lat_0=radar_lat,
                     lon_0=radar_lon,
                     embellish = False,
                     mask_outside=True)
        
        add_rings_radials(display, radar_lat, radar_lon, max_range, ax, add_logos, fig, num_fields, nrows, ncols)

        Brazil_list  = ['AL1','JG1','MC1','NT1','PE1','SF1','ST1','SV1','TM1']
        if site == 'KWAJ': 
            reef_dir = os.path.dirname(__file__)
            reader = shpreader.Reader(reef_dir + '/shape_files/ne_10m_reefs.shp')
            reef = list(reader.geometries())
            REEF = cfeature.ShapelyFeature(reef, ccrs.PlateCarree())
            ax.add_feature(REEF, facecolor='none', edgecolor='black', lw=0.25)
        if site == 'RODN': 
            island_dir = os.path.dirname(__file__)
            reader = shpreader.Reader(island_dir + '/shape_files/ne_10m_minor_islands.shp')
            island = list(reader.geometries())
            ISLAND = cfeature.ShapelyFeature(island, ccrs.PlateCarree())
            ax.add_feature(ISLAND, facecolor='none', edgecolor='black', lw=0.25)
            ax.coastlines(edgecolor='black', lw=0.25)
        if site in Brazil_list:
            ax.coastlines(edgecolor='black', lw=0.5)

        if index == num_fields-1:
            add_logo_ppi(display, radar_lat, radar_lon, max_range, ax, add_logos, fig, num_fields, nrows, ncols)
            if num_fields >= 2:
                plt.suptitle(mytitle, fontsize = 8*ncols, weight ='bold', y=(1.0+(ncols*0.025)))
                
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

# ****************************************************************************************

def plot_fields_PPI_QC(radar, sweep=0, fields=['CZ'], max_range=150, png=False, outdir='', add_logos=True):

    site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep = get_radar_info(radar, sweep) 

    #
    # *** Calculate bounding limits for map
    #
    radar_lat = radar.latitude['data'][0]
    radar_lon = radar.longitude['data'][0]
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

    num_fields = len(fields)
    nrows = math.ceil((num_fields)/4)
    if nrows < 1 : nrows = 1
    if num_fields <= 4:
        width=num_fields * 6
        height = float((nrows)*4.5)
        ncols=num_fields 
    elif num_fields > 4:
        width=24
        height = float((nrows)*4.5)
        ncols=round((num_fields)/2)

    r_c = []
    for x in range(nrows):
        for y in range(ncols):
            r_c.append((x,y))
    #
    # *** Plotting Begins
    #
    set_plot_size_parms_ppi(num_fields)

    display = pyart.graph.RadarDisplay(radar)

    if num_fields < 2:
        fig = plt.figure(figsize=[width, height], constrained_layout=False)
        spec = plt.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    else:
        fig = plt.figure(figsize=[width, height], constrained_layout=False)
        spec = plt.GridSpec(ncols=ncols, nrows=nrows, figure=fig, left=0.0, right=1.0, top=1.0, bottom=0, wspace=0.000000009, hspace=0.15)
    
    for index, field in enumerate(fields):
        
        units,vmin,vmax,cmap,title,Nbins = get_field_info(radar, field)
        
        if num_fields < 2:
            title = '{} {} {} {} UTC PPI Elev: {:2.1f} deg'.format(site,field,mydate,mytime,elv)
        else:
            mytitle = '{} {} {} UTC PPI {:2.1f} deg'.format(site,mydate,mytime,elv)
        
        if Nbins == 0:
            cmap=cmap
        else:
            cmap=discrete_cmap(Nbins, base_cmap=cmap)

        ax = fig.add_subplot(spec[r_c[index]])
        display.plot_ppi(field, sweep=sweep, vmin=vmin, vmax=vmax, cmap=cmap, 
                         colorbar_label=units, mask_outside=True, title=title,
                         axislabels_flag=False)
        display.set_limits(xlim=[-max_range,max_range], ylim=[-max_range,max_range])

        for rng in range(50,max_range+50,50):
            display.plot_range_ring(rng, col = 'k', ls='-', lw=0.5)
        display.plot_grid_lines(col="k", ls=":")
        display.set_aspect_ratio(aspect_ratio=1.0)
        ax.set_xticklabels(lat_grid, rotation = 45)
        plt.yticks(rotation=90, va = 'center')
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Longitude")
    
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

# ****************************************************************************************

def plot_fields_RHI(radar, sweep=0, fields=['CZ'], ymax=10, xmax=150, png=False, outdir='', add_logos=True):

    #
    # *** Get radar elevation, date, time
    # *** Rename fields to a standard for plotting purposes
    #
    
    site, mydate, mytime, azi, year, month, day, hh, mm, ss, string_csweep = get_radar_info(radar, sweep)

    #
    # *** Set bounding limits for plot
    #
    xlim=[0,xmax]
    ylim=[0,ymax]
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
    display = pyart.graph.RadarMapDisplay(radar)
    
    if num_fields < 2:
        fig = plt.figure(figsize=[width, height], constrained_layout=False)
    else:
        fig = plt.figure(figsize=[width, height], constrained_layout=True)
   
    spec = plt.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
        
    for index, field in enumerate(fields):

        units,vmin,vmax,cmap,title,Nbins = get_field_info(radar, field)

        if Nbins == 0:
            cmap=cmap
        else:
            cmap=discrete_cmap(Nbins, base_cmap=cmap)
        
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
        
        if add_logos:  annotate_plot_rhi(ax,fig,num_fields,nrows)
        
        if field == 'FH' or field == 'FH2': display.cbs[index] = adjust_fhc_colorbar_for_pyart(display.cbs[index])
        if field == 'MRC' or field == 'MRC2': display.cbs[index] = adjust_meth_colorbar_for_pyart(display.cbs[index])
        if field == 'FS' or field == 'FH2': display.cbs[index] = adjust_fhc_colorbar_for_pyart(display.cbs[index])
        if field == 'FW' or field == 'FH2': display.cbs[index] = adjust_fhw_colorbar_for_pyart(display.cbs[index])

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
            fig.savefig(outdir_daily + '/' + png_file, dpi=240, bbox_inches='tight')
            print('  --> ' + outdir_daily + '/' + png_file, '', sep='\n')
        elif num_fields >1:
            png_file = '{}_{}_{}_{}_{}panel_{:2.1f}AZ_RHI.png'.format(site,year
                                                    ,month+day,hh+mm+ss,num_fields,azi)
            outdir_multi = outdir + '/' + year + '/' + month + day + '/multi/'
            os.makedirs(outdir_multi, exist_ok=True)
            fig.savefig(outdir_multi + '/' + png_file, dpi=240, bbox_inches='tight')
            print('  --> ' + outdir_multi + '/' + png_file, '', sep='\n')
        if outdir == '':
            outdir = os.getcwd()
    else:
        plt.show()

# ****************************************************************************************

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
        vmax=70
        Nbins = 14
        title = 'Corrected Reflectivity [dBZ]'
        cmap='pyart_NWSRef'
    elif field == 'DZ':
        units='Zh [dBZ]'
        vmin=0
        vmax=70
        Nbins = 14
        title = 'RAW Reflectivity [dBZ]'
        cmap='pyart_NWSRef'
    elif field == 'DR':
        units='Zdr [dB]'
        vmin=-1
        vmax=3
        Nbins = 16
        title = 'Differential Reflectivity [dB]'
        cmap='pyart_HomeyerRainbow'
    elif field == 'VR':
        units='Velocity [m/s]'
        vmin=-15
        vmax=15
        Nbins = 15
        title = 'Radial Velocity [m/s]'
        cmap='pyart_balance'
    elif field == 'corrected_velocity':
        units='Velocity [m/s]'
        vmin=-15
        vmax=15
        Nbins = 15
        title = 'Dealiased Radial Velocity [m/s]'
        cmap='pyart_balance'
    elif field == 'KD':
        units='Kdp [deg/km]'
        vmin=-2
        vmax=5
        Nbins = 8
        title = 'Specific Differential Phase [deg/km]'
        cmap='pyart_HomeyerRainbow'
    elif field == 'KDPB':
        units='Kdp [deg/km]'
        vmin=-2
        vmax=5
        Nbins = 8
        title = 'Specific Differential Phase [deg/km] (Bringi)'
        cmap='pyart_HomeyerRainbow'
    elif field == 'PH':
        units='PhiDP [deg]'
        vmin=0
        vmax=360
        Nbins = 36
        title ='Differential Phase [deg]' 
        cmap='pyart_Carbone42'
    elif field == 'PHM':
        units='PhiDP [deg]'
        vmin=0
        vmax=360
        Nbins = 36
        title ='Differential Phase [deg] Marks' 
        cmap='pyart_Carbone42'
    elif field == 'RH':
        units='Correlation'
        vmin=0.8
        vmax=1.0
        Nbins = 9
        title = 'Correlation Coefficient'
        cmap='pyart_LangRainbow12'
    elif field == 'SD':
        units='Std(PhiDP)'
        vmin=0
        vmax=70
        Nbins = 14
        title = 'Standard Deviation of PhiDP'
        cmap='pyart_NWSRef'
    elif field == 'SQ':
        units='SQI'
        vmin=0
        vmax=1
        Nbins = 10
        title = 'Signal Quality Index'
        cmap='pyart_LangRainbow12'
    elif field == 'FH':
        units='HID'
        vmin=0
        vmax=11
        Nbins = 0
        title = 'Summmer Hydrometeor Identification'
        cmap=cmaphid
    elif field == 'FS':
        units='HID'
        vmin=0
        vmax=11
        Nbins = 0
        title = 'Summer Hydrometeor Identification'
        cmap=cmaphid
    elif field == 'FW':
        units='HID'
        vmin=0
        vmax=8
        Nbins = 0
        title = 'Winter Hydrometeor Identification'
        cmap=cmaphidw
    elif field == 'MW':
        units='Water Mass [g/m^3]'
        vmin=-1
        vmax=3
        Nbins = 8
        title = 'Water Mass [g/m^3]'
        cmap='pyart_BlueBrown10'
    elif field == 'MI':
        units='Ice Mass [g/m^3]'
        vmin=-1
        vmax=3
        Nbins = 8
        title ='Ice Mass [g/m^3]'
        cmap='pyart_BlueBrown10'
    elif field == 'RC':
        units='HIDRO Rain Rate [mm/hr]'
        vmin=0
        vmax=160
        Nbins = 30
        title ='HIDRO Rain Rate [mm/hr]'
        cmap='pyart_NWSRef'
    elif field == 'RP':
        units='PolZR Rain Rate [mm/hr]'
        vmin=0
        vmax=160
        Nbins = 30
        title ='PolZR Rain Rate [mm/hr]'
        cmap='pyart_NWSRef'    
    elif field == 'MRC':
        units='HIDRO Method'
        vmin=0
        vmax=5
        Nbins = 0
        title ='HIDRO Method'
        cmap=cmapmeth
    elif field == 'DM':
        units='DM [mm]'
        vmin=0
        vmax=4
        Nbins = 8
        title ='DM [mm]'
        cmap='pyart_BlueBrown10'
    elif field == 'NW':
        units='Log[Nw, m^-3 mm^-1]'
        vmin=0
        vmax=6
        Nbins = 12
        title ='Log[Nw, m^-3 mm^-1]'
        cmap='pyart_BlueBrown10'

    return units,vmin,vmax,cmap,title,Nbins

# ****************************************************************************************

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
    if 'site_name' in radar.metadata.keys():
        site = radar.metadata['site_name'].upper()
    elif 'instrument_name' in radar.metadata.keys():
        if isinstance(radar.metadata['instrument_name'], bytes):
            site = radar.metadata['instrument_name'].decode().upper()
        else:
            site = radar.metadata['instrument_name'].upper()
    else:
        site=''

    if site == 'NPOL1': site = 'NPOL'         
    if site == 'LAVA1': site = 'KWAJ'
    if site == b'AN1-P\x00\x00\x00': site = 'AL1'
    if site == b'JG1-P\x00\x00\x00': site = 'JG1'
    if site == b'MC1-P\x00\x00\x00': site = 'MC1'
    if site == b'NT1-P\x00\x00\x00': site = 'NT1'
    if site == b'PE1-P\x00\x00\x00': site = 'PE1'
    if site == b'SF1-P\x00\x00\x00': site = 'SF1'
    if site == b'ST1-P\x00\x00\x00': site = 'ST1'
    if site == b'SV1-P\x00\x00\x00': site = 'SV1'
    if site == b'TM1-P\x00\x00\x00': site = 'TM1'

    return site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep

# ****************************************************************************************

def add_logo_ppi(display, radar_lat, radar_lon, max_range, ax, add_logos, fig, num_fields, nrows, ncols):
 
    # Annotate NASA GPM logos
    if add_logos:
        logo_dir = os.path.dirname(__file__)
        nasalogo = Image.open(logo_dir + '/nasa.png')
        gpmlogo = Image.open(logo_dir + '/gpm.png')
        
        if num_fields < 2:
            imageboxnasa = OffsetImage(nasalogo, zoom=0.06)
            imageboxgpm = OffsetImage(gpmlogo, zoom=0.03)
            imageboxnasa.image.axes = ax
            imageboxgpm.image.axes = ax 
    
            abnasa = AnnotationBbox(imageboxnasa,[0,0], xybox=[.095, .93],
                                    xycoords= 'axes pixels', boxcoords='axes fraction',
                                    pad=0.0, frameon=False)
            abgpm = AnnotationBbox(imageboxgpm,[0,0], xybox=[.89, .93],                               
                                   xycoords= 'axes pixels', boxcoords='axes fraction',
                                   pad=0.0, frameon=False)
            ax.add_artist(abnasa)
            ax.add_artist(abgpm)
        else:
            imageboxnasa = OffsetImage(nasalogo, zoom=0.035*ncols)
            imageboxgpm = OffsetImage(gpmlogo, zoom=0.018*ncols)
            imageboxnasa.image.axes = fig
            imageboxgpm.image.axes = fig
            abnasa = AnnotationBbox(imageboxnasa,[0,0], xybox=[ncols/100, 1.0+(ncols*0.02)],
                                    xycoords= 'figure pixels', boxcoords='figure fraction',
                                    pad=0.0, frameon=False)
            abgpm = AnnotationBbox(imageboxgpm,[0,0], xybox=[3.8/ncols, 1.0+(ncols*0.03)],                               
                                   xycoords= 'figure pixels', boxcoords='figure fraction',
                                   pad=0.0, frameon=False)
            fig.add_artist(abnasa)
            fig.add_artist(abgpm)
    
    return

# ****************************************************************************************

def add_rings_radials(display, radar_lat, radar_lon, max_range, ax, add_logos, fig, num_fields, nrows, ncols):

    # NASA WFF instrument pad locations
    Pad_lon = -75.471
    Pad_lat =  37.934
    PCMK_lon = -75.515
    PCMK_lat =  38.078
    dtor = math.pi/180.0

    maxrange_meters = max_range * 1000.
    meters_to_lat = 1. / 111177.
    meters_to_lon =  1. / (111177. * math.cos(radar_lat * dtor))
    display.plot_cross_hair(10,npts=100)
    #for rng in range(20,max_range+20,20):
    #    display.plot_range_ring(rng, line_style='k--', lw=0.5)
    for rng in range(50,max_range+50,50):
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
    '''
    lakes = cfeature.NaturalEarthFeature(
                                category='physical',
                                name='lakes',
                                scale='10m',
                                facecolor='lightcyan')
    ax.add_feature(lakes, edgecolor='black', lw=0.25, zorder=0)
    '''
    county_dir = os.path.dirname(__file__)
    reader = shpreader.Reader(county_dir + '/shape_files/countyl010g.shp')
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='black', lw=0.25)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),facecolor=("lightcyan"))
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor=("wheat"), edgecolor=None, alpha=1)
    ax.add_feature(cfeature.LAKES.with_scale('10m'),facecolor=("lightcyan"), edgecolor='black',  lw=0.25, zorder=0)
    #ax.add_feature(cfeature.RIVERS,facecolor=("lightcyan"), edgecolor="lightcyan", zorder=0)

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
  
def annotate_plot_rhi(ax,fig,num_fields,nrows):
    
    # Annotate NASA GPM logos
    logo_dir = os.path.dirname(__file__)
    nasalogo = Image.open(logo_dir + '/nasa.png')
    gpmlogo = Image.open(logo_dir + '/gpm.png')
    
    if num_fields < 2:
        imageboxnasa = OffsetImage(nasalogo, zoom=0.07)
        imageboxgpm = OffsetImage(gpmlogo, zoom=0.03)
        imageboxnasa.image.axes = ax
        imageboxgpm.image.axes = ax
        abnasa = AnnotationBbox(imageboxnasa,[0,0], xybox=[.065, .915],
                                xycoords= 'axes pixels', boxcoords='axes fraction',
                                pad=-10.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm,[0,0], xybox=[.93, .925],                               
                               xycoords= 'axes pixels', boxcoords='axes fraction',
                               pad=0.0, frameon=False)
        ax.add_artist(abnasa)
        ax.add_artist(abgpm)
    else:
        imageboxnasa = OffsetImage(nasalogo, zoom=0.15)
        imageboxgpm = OffsetImage(gpmlogo, zoom=0.08)
        imageboxnasa.image.axes = fig
        imageboxgpm.image.axes = fig
        abnasa = AnnotationBbox(imageboxnasa,[0,0], xybox=[140,250*nrows],
                                xycoords= 'figure points', boxcoords='figure points',
                                pad=0.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm,[0,0], xybox=[1550,250*nrows],
                               xycoords= 'figure points', boxcoords='figure points',
                               pad=0.0, frameon=False)
        fig.add_artist(abnasa)
        fig.add_artist(abgpm)
    
    return

# ****************************************************************************************

def adjust_fhc_colorbar_for_pyart(cb):
    '''
    HID types:           Species #:
    -------------------------------
    Unclassified             0
    Drizzle                  1
    Rain                     2
    Ice Crystals             3
    Aggregates               4
    Wet Snow                 5
    Vertical Ice             6
    Low-Density Graupel      7
    High-Density Graupel     8
    Hail                     9
    Big Drops                10
    '''

    cb.set_ticks(np.arange(0.5, 11, 1.0))
    #cb.ax.set_yticklabels(['UC', 'DZ', 'RN', 'CR', 'DS',
    #                       'WS', 'VI', 'LDG',
    #                       'HDG', 'HA', 'BD'])
    cb.ax.set_yticklabels(['No Echo', 'Drizzle', 'Rain', 'Ice Crystals', 
                           'Aggregates', 'Wet Snow', 'Vertical Ice', 
                           'LD Graupel', 'HD Graupel', 'Hail', 'Big Drops'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

# ****************************************************************************************

def adjust_fhw_colorbar_for_pyart(cb):
    cb.set_ticks(np.arange(0.5, 8, 1.0))
    '''
    Cateories:
    0  = Unclassified
    1  = Ice Crystals
    2  = Plates
    3  = Dendrites
    4  = Aggregates
    5  = Wet Snow
    6  = Frozen precip
    7  = Rain
    '''
    cb.ax.set_yticklabels(['No Echo', 'Ice Crystals', 'Plates', 'Dendrites', 
                           'Aggregates', 'Wet Snow','Frozen Precip',
                           'Rain'])
    #cb.ax.set_yticklabels(['UC', 'IC', 'PL', 'DE', 'AG',
    #                       'WS', 'FP', 'RA'])
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

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    
    color_list = base(np.linspace(0, 1, N, 0))
    cmap_name = base.name + str(N)
    return plt.cm.colors.ListedColormap(color_list, color_list, N)

# ****************************************************************************************
def plot_conv_strat(self):

    self.ds['convsf'].values[self.ds['convsf'].values == 0] = np.nan

    radar_lat = self.radar.latitude['data'][0]
    radar_lon = self.radar.longitude['data'][0]

    dtor = math.pi/180.0
    meters_to_lat = 1. / 111177.
    meters_to_lon =  1. / (111177. * math.cos(radar_lat * dtor))
    lonl = radar_lon + math.cos((90.-90) * dtor) * meters_to_lon * (200 * 1000.)
    lonr = radar_lon + math.cos((90.-270) * dtor) * meters_to_lon * (200 * 1000.)
    latt = radar_lat + math.sin((90.-0) * dtor) * meters_to_lat * (200 * 1000.)
    latb = radar_lat + math.sin((90.-180) * dtor) * meters_to_lat * (200 * 1000.)

    #Get first and last datetime from DS
    time = self.ds.time.values[0]
    datetime = str(time)
    DT = pd.to_datetime(str(datetime))
    year = str(DT.year).zfill(4)
    month = str(DT.month).zfill(2)
    day = str(DT.day).zfill(2)
    hour = str(DT.hour).zfill(2)
    min = str(DT.minute).zfill(2)
    site = 'MELB'
    field = 'CS'
    mydate = month + '/' + day + '/' + year
    mytime = hour + ':' + min
    elv = 0

    title = '{} {} {} {} UTC PPI Elev: {:2.1f} deg'.format(site,field,mydate,mytime,elv)

    fig = plt.figure(facecolor='white')
    proj = ccrs.PlateCarree()
    projection = ccrs.LambertConformal(central_latitude=self.radar.latitude['data'][0],
                                       central_longitude=self.radar.longitude['data'][0])
    
    display = pyart.graph.GridMapDisplay(self.cs_grid)
    args = {}
    args.update({'aspect': 1.0})
    ax = plt.axes(projection=proj, **args)
    map_limit=[lonr-0.01, lonl+0.01, latb-0.01, latt+0.01]
    ax.set_extent(map_limit, crs=proj)
    ax=plt.subplot(1,1,1)

    display.plot_grid('convsf', vmin=0, vmax=2, cmap=plt.get_cmap('viridis', 3), projection=proj,
                  transform=proj, ax=ax, ticks=[1/3, 1, 5/3],
                  ticklabs=['', 'Stratiform', 'Convective'],
                  embellish = False, add_grid_lines=False, title=title)

    # Add state and countines to map
    states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none')

    reader = shpreader.Reader('/Users/jpippitt/GVradarV1.0/gvradar/county/countyl010g.shp')
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='black', lw=0.25)

    # Add cartopy grid lines
    grid_lines = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', x_inline=False)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    #grid_lines.xpadding = 5
    grid_lines.xformatter = LONGITUDE_FORMATTER
    grid_lines.yformatter = LATITUDE_FORMATTER
    grid_lines.xlabel_style = {'size': 6, 'color': 'black', 'rotation': 0, 'weight': 'bold', 'ha': 'center'}
    grid_lines.ylabel_style = {'size': 6, 'color': 'black', 'rotation': 90, 'weight': 'bold', 'va': 'bottom', 'ha': 'center'}

    dtor = math.pi/180.0

    maxrange_meters = 200 * 1000.
    meters_to_lat = 1. / 111177.
    meters_to_lon =  1. / (111177. * math.cos(radar_lat * dtor))

    #for rng in range(50,250,1):
    #    plot_range_ring(ax, rng, line_style='k--', lw=0.5)
    get_rings(self, ax)
        
    for azi in range(0,360,30):
        azimuth = 90. - azi
        dazimuth = azimuth * dtor
        lon_maxrange = radar_lon + math.cos(dazimuth) * meters_to_lon * maxrange_meters
        lat_maxrange = radar_lat + math.sin(dazimuth) * meters_to_lat * maxrange_meters
        plot_line_geo(ax, [radar_lon, lon_maxrange], [radar_lat, lat_maxrange])

    ax.add_feature(cfeature.OCEAN,facecolor=("lightcyan"), scale='10m')
    ax.add_feature(cfeature.LAND, facecolor=("wheat"), edgecolor=None, alpha=1, scale='10m')
    ax.add_feature(cfeature.LAKES,facecolor=("lightcyan"), scale='10m')

# ****************************************************************************************

def plot_range_ring(ax, range_ring_location_km, npts=360,
                    color='k', line_style='-', **kwargs):

    angle = np.linspace(0., 2.0 * np.pi, npts)
    xpts = range_ring_location_km * 1000. * np.sin(angle)
    ypts = range_ring_location_km * 1000. * np.cos(angle)
    plot_line_xy(xpts, ypts, ax, color=color, line_style=line_style,
                          **kwargs)

def plot_line_xy(line_x, line_y, ax, color='r', line_style='solid',
                     **kwargs):
    kwargs = {}
    kwargs.update({'transform': cartopy.crs.PlateCarree()})
    ax.plot(line_x, line_y, color, line_style, **kwargs)

def plot_line_geo(ax, line_lons, line_lats, line_style='r-', **kwargs):
    
    kwargs = {}
    kwargs.update({'transform': cartopy.crs.PlateCarree()})
    ax.plot(line_lons, line_lats, color="black", linestyle = 'dashed', linewidth=0.5, **kwargs)

def get_rings(self, ax):

    radar_lat = self.radar.latitude['data'][0]
    radar_lon = self.radar.longitude['data'][0]
    dtor = math.pi/180.0
    maxrange_meters = 200 * 1000.
    meters_to_lat = 1. / 111177.
    meters_to_lon =  1. / (111177. * math.cos(radar_lat * dtor))
    kwargs = {}
    kwargs.update({'transform': cartopy.crs.PlateCarree()})

    for azi in range(0,360,1):
        range_ring_meters = 50 * 1000.
        azimuth = 90. - azi
        dazimuth = azimuth * dtor
        lon_maxrange = radar_lon + math.cos(dazimuth) * meters_to_lon * range_ring_meters
        lat_maxrange = radar_lat + math.sin(dazimuth) * meters_to_lat * range_ring_meters
        ax.plot([lon_maxrange-.001, lon_maxrange], [lat_maxrange-.001, lat_maxrange], color="black", linestyle = 'solid', linewidth=0.5, **kwargs)
    for azi in range(0,360,1):
        range_ring_meters = 100 * 1000.
        azimuth = 90. - azi
        dazimuth = azimuth * dtor
        lon_maxrange = radar_lon + math.cos(dazimuth) * meters_to_lon * range_ring_meters
        lat_maxrange = radar_lat + math.sin(dazimuth) * meters_to_lat * range_ring_meters
        ax.plot([lon_maxrange-.001, lon_maxrange], [lat_maxrange-.001, lat_maxrange], color="black", linestyle = 'solid', linewidth=0.5, **kwargs)
    for azi in range(0,360,1):
        range_ring_meters = 150 * 1000.
        azimuth = 90. - azi
        dazimuth = azimuth * dtor
        lon_maxrange = radar_lon + math.cos(dazimuth) * meters_to_lon * range_ring_meters
        lat_maxrange = radar_lat + math.sin(dazimuth) * meters_to_lat * range_ring_meters
        ax.plot([lon_maxrange-.001, lon_maxrange], [lat_maxrange-.001, lat_maxrange], color="black", linestyle = 'solid', linewidth=0.5, **kwargs)
    for azi in range(0,360,1):
        range_ring_meters = 200 * 1000.
        azimuth = 90. - azi
        dazimuth = azimuth * dtor
        lon_maxrange = radar_lon + math.cos(dazimuth) * meters_to_lon * range_ring_meters
        lat_maxrange = radar_lat + math.sin(dazimuth) * meters_to_lat * range_ring_meters
        ax.plot([lon_maxrange-.001, lon_maxrange], [lat_maxrange-.001, lat_maxrange], color="black", linestyle = 'solid', linewidth=0.5, **kwargs)    
# ****************************************************************************************
