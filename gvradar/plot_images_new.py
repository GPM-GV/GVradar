import numpy as np
import math
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['backend'] = 'Qt5Agg'
matplotlib.rcParams['agg.path.chunksize'] = 0
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import pyart
import datetime
from cftime import date2num, num2date
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
# Custom Colormaps for Polarimetric Variables
# ****************************************************************************************

# Refined rhoHV colormap (0.7-1.0 range with white for < 0.7)
scale_rhohv = [
    (1.0, 1.0, 1.0),                 # < 0.70 - white (poor quality)
    (0.7969, 0.9961, 0.9961),        # 0.70 - cyan/white
    (0.0156, 0.9102, 0.9023),        # 0.75 - bright cyan
    (0.0039, 0.6211, 0.9531),        # 0.80 - blue-cyan
    (0.0117, 0, 0.9531),             # 0.85 - blue
    (0.0078, 0.9883, 0.0078),        # 0.88 - bright green
    (0.0039, 0.7695, 0.0039),        # 0.90 - green
    (0, 0.5547, 0),                  # 0.92 - dark green
    (0.9883, 0.9688, 0.0078),        # 0.94 - yellow
    (0.8945, 0.7344, 0),             # 0.95 - orange-yellow
    (0.9883, 0.582, 0),              # 0.96 - orange
    (0.9883, 0, 0),                  # 0.97 - red
    (0.8281, 0, 0),                  # 0.98 - dark red
    (0.7344, 0, 0),                  # 0.985 - darker red
    (0.6, 0, 0),                     # 0.99 - maroon
    (0.4, 0, 0),                     # 0.994 - dark maroon
    (0.2, 0, 0)                      # 0.997-1.0 - very dark red
]

cbar_limits_rhohv = [
    0.0, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 
    0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.994, 0.997, 1.00
]

rhohv_cmap = colors.LinearSegmentedColormap.from_list('rhohv_refined', scale_rhohv)
rhohv_norm = colors.BoundaryNorm(cbar_limits_rhohv, rhohv_cmap.N)

# Refined ZDR colormap (-2 to 5 dB range)
scale_zdr = [
    (0.5, 0, 0.5),           # -2.0 dB - purple (artifacts/vertical ice)
    (0.3, 0.3, 0.8),         # -1.5 dB - blue-purple
    (0.2, 0.5, 0.9),         # -1.0 dB - blue
    (0.5, 0.7, 0.9),         # -0.5 dB - light blue
    (0.8, 0.8, 0.8),         #  0.0 dB - gray (small/spherical drops)
    (0.6, 0.9, 0.6),         #  0.5 dB - light green (drizzle)
    (0.2, 0.8, 0.2),         #  1.0 dB - green (light rain)
    (0.0, 0.6, 0.0),         #  1.5 dB - dark green (moderate rain)
    (0.9, 0.9, 0.0),         #  2.0 dB - yellow (heavy rain)
    (0.9, 0.7, 0.0),         #  2.5 dB - orange-yellow
    (0.9, 0.5, 0.0),         #  3.0 dB - orange (large drops)
    (0.9, 0.2, 0.0),         #  3.5 dB - red-orange
    (0.8, 0.0, 0.0),         #  4.0 dB - red (very large drops/melting)
    (0.5, 0.0, 0.0),         #  5.0 dB - dark red
]

cbar_limits_zdr = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

zdr_cmap = colors.LinearSegmentedColormap.from_list('zdr_refined', scale_zdr)
zdr_norm = colors.BoundaryNorm(cbar_limits_zdr, zdr_cmap.N)

# ****************************************************************************************

class PlottingCache:
    """Cache manager for expensive plotting operations - optimized like GVview"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._map_features_cache = None
            self._logos_cache = {}
            self._field_cache = {}
            self._coordinate_cache = {}
            self._ocean_cache = None
            self._lakes_cache = None
            self._initialized = True
    
    def get_map_features(self):
        """Get all map features (counties, states, reefs, minor islands)"""
        if self._map_features_cache is None:
            print("Loading map features (first time only)...")
            start = time.time()
            self._map_features_cache = self._load_all_map_features()
            print(f"Map features loaded in {time.time() - start:.2f}s")
        return self._map_features_cache
    
    def get_ocean_feature(self):
        """Get cached ocean feature"""
        if self._ocean_cache is None:
            self._ocean_cache = cfeature.OCEAN.with_scale('10m')
        return self._ocean_cache
    
    def get_lakes_feature(self):
        """Get cached lakes feature"""
        if self._lakes_cache is None:
            self._lakes_cache = cfeature.LAKES.with_scale('10m')
        return self._lakes_cache
    
    def _load_all_map_features(self):
        """Load counties, states, reefs, and minor islands"""
        try:
            base_dir = os.path.dirname(__file__)
            shapefile_dir = os.path.join(base_dir, "shape_files/")
            
            # ==================== Load US Counties ====================
            COUNTIES = None
            county_shapefile = os.path.join(shapefile_dir, "countyl010g.shp")
            if os.path.exists(county_shapefile):
                try:
                    reader = shpreader.Reader(county_shapefile)
                    counties = list(reader.geometries())
                    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
                except Exception as e:
                    print(f"⚠️ Could not load counties: {e}")
    
            # ==================== Load States ====================
            STATES = cfeature.NaturalEarthFeature(
                                        category='cultural',
                                        name='admin_1_states_provinces_lines',
                                        scale='10m',
                                        facecolor='none')
            
            # ==================== Load Reefs ====================
            REEFS = None
            reef_shapefile = os.path.join(shapefile_dir, "ne_10m_reefs.shp")
            if os.path.exists(reef_shapefile):
                try:
                    reef_reader = shpreader.Reader(reef_shapefile)
                    reefs = list(reef_reader.geometries())
                    REEFS = cfeature.ShapelyFeature(reefs, ccrs.PlateCarree())
                except Exception as e:
                    print(f"⚠️ Could not load reefs: {e}")
            
            # ==================== Load Minor Islands ====================
            MINOR_ISLANDS = None
            islands_shapefile = os.path.join(shapefile_dir, "ne_10m_minor_islands.shp")
            if os.path.exists(islands_shapefile):
                try:
                    islands_reader = shpreader.Reader(islands_shapefile)
                    islands = list(islands_reader.geometries())
                    MINOR_ISLANDS = cfeature.ShapelyFeature(islands, ccrs.PlateCarree())
                except Exception as e:
                    print(f"⚠️ Could not load minor islands: {e}")
            
            return COUNTIES, STATES, REEFS, MINOR_ISLANDS
            
        except Exception as e:
            print(f"Error loading shapefiles: {e}")
            STATES = cfeature.NaturalEarthFeature(
                                        category='cultural',
                                        name='admin_1_states_provinces_lines',
                                        scale='50m',
                                        facecolor='none')
            return None, STATES, None, None
    
    def get_logo(self, logo_name):
        if logo_name not in self._logos_cache:
            logo_dir = os.path.dirname(__file__)
            logo_path = os.path.join(logo_dir, f'{logo_name}.png')
            self._logos_cache[logo_name] = Image.open(logo_path)
        return self._logos_cache[logo_name]
    
    def get_processed_field(self, radar, field_name):
        cache_key = f"{field_name}_{id(radar)}"
        if cache_key not in self._field_cache:
            self._field_cache[cache_key] = self._process_field(radar, field_name)
        return self._field_cache[cache_key]
    
    def _process_field(self, radar, field_name):
        if field_name == 'RC':
            rc = radar.fields['RC']['data'].copy()
            rc[rc < 0.01] = np.nan
            return {"data": rc, "units": "mm/h",
                   "long_name": "HIDRO Rainfall Rate", "_FillValue": -32767.0,
                   "standard_name": "HIDRO Rainfall Rate"}
        elif field_name == 'RP':
            rp = radar.fields['RP']['data'].copy()
            rp[rp < 0.01] = np.nan
            return {"data": rp, "units": "mm/h",
                   "long_name": "Polzr_Rain_Rate", "_FillValue": -32767.0,
                   "standard_name": "Polzr_Rain_Rate"}
        elif field_name == 'RA':
            ra = radar.fields['RA']['data'].copy()
            ra[ra < 0.01] = np.nan
            return {"data": ra, "units": "mm/h",
                   "long_name": "A_Rain_Rate", "_FillValue": -32767.0,
                   "standard_name": "A_Rain_Rate"}
        return None
    
    def get_coordinate_transform(self, radar_lat, radar_lon, max_range):
        # Use rounded values for stable cache keys
        cache_key = f"{round(radar_lat, 6)}_{round(radar_lon, 6)}_{int(max_range)}"
        if cache_key not in self._coordinate_cache:
            self._coordinate_cache[cache_key] = self._calculate_coordinates(radar_lat, radar_lon, max_range)
        return self._coordinate_cache[cache_key]
    
    def _calculate_coordinates(self, radar_lat, radar_lon, max_range):
        dtor = math.pi/180.0
        maxrange_meters = max_range * 1000.
        meters_to_lat = 1. / 111177.
        meters_to_lon = 1. / (111177. * math.cos(radar_lat * dtor))

        min_lat = radar_lat - maxrange_meters * meters_to_lat
        max_lat = radar_lat + maxrange_meters * meters_to_lat
        min_lon = radar_lon - maxrange_meters * meters_to_lon
        max_lon = radar_lon + maxrange_meters * meters_to_lon
        
        lon_grid = np.arange(round(min_lon, 2) - 1.00, round(max_lon, 2) + 1.00, 1.0)
        lat_grid = np.arange(round(min_lat, 2) - 1.00, round(max_lat, 2) + 1.00, 1.0)
        
        return {
            'min_lat': min_lat, 'max_lat': max_lat,
            'min_lon': min_lon, 'max_lon': max_lon,
            'lon_grid': lon_grid, 'lat_grid': lat_grid,
            'meters_to_lat': meters_to_lat, 'meters_to_lon': meters_to_lon
        }

# Initialize cache and configure matplotlib
_cache = PlottingCache()

def configure_matplotlib():
    """Configure matplotlib settings once"""
    plt.ioff()  # Turn off interactive mode for batch processing
    plt.rcParams.update({
        'font.weight': 'bold',
        'axes.facecolor': 'black',
        'agg.path.chunksize': 0,
        'figure.max_open_warning': 0,
        'axes.formatter.useoffset': False
    })

configure_matplotlib()

# Global flag to track if warmup has been done
_plotting_warmed_up = False

# ****************************************************************************************

def warmup_plotting_engine(radar, sweep, max_range):
    """
    Warm up PyART plotting engine by creating a dummy single plot.
    This initializes coordinate transformations and Cartopy projections.
    Similar to how GVview naturally warms up when showing single plot first.
    """
    print("  Warming up plotting engine (first time only)...")
    warmup_start = time.time()
    
    radar_lat = radar.latitude['data'][0]
    radar_lon = radar.longitude['data'][0]
    
    # Create a minimal figure (don't display or save it)
    projection = ccrs.LambertConformal(radar_lon, radar_lat)
    
    # Create dummy figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), subplot_kw={'projection': projection})
    ax.set_facecolor('black')
    
    # Create display object and do ONE simple plot to warm up
    display = pyart.graph.RadarMapDisplay(radar)
    
    # Get first available field
    field = list(radar.fields.keys())[0]
    
    # Do a minimal plot_ppi_map to initialize everything
    try:
        coord_data = _cache.get_coordinate_transform(radar_lat, radar_lon, max_range)
        display.plot_ppi_map(field, sweep, 
                           resolution='50m', 
                           projection=projection, 
                           ax=ax,
                           min_lon=coord_data['min_lon'], 
                           max_lon=coord_data['max_lon'],
                           min_lat=coord_data['min_lat'], 
                           max_lat=coord_data['max_lat'],
                           add_grid_lines=False, 
                           lat_0=radar_lat, 
                           lon_0=radar_lon,
                           embellish=False)
    except Exception as e:
        print(f"    Warmup plot failed (non-critical): {e}")
    
    # Close without displaying or saving
    plt.close(fig)
    
    print(f"  Warmup complete in {time.time() - warmup_start:.2f}s")

# ****************************************************************************************

def calculate_layout(num_fields):
    """Calculate optimal layout for given number of fields"""
    nrows = math.ceil(num_fields / 4)
    if nrows < 1: 
        nrows = 1
    
    if num_fields <= 4:
        width = num_fields * 6
        height = float(nrows * 4.5)
        ncols = num_fields 
    else:
        height = float(nrows * 4.5)
        ncols = round(num_fields / 2)
        width = ncols * 6

    positions = []
    for x in range(nrows):
        for y in range(ncols):
            positions.append((x, y))
    
    return {
        'nrows': nrows,
        'ncols': ncols,
        'width': width,
        'height': height,
        'positions': positions
    }

def calculate_layout_rhi(num_fields):
    """Calculate RHI layout"""
    nrows = (num_fields + 1) // 2
    if num_fields < 2:
        width = 12
        height = 3.5
        ncols = 1
    else:
        width = 24
        height = 3.5 * nrows
        ncols = 2

    positions = []
    for y in range(ncols):
        for x in range(nrows):
            positions.append((x, y))
    
    return {
        'nrows': nrows,
        'ncols': ncols,
        'width': width,
        'height': height,
        'positions': positions
    }

def create_figure(layout):
    """Create optimized figure"""
    return plt.figure(figsize=[layout['width'], layout['height']], constrained_layout=False)

def create_gridspec(layout, fig):
    """Create optimized grid specification with space for title"""
    if layout['ncols'] < 2:
        return plt.GridSpec(ncols=layout['ncols'], nrows=layout['nrows'], figure=fig)
    else:
        # Leave space at TOP for title and logos
        return plt.GridSpec(ncols=layout['ncols'], nrows=layout['nrows'], figure=fig, 
                          left=0.0, right=1.0, top=0.92, bottom=0.0,  # top=0.92 instead of 1.0!
                          wspace=0.000000009, hspace=0.15)

# ****************************************************************************************

def plot_fields(self):
    """
    Calls plotting programs based on user defined dictionary parameters.
    Optimized version with caching and vectorized operations.
    """
    global _plotting_warmed_up
    
    start = time.time()
    
    # Pre-calculate sweep numbers
    if self.sweeps_to_plot == 'all':
        sweepn = list(range(len(self.radar.sweep_number['data'][:])))
    else:
        sweepn = self.sweeps_to_plot

    if self.scan_type == 'RHI':
        print('Plotting RHI images...')
        if self.plot_multi:
            for isweeps in range(len(sweepn)):
                sweep = sweepn[isweeps]
                if self.png: 
                    os.makedirs(self.plot_dir, exist_ok=True)
                plot_fields_RHI(self.radar, sweep=sweep, fields=self.fields_to_plot, 
                              ymax=self.max_height, xmax=self.max_range, png=self.png, 
                              outdir=self.plot_dir, add_logos=self.add_logos, 
                              mask_outside=self.mask_outside)
        
        if self.plot_single:
            for field in self.fields_to_plot:
                print(field)
                plot_dir = os.path.join(self.plot_dir, field)
                if self.png: 
                    os.makedirs(plot_dir, exist_ok=True)
                for isweeps in range(len(sweepn)):
                    sweep = sweepn[isweeps]
                    plot_fields_RHI(self.radar, sweep=sweep, fields=[field], 
                                  ymax=self.max_height, xmax=self.max_range, 
                                  png=self.png, outdir=plot_dir, 
                                  add_logos=self.add_logos, mask_outside=self.mask_outside)

    elif self.scan_type == 'PPI':
        print('Plotting PPI images...')
        
        # Pre-load ALL map features only if needed (cached after first load)
        if not self.plot_fast:
            COUNTIES, STATES, REEFS, MINOR_ISLANDS = _cache.get_map_features()
            OCEAN = _cache.get_ocean_feature()
            LAKES = _cache.get_lakes_feature()
            
            # WARM UP plotting engine for multi-panel plots (mimics GVview workflow)
            if self.plot_multi and len(self.fields_to_plot) > 1 and not _plotting_warmed_up:
                warmup_plotting_engine(self.radar, sweepn[0], self.max_range)
                _plotting_warmed_up = True
        else:
            COUNTIES = STATES = REEFS = MINOR_ISLANDS = OCEAN = LAKES = None
            
        if self.plot_multi:
            for isweeps in range(len(sweepn)):
                sweep = sweepn[isweeps]
                if self.png: 
                    os.makedirs(self.plot_dir, exist_ok=True)
                if self.plot_fast:
                    plot_fields_PPI_QC(self.radar, sweep=sweep, fields=self.fields_to_plot, 
                                     max_range=self.max_range, png=self.png, 
                                     outdir=self.plot_dir, add_logos=self.add_logos, 
                                     mask_outside=self.mask_outside)
                else:
                    plot_fields_PPI(self.radar, COUNTIES, STATES, REEFS, MINOR_ISLANDS,
                                  OCEAN, LAKES, sweep=sweep, 
                                  fields=self.fields_to_plot, max_range=self.max_range, 
                                  png=self.png, outdir=self.plot_dir, 
                                  add_logos=self.add_logos, mask_outside=self.mask_outside)
        
        if self.plot_single:
            for field in self.fields_to_plot:
                print(field)
                plot_dir = os.path.join(self.plot_dir, field)
                if self.png: 
                    os.makedirs(plot_dir, exist_ok=True)
                for isweeps in range(len(sweepn)):
                    sweep = sweepn[isweeps]
                    if self.plot_fast:
                        plot_fields_PPI_QC(self.radar, sweep=sweep, fields=[field], 
                                         max_range=self.max_range, png=self.png, 
                                         outdir=plot_dir, add_logos=self.add_logos, 
                                         mask_outside=self.mask_outside)
                    else:
                        plot_fields_PPI(self.radar, COUNTIES, STATES, REEFS, MINOR_ISLANDS,
                                      OCEAN, LAKES, sweep=sweep, fields=[field], 
                                      max_range=self.max_range, png=self.png, 
                                      outdir=plot_dir, add_logos=self.add_logos, 
                                      mask_outside=self.mask_outside)

    end = time.time()
    print(f'plotting time: {end - start:.2f} seconds')

# ****************************************************************************************

def plot_fields_PPI(radar, COUNTIES, STATES, REEFS, MINOR_ISLANDS, OCEAN, LAKES,
                   sweep=0, fields=['CZ'], max_range=150, 
                   mask_outside=True, png=False, outdir='', add_logos=True):
    """Optimized PPI with GVview-style rendering"""
    
    plot_start = time.time()
    
    # Get radar info
    site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep = get_radar_info(radar, sweep) 
    radar_lat = radar.latitude['data'][0]
    radar_lon = radar.longitude['data'][0]
    coord_data = _cache.get_coordinate_transform(radar_lat, radar_lon, max_range)
    
    display = pyart.graph.RadarMapDisplay(radar)
    num_fields = len(fields)
    layout = calculate_layout(num_fields)
    set_plot_size_parms_ppi(num_fields)

    # Create figure
    fig = create_figure(layout)
    spec = create_gridspec(layout, fig)
    
    # Create regular axes with GridSpec
    axes = []
    positions = []
    for index in range(num_fields):
        ax = fig.add_subplot(spec[layout['positions'][index]])
        axes.append(ax)
        positions.append(ax.get_position())
    
    # Replace with projected axes and plot
    for index, field in enumerate(fields):
        field_start = time.time()
        print(f"\n  Field {index+1}/{num_fields}: {field}")
        
        # Replace with projected axis
        ax = axes[index]
        pos = positions[index]
        ax.remove()
        
        projection = ccrs.LambertConformal(radar_lon, radar_lat)
        ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height], projection=projection)
        ax.set_facecolor('black')
        axes[index] = ax
        
        # Get field info
        units, vmin, vmax, cmap, title, Nbins, norm = get_field_info(radar, field)
        
        if num_fields < 2:
            title = f'{site} {field} {mydate} {mytime} UTC PPI Elev: {elv:2.1f} deg'
        else:
            mytitle = f'{site} {mydate} {mytime} UTC PPI {elv:2.1f} deg'
        
        if Nbins > 0:
            cmap = discrete_cmap(Nbins, base_cmap=cmap)

        # Plot
        if field in ['RC', 'RP', 'RA']:
            processed_field = _cache.get_processed_field(radar, field)
            plot_name = f"{field}_plot"
            radar.add_field(plot_name, processed_field, replace_existing=True)
            levels = [0, 5, 10, 15, 20, 25, 100, 150, 200, 250, 300]
            midnorm = MidpointNormalize(vmin=0, vcenter=25, vmax=300)
            display.plot_ppi_map(plot_name, sweep, vmin=vmin, vmax=vmax,
                               resolution='50m', title=title, projection=projection, ax=ax,
                               cmap=cmap, norm=midnorm, ticks=levels, colorbar_label=units,
                               min_lon=coord_data['min_lon'], max_lon=coord_data['max_lon'],
                               min_lat=coord_data['min_lat'], max_lat=coord_data['max_lat'],
                               lon_lines=coord_data['lon_grid'], lat_lines=coord_data['lat_grid'],
                               add_grid_lines=False, lat_0=radar_lat, lon_0=radar_lon,
                               embellish=False, mask_outside=mask_outside)
        elif field == 'DR' and norm is not None:
            display.plot_ppi_map(field, sweep, vmin=vmin, vmax=vmax,
                               resolution='50m', title=title, projection=projection, ax=ax,
                               cmap=cmap, norm=norm, ticks=cbar_limits_zdr, colorbar_label=units,
                               min_lon=coord_data['min_lon'], max_lon=coord_data['max_lon'],
                               min_lat=coord_data['min_lat'], max_lat=coord_data['max_lat'],
                               lon_lines=coord_data['lon_grid'], lat_lines=coord_data['lat_grid'],
                               add_grid_lines=False, lat_0=radar_lat, lon_0=radar_lon,
                               embellish=False, mask_outside=mask_outside)
        elif field == 'RH' and norm is not None:
            display.plot_ppi_map(field, sweep, vmin=vmin, vmax=vmax,
                               resolution='50m', title=title, projection=projection, ax=ax,
                               cmap=cmap, norm=norm, ticks=cbar_limits_rhohv, colorbar_label=units,
                               min_lon=coord_data['min_lon'], max_lon=coord_data['max_lon'],
                               min_lat=coord_data['min_lat'], max_lat=coord_data['max_lat'],
                               lon_lines=coord_data['lon_grid'], lat_lines=coord_data['lat_grid'],
                               add_grid_lines=False, lat_0=radar_lat, lon_0=radar_lon,
                               embellish=False, mask_outside=mask_outside)
        else:
            display.plot_ppi_map(field, sweep, vmin=vmin, vmax=vmax,
                               resolution='50m', title=title, projection=projection, ax=ax,
                               cmap=cmap, colorbar_label=units,
                               min_lon=coord_data['min_lon'], max_lon=coord_data['max_lon'],
                               min_lat=coord_data['min_lat'], max_lat=coord_data['max_lat'],
                               lon_lines=coord_data['lon_grid'], lat_lines=coord_data['lat_grid'],
                               add_grid_lines=False, lat_0=radar_lat, lon_0=radar_lon,
                               embellish=False, mask_outside=mask_outside)
        
        # Add features
        add_rings_radials_optimized_gvstyle(year, site, display, radar_lat, radar_lon, max_range, 
                                          ax, add_logos, fig, num_fields, layout, 
                                          COUNTIES, STATES, REEFS, MINOR_ISLANDS)
        
        adjust_special_colorbars(field, display, index)
        print(f"    Field time: {time.time() - field_start:.2f}s")
    
    # *** Add title and logos AFTER all plotting ***
    if num_fields >= 2:
        # Title with correct y position (0.98 instead of 1.1)
        plt.suptitle(mytitle, fontsize=8*layout['ncols'], weight='bold', y=0.99)
    
    # Use existing logo function
    add_logo_ppi_optimized(axes[-1], add_logos, fig, num_fields, layout)
    
    # Pre-render
    #render_start = time.time()
    #print(f"\n  Pre-rendering figure...")
    #fig.canvas.draw()
    #print(f"  Pre-render time: {time.time() - render_start:.2f}s")
    
    # Save
    save_start = time.time()
    save_plot(png, outdir, site, year, month, day, hh, mm, ss, string_csweep, fields, num_fields, 'PPI', fig)
    print(f"  Save time: {time.time() - save_start:.2f}s")
    print(f"\n  TOTAL: {time.time() - plot_start:.2f}s")
    plt.close('all')

# ****************************************************************************************

def plot_fields_PPI_QC(radar, sweep=0, fields=['CZ'], max_range=150, mask_outside=True, 
                      png=False, outdir='', add_logos=True):
    """Optimized fast PPI plotting"""
    
    site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep = get_radar_info(radar, sweep) 
    
    num_fields = len(fields)
    layout = calculate_layout(num_fields)
    
    set_plot_size_parms_ppi(num_fields)
    display = pyart.graph.RadarDisplay(radar)

    fig = create_figure(layout)
    spec = create_gridspec(layout, fig)
    
    for index, field in enumerate(fields):
        units, vmin, vmax, cmap, title, Nbins, norm = get_field_info(radar, field)
        
        if num_fields < 2:
            title = f'{site} {field} {mydate} {mytime} UTC PPI Elev: {elv:2.1f} deg'
        else:
            mytitle = f'{site} {mydate} {mytime} UTC PPI {elv:2.1f} deg'
        
        if Nbins == 0:
            cmap = cmap
        else:
            cmap = discrete_cmap(Nbins, base_cmap=cmap)

        ax = fig.add_subplot(spec[layout['positions'][index]])
        ax.set_facecolor('black')
        
        # Plot with norm if available
        if norm is not None:
            display.plot_ppi(field, sweep=sweep, vmin=vmin, vmax=vmax, cmap=cmap, 
                            norm=norm, colorbar_label=units, mask_outside=mask_outside, 
                            title=title, axislabels_flag=False)
        else:
            display.plot_ppi(field, sweep=sweep, vmin=vmin, vmax=vmax, cmap=cmap, 
                            colorbar_label=units, mask_outside=mask_outside, title=title,
                            axislabels_flag=False)
                            
        display.set_limits(xlim=[-max_range, max_range], ylim=[-max_range, max_range])

        # Add range rings efficiently
        add_range_rings_fast(display, max_range)
        display.plot_grid_lines(col="white", ls=":")
        display.set_aspect_ratio(aspect_ratio=1.0)
        
        # Clean up labels
        ax.set_xticklabels("", rotation=0)
        ax.set_yticklabels("", rotation=90)
        ax.set_xlabel("")
        ax.set_ylabel("")

        if num_fields >= 2:
            plt.suptitle(mytitle, fontsize=8*layout['ncols'], weight='bold', 
                       y=(1.0 + (layout['nrows'] * 0.055)))
    
        adjust_special_colorbars(field, display, index)
    
    save_plot(png, outdir, site, year, month, day, hh, mm, ss, string_csweep, 
             fields, num_fields, 'PPI', fig)
    plt.close('all')

# ****************************************************************************************

def plot_fields_RHI(radar, sweep=0, fields=['CZ'], ymax=10, xmax=150, png=False, 
                   outdir='', add_logos=True, mask_outside=True):
    """Optimized RHI plotting"""
    
    site, mydate, mytime, azi, year, month, day, hh, mm, ss, string_csweep = get_radar_info(radar, sweep)

    xlim = [0, xmax]
    ylim = [0, ymax]
    num_fields = len(fields)
    layout = calculate_layout_rhi(num_fields)
    
    set_plot_size_parms_rhi(num_fields)
    display = pyart.graph.RadarMapDisplay(radar)
    
    fig = create_figure_rhi(layout)
    spec = create_gridspec_rhi(layout, fig)
        
    for index, field in enumerate(fields):
        units, vmin, vmax, cmap, title, Nbins, norm = get_field_info(radar, field)

        if Nbins == 0:
            cmap = cmap
        else:
            cmap = discrete_cmap(Nbins, base_cmap=cmap)
        
        if num_fields < 2:
            title = f'{site} {field} {mydate} {mytime} UTC RHI Azi: {azi:2.1f}'
        else:
            mytitle = f'{site} {mydate} {mytime} UTC RHI {azi:2.1f} Azi'
     
        ax = fig.add_subplot(spec[layout['positions'][index]])
        ax.set_facecolor('black')
        
        zero_list = ['RC', 'RP', 'DM', 'NW']
        if field in zero_list: 
            mask_outside = True
            
        # Handle special rain rate fields
        if field in ['RC', 'RP']:
            processed_field = _cache.get_processed_field(radar, field)
            plot_name = f"{field}_plot"
            radar.add_field(plot_name, processed_field, replace_existing=True)
            
            levels = [0, 5, 10, 15, 20, 25, 100, 150, 200, 250, 300]
            midnorm = MidpointNormalize(vmin=0, vcenter=25, vmax=300)
            display.plot_rhi(plot_name, sweep, vmin=vmin, vmax=vmax, cmap=cmap,
                           title=title, mask_outside=mask_outside,
                           colorbar_label=units, norm=midnorm, ticks=levels)
        
        # Special handling for fields with custom norm
        elif norm is not None:
            ticks = cbar_limits_zdr if field == 'DR' else cbar_limits_rhohv
            display.plot_rhi(field, sweep, vmin=vmin, vmax=vmax, cmap=cmap,
                           title=title, mask_outside=mask_outside, 
                           colorbar_label=units, norm=norm, ticks=ticks)
        else:        
            display.plot_rhi(field, sweep, vmin=vmin, vmax=vmax, cmap=cmap,
                           title=title, mask_outside=mask_outside, colorbar_label=units)
                           
        display.set_limits(xlim, ylim, ax=ax)
        display.plot_grid_lines(col='white')
        
        if add_logos:  
            annotate_plot_rhi_optimized(ax, fig, num_fields, layout)
        
        adjust_special_colorbars(field, display, index)

    if num_fields >= 2:
        plt.suptitle(mytitle, fontsize=28, weight='bold')
    
    save_plot(png, outdir, site, year, month, day, hh, mm, ss, string_csweep, 
             fields, num_fields, 'RHI', fig, azi)
    plt.close('all')

# ****************************************************************************************

def create_figure_rhi(layout):
    """Create RHI figure"""
    if layout['ncols'] < 2:
        return plt.figure(figsize=[layout['width'], layout['height']], constrained_layout=False)
    else:
        return plt.figure(figsize=[layout['width'], layout['height']], constrained_layout=True)

def create_gridspec_rhi(layout, fig):
    """Create RHI grid specification"""
    return plt.GridSpec(ncols=layout['ncols'], nrows=layout['nrows'], figure=fig)

def add_range_rings_fast(display, max_range):
    """Add range rings efficiently"""
    for rng in range(50, max_range + 50, 50):
        display.plot_range_ring(rng, col='white', ls='-', lw=0.5)

# ****************************************************************************************

def add_rings_radials_optimized(year, site, display, radar_lat, radar_lon, max_range, 
                              ax, add_logos, fig, num_fields, layout, 
                              COUNTIES, STATES, REEFS, MINOR_ISLANDS, OCEAN, LAKES,
                              add_heavy_features=True):  # New parameter
    """Optimized rings and radials with cached coordinates and features"""
    
    coord_data = _cache.get_coordinate_transform(radar_lat, radar_lon, max_range)
    
    display.plot_cross_hair(10, npts=100)
    
    # Add range rings efficiently
    if site in ['KaD3R', 'KuD3R']:
        rings = range(20, max_range + 20, 20)
    else:
        rings = range(50, max_range + 50, 50)
    
    for rng in rings:
        display.plot_range_ring(rng, line_style='--', lw=0.5, color='white')

    # Add radials efficiently using vectorized operations
    add_radials_vectorized(display, radar_lat, radar_lon, max_range, coord_data)
    
    # Add special location markers
    add_location_markers(site, year, display)
    
    # ONLY add heavy features (ocean/lakes) to first subplot
    if add_heavy_features:
        if OCEAN:
            ax.add_feature(OCEAN, facecolor="#414141")
        if LAKES:
            ax.add_feature(LAKES, facecolor="#414141", edgecolor='white', lw=0.25, zorder=0)
    
    # Add lighter features to ALL subplots
    if COUNTIES and STATES:
        ax.add_feature(STATES, facecolor='none', edgecolor='white', lw=0.5)
        ax.add_feature(COUNTIES, facecolor='none', edgecolor='white', lw=0.25)

    # Add site-specific features with cached shapefiles
    add_site_specific_features(site, ax, REEFS, MINOR_ISLANDS)
    
    # Add cartopy grid lines
    add_grid_lines_optimized(ax)

# ****************************************************************************************
def add_rings_radials_optimized_gvstyle(year, site, display, radar_lat, radar_lon, max_range, 
                                        ax, add_logos, fig, num_fields, layout, 
                                        COUNTIES, STATES, REEFS, MINOR_ISLANDS):
    """Add features like GVview - create OCEAN/LAKES fresh"""
    
    coord_data = _cache.get_coordinate_transform(radar_lat, radar_lon, max_range)
    display.plot_cross_hair(10, npts=100)
    
    # Range rings
    rings = range(20, max_range + 20, 20) if site in ['KaD3R', 'KuD3R'] else range(50, max_range + 50, 50)
    for rng in rings:
        display.plot_range_ring(rng, line_style='--', lw=0.5, color='white')
    
    add_radials_vectorized(display, radar_lat, radar_lon, max_range, coord_data)
    add_location_markers(site, year, display)
    
    # *** CREATE OCEAN/LAKES FRESH - DON'T USE CACHED VERSIONS ***
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor="#414141", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor="#414141", edgecolor='white', lw=0.25, zorder=0)
    
    # Political boundaries
    ax.add_feature(cfeature.COASTLINE, color='white', linewidth=0.5, zorder=10)
    ax.add_feature(cfeature.BORDERS, color='white', linewidth=0.5, zorder=10)
    
    if STATES:
        ax.add_feature(STATES, facecolor='none', edgecolor='white', lw=0.5, zorder=10)
    if COUNTIES:
        ax.add_feature(COUNTIES, facecolor='none', edgecolor='white', lw=0.25, zorder=10)
    
    add_site_specific_features(site, ax, REEFS, MINOR_ISLANDS)
    add_grid_lines_optimized(ax)

# ****************************************************************************************

def add_radials_vectorized(display, radar_lat, radar_lon, max_range, coord_data):
    """Add radials using vectorized operations"""
    azimuths = np.arange(0, 360, 30)
    dtor = math.pi / 180.0
    maxrange_meters = max_range * 1000.
    
    for azi in azimuths:
        azimuth = 90. - azi
        dazimuth = azimuth * dtor
        lon_maxrange = radar_lon + math.cos(dazimuth) * coord_data['meters_to_lon'] * maxrange_meters
        lat_maxrange = radar_lat + math.sin(dazimuth) * coord_data['meters_to_lat'] * maxrange_meters
        display.plot_line_geo([radar_lon, lon_maxrange], [radar_lat, lat_maxrange],
                            line_style='--', lw=0.5, color='white')

# ****************************************************************************************

def add_location_markers(site, year, display):
    """Add location markers efficiently"""
    # NASA WFF instrument pad locations (pre-calculated constants)
    markers = {
        'Pad': (-75.471, 37.934),
        'PCMK': (-75.515, 38.078),
        'IMPACTS_2022': (-72.29597, 41.80966),
        'IMPACTS_D3R': (-72.25770, 41.81795),
        'IMPACTS_2024': (-72.29597, 41.80966),
        'KTBW': (-82.406635, 27.115769)
    }
    
    wff_list = ['KDOX', 'KAKQ', 'NPOL', 'KuD3R', 'KaD3R']
    impacts_list = ['KOKX', 'KBOX']
    
    if site in wff_list:
        for marker_name in ['Pad', 'PCMK']:
            lon, lat = markers[marker_name]
            display.plot_point(lon, lat, symbol='v', markersize=5, color='white')
    
    if site in impacts_list:
        year_int = int(year)
        if year_int in [2021, 2022, 2023]: 
            lon, lat = markers['IMPACTS_2022']
            display.plot_point(lon, lat, symbol='v', markersize=3, color='white')
        if year_int in [2022, 2023, 2024, 2025]:
            lon, lat = markers['IMPACTS_D3R']
            display.plot_point(lon, lat, symbol='v', markersize=3, color='white')
            if year_int in [2024, 2025]:
                lon, lat = markers['IMPACTS_2024']
                display.plot_point(lon, lat, symbol='v', markersize=3, color='white')
    
    if site == 'KTBW':
        lon, lat = markers['KTBW']
        display.plot_point(lon, lat, symbol='v', markersize=5, color='white')

# ****************************************************************************************

def add_site_specific_features(site, ax, REEFS=None, MINOR_ISLANDS=None):
    """Add site-specific features efficiently using cached shapefiles"""
    Brazil_list = ['AL1', 'JG1', 'MC1', 'NT1', 'PE1', 'SF1', 'ST1', 'SV1', 'TM1']
    
    if site == 'KWAJ': 
        if REEFS:
            ax.add_feature(REEFS, facecolor='none', edgecolor='white', lw=0.25)
    
    elif site == 'RODN': 
        if MINOR_ISLANDS:
            ax.add_feature(MINOR_ISLANDS, facecolor='none', edgecolor='white', lw=0.25)
        ax.coastlines(edgecolor='white', lw=0.25)
    
    elif site in Brazil_list:
        ax.coastlines(edgecolor='white', lw=0.5)

# ****************************************************************************************

def add_grid_lines_optimized(ax):
    """Add cartopy grid lines efficiently"""
    grid_lines = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', x_inline=False)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    grid_lines.xformatter = LONGITUDE_FORMATTER
    grid_lines.yformatter = LATITUDE_FORMATTER
    grid_lines.xlabel_style = {'size': 6, 'color': 'black', 'rotation': 0, 
                              'weight': 'bold', 'ha': 'center'}
    grid_lines.ylabel_style = {'size': 6, 'color': 'black', 'rotation': 90, 
                              'weight': 'bold', 'va': 'bottom', 'ha': 'center'}

# ****************************************************************************************

def add_logo_ppi_optimized(ax, add_logos, fig, num_fields, layout):
    """Optimized logo addition with caching - adjusted for GridSpec top=0.92"""
    if not add_logos:
        return
        
    nasalogo = _cache.get_logo('nasa')
    gpmlogo = _cache.get_logo('gpm')
    
    if num_fields < 2:
        imageboxnasa = OffsetImage(nasalogo, zoom=0.06)
        imageboxgpm = OffsetImage(gpmlogo, zoom=0.03)
        imageboxnasa.image.axes = ax
        imageboxgpm.image.axes = ax 

        abnasa = AnnotationBbox(imageboxnasa, [0,0], xybox=[.095, .93],
                                xycoords='axes pixels', boxcoords='axes fraction',
                                pad=0.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm, [0,0], xybox=[.89, .93],                               
                               xycoords='axes pixels', boxcoords='axes fraction',
                               pad=0.0, frameon=False)
        ax.add_artist(abnasa)
        ax.add_artist(abgpm)
    else:
        ncols = layout['ncols']
        
        # Adjusted y positions for GridSpec top=0.92 (logos go in 0.92-1.0 space)
        if ncols == 2:
            imageboxnasa = OffsetImage(nasalogo, zoom=0.035*ncols)
            imageboxgpm = OffsetImage(gpmlogo, zoom=0.018*ncols)
            imageboxnasa.image.axes = fig
            imageboxgpm.image.axes = fig
            abnasa = AnnotationBbox(imageboxnasa, [0,0], xybox=[0.045, 0.985],  # Changed from 1.1
                                    xycoords='figure pixels', boxcoords='figure fraction',
                                    pad=0.0, frameon=False)
            abgpm = AnnotationBbox(imageboxgpm, [0,0], xybox=[1-.08, 0.985],  # Changed from 1.1
                                   xycoords='figure pixels', boxcoords='figure fraction',
                                   pad=0.0, frameon=False)
        elif ncols == 3:
            imageboxnasa = OffsetImage(nasalogo, zoom=0.035*ncols)
            imageboxgpm = OffsetImage(gpmlogo, zoom=0.0185*ncols)
            imageboxnasa.image.axes = fig
            imageboxgpm.image.axes = fig
            abnasa = AnnotationBbox(imageboxnasa, [0,0], xybox=[0.045, 0.975],  # Changed from 1.061
                                    xycoords='figure pixels', boxcoords='figure fraction',
                                    pad=0.0, frameon=False)
            abgpm = AnnotationBbox(imageboxgpm, [0,0], xybox=[1-.055, 0.975],  # Changed from 1.061
                                   xycoords='figure pixels', boxcoords='figure fraction',
                                   pad=0.0, frameon=False)
        elif ncols == 4:
            imageboxnasa = OffsetImage(nasalogo, zoom=0.030*ncols)
            imageboxgpm = OffsetImage(gpmlogo, zoom=0.015*ncols)
            imageboxnasa.image.axes = fig
            imageboxgpm.image.axes = fig
            abnasa = AnnotationBbox(imageboxnasa, [0,0], xybox=[0.03, 0.975],  # Changed from 1.066
                                    xycoords='figure pixels', boxcoords='figure fraction',
                                    pad=0.0, frameon=False)
            abgpm = AnnotationBbox(imageboxgpm, [0,0], xybox=[1-.08, 0.970],  # Changed from 1.061
                                   xycoords='figure pixels', boxcoords='figure fraction',
                                   pad=0.0, frameon=False)
        else:
            # Default for other column counts
            imageboxnasa = OffsetImage(nasalogo, zoom=0.035*ncols)
            imageboxgpm = OffsetImage(gpmlogo, zoom=0.018*ncols)
            imageboxnasa.image.axes = fig
            imageboxgpm.image.axes = fig
            abnasa = AnnotationBbox(imageboxnasa, [0,0], xybox=[0.045, 0.98],
                                    xycoords='figure pixels', boxcoords='figure fraction',
                                    pad=0.0, frameon=False)
            abgpm = AnnotationBbox(imageboxgpm, [0,0], xybox=[1-.055, 0.98],
                                   xycoords='figure pixels', boxcoords='figure fraction',
                                   pad=0.0, frameon=False)
        
        fig.add_artist(abnasa)
        fig.add_artist(abgpm)

# ****************************************************************************************

def annotate_plot_rhi_optimized(ax, fig, num_fields, layout):
    """Optimized RHI annotation with caching"""
    nasalogo = _cache.get_logo('nasa')
    gpmlogo = _cache.get_logo('gpm')
    
    if num_fields < 2:
        imageboxnasa = OffsetImage(nasalogo, zoom=0.07)
        imageboxgpm = OffsetImage(gpmlogo, zoom=0.03)
        imageboxnasa.image.axes = ax
        imageboxgpm.image.axes = ax
        
        abnasa = AnnotationBbox(imageboxnasa, [0,0], xybox=[.065, .915],
                                xycoords='axes pixels', boxcoords='axes fraction',
                                pad=-10.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm, [0,0], xybox=[.93, .925],                               
                               xycoords='axes pixels', boxcoords='axes fraction',
                               pad=0.0, frameon=False)
        ax.add_artist(abnasa)
        ax.add_artist(abgpm)
    else:
        nrows = layout['nrows']
        imageboxnasa = OffsetImage(nasalogo, zoom=0.12)
        imageboxgpm = OffsetImage(gpmlogo, zoom=0.06)
        imageboxnasa.image.axes = fig
        imageboxgpm.image.axes = fig
        
        abnasa = AnnotationBbox(imageboxnasa, [0,0], xybox=[140, 245*nrows],
                                xycoords='figure points', boxcoords='figure points',
                                pad=0.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm, [0,0], xybox=[1550, 245*nrows],
                               xycoords='figure points', boxcoords='figure points',
                               pad=0.0, frameon=False)
        fig.add_artist(abnasa)
        fig.add_artist(abgpm)

# ****************************************************************************************

def adjust_special_colorbars(field, display, index):
    """Adjust special colorbars efficiently"""
    colorbar_adjustments = {
        'FH': adjust_fhc_colorbar_for_pyart,
        'FH2': adjust_fhc_colorbar_for_pyart,
        'MRC': adjust_meth_colorbar_for_pyart,
        'MRC2': adjust_meth_colorbar_for_pyart,
        'FS': adjust_fhc_colorbar_for_pyart,
        'FW': adjust_fhw_colorbar_for_pyart,
        'NT': adjust_fhw_colorbar_for_pyart,
        'EC': adjust_ec_colorbar_for_pyart
    }
    
    if field in colorbar_adjustments:
        display.cbs[index] = colorbar_adjustments[field](display.cbs[index])

# ****************************************************************************************

def save_plot(png, outdir, site, year, month, day, hh, mm, ss, string_csweep, 
             fields, num_fields, plot_type, fig, azi=None):
    """Optimized plot saving - force rendering"""
    if not png:
        plt.show()
        return
        
    save_start = time.time()
    
    if outdir == '':
        outdir = os.getcwd()
    
    dpi = 150
    
    if num_fields == 1:
        field = fields[0]
        if plot_type == 'RHI' and azi is not None:
            png_file = f'{site}_{year}_{month+day}_{hh+mm+ss}_{field}_{azi:2.1f}AZ_RHI.png'
        else:
            png_file = f'{site}_{year}_{month+day}_{hh+mm+ss}_{field}_sw{string_csweep}_{plot_type}.png'
        
        outdir_daily = outdir
        os.makedirs(outdir_daily, exist_ok=True)
        filepath = os.path.join(outdir_daily, png_file)
        
    else:
        if plot_type == 'RHI' and azi is not None:
            png_file = f'{site}_{year}_{month+day}_{hh+mm+ss}_{num_fields}panel_{azi:2.1f}AZ_RHI.png'
        else:
            png_file = f'{site}_{year}_{month+day}_{hh+mm+ss}_{num_fields}panel_sw{string_csweep}_{plot_type}.png'
        
        outdir_multi = os.path.join(outdir, 'Multi')
        os.makedirs(outdir_multi, exist_ok=True)
        filepath = os.path.join(outdir_multi, png_file)

    # Now save
    print(f"    [save] Starting fig.savefig()...")
    t0 = time.time()
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"    [save] fig.savefig() took: {time.time()-t0:.2f}s")
    
    print(f'  --> {filepath}')

# ****************************************************************************************

def set_plot_size_parms_rhi(num_fields):
    """Optimized RHI plot size parameters"""
    if num_fields < 2:
        font_config = {
            'font.size': 8,
            'axes.titlesize': 10,
            'axes.labelsize': 8,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 6,
            'figure.titlesize': 10
        }
    else:
        font_config = {
            'font.size': 12,
            'axes.titlesize': 20,
            'axes.labelsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 20
        }
    
    # Apply all at once
    plt.rcParams.update(font_config)

# ****************************************************************************************

def set_plot_size_parms_ppi(num_fields):
    """Optimized PPI plot size parameters"""
    if num_fields < 2:
        font_config = {
            'font.size': 8,
            'axes.titlesize': 10,
            'axes.labelsize': 8,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 6,
            'figure.titlesize': 10
        }
    else:
        font_config = {
            'font.size': 8,
            'axes.titlesize': 14,
            'axes.labelsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 14
        }
    
    # Apply all at once
    plt.rcParams.update(font_config)

# ****************************************************************************************

def get_field_info(radar, field):
    """Optimized field info retrieval with pre-computed values"""
    
    # Pre-computed colormap configurations
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
                  'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    hid_colors_summer = ['White','LightBlue','MediumBlue','Darkorange','LightPink',
                        'Cyan','DarkGray', 'Lime','Yellow','Red','Fuchsia']
    hid_colors_winter = ['White','Orange', 'Purple', 'Fuchsia', 'Pink', 'Cyan',
                        'LightBlue', 'Blue']
    ec_hid_colors = ['White','LightPink','Darkorange','LightBlue','Lime','MediumBlue','DarkGray',
                     'Cyan','Red','Yellow']                     
    
    cmaphidw = colors.ListedColormap(hid_colors_winter)
    cmaphid = colors.ListedColormap(hid_colors_summer)
    cmaphidec = colors.ListedColormap(ec_hid_colors)
    cmapmeth = colors.ListedColormap(hid_colors[0:6])

    # Field configuration dictionary for faster lookup
    field_configs = {
        'CZ': {'units': 'Zh [dBZ]', 'vmin': 0, 'vmax': 70, 'Nbins': 14, 
               'title': 'Corrected Reflectivity [dBZ]', 'cmap': check_cm('NWSRef')},
        'DZ': {'units': 'Zh [dBZ]', 'vmin': 0, 'vmax': 70, 'Nbins': 14,
               'title': 'RAW Reflectivity [dBZ]', 'cmap': check_cm('NWSRef')},
        'DR': {'units': 'Zdr [dB]', 'vmin': -2, 'vmax': 5, 'Nbins': 0,
               'title': 'Differential Reflectivity [dB]', 'cmap': zdr_cmap, 'norm': zdr_norm},
        'RH': {'units': 'ρHV', 'vmin': 0.0, 'vmax': 1.0, 'Nbins': 0,
               'title': 'Cross-Correlation Coefficient', 'cmap': rhohv_cmap, 'norm': rhohv_norm},
        'VR': {'units': 'Velocity [m/s]', 'vmin': -20, 'vmax': 20, 'Nbins': 12,
               'title': 'Radial Velocity [m/s]', 'cmap': check_cm('NWSVel')},
        'SW': {'units': 'Spectrum Width', 'vmin': 0, 'vmax': 21, 'Nbins': 12,
               'title': 'Spectrum Width', 'cmap': check_cm('NWS_SPW')},
        'corrected_velocity': {'units': 'Velocity [m/s]', 'vmin': -20, 'vmax': 20, 'Nbins': 12,
                              'title': 'Dealiased Radial Velocity [m/s]', 'cmap': check_cm('NWSVel')},
        'KD': {'units': 'Kdp [deg/km]', 'vmin': -2, 'vmax': 3, 'Nbins': 10,
               'title': 'Specific Differential Phase [deg/km]', 'cmap': check_cm('HomeyerRainbow')},
        'KDPB': {'units': 'Kdp [deg/km]', 'vmin': -2, 'vmax': 5, 'Nbins': 8,
                 'title': 'Specific Differential Phase [deg/km] (Bringi)', 'cmap': check_cm('HomeyerRainbow')},
        'PH': {'units': 'PhiDP [deg]', 'vmin': 0, 'vmax': 360, 'Nbins': 36,
               'title': 'Differential Phase [deg]', 'cmap': check_cm('Carbone42')},
        'PHM': {'units': 'PhiDP [deg]', 'vmin': 0, 'vmax': 360, 'Nbins': 36,
                'title': 'Differential Phase [deg] Marks', 'cmap': check_cm('Carbone42')},
        'PHIDPB': {'units': 'PhiDP [deg]', 'vmin': 0, 'vmax': 360, 'Nbins': 36,
                   'title': 'Differential Phase [deg] Bringi', 'cmap': check_cm('Carbone42')},
        'SD': {'units': 'Std(PhiDP)', 'vmin': 0, 'vmax': 70, 'Nbins': 14,
               'title': 'Standard Deviation of PhiDP', 'cmap': check_cm('NWSRef')},
        'SQ': {'units': 'SQI', 'vmin': 0, 'vmax': 1, 'Nbins': 10,
               'title': 'Signal Quality Index', 'cmap': check_cm('LangRainbow12')},
        'FH': {'units': 'HID', 'vmin': 0, 'vmax': 11, 'Nbins': 0,
               'title': 'Summer Hydrometeor Identification', 'cmap': cmaphid},
        'FH2': {'units': 'HID', 'vmin': 0, 'vmax': 11, 'Nbins': 0,
               'title': 'Summer Hydrometeor Identification', 'cmap': cmaphid},
        'FS': {'units': 'HID', 'vmin': 0, 'vmax': 11, 'Nbins': 0,
               'title': 'Summer Hydrometeor Identification', 'cmap': cmaphid},
        'FW': {'units': 'HID', 'vmin': 0, 'vmax': 8, 'Nbins': 0,
               'title': 'Winter Hydrometeor Identification', 'cmap': cmaphidw},
        'NT': {'units': 'HID', 'vmin': 0, 'vmax': 8, 'Nbins': 0,
               'title': 'No TEMP Winter Hydrometeor Identification', 'cmap': cmaphidw},
        'EC': {'units': 'HID', 'vmin': 0, 'vmax': 9, 'Nbins': 0,
               'title': 'Radar Echo Classification', 'cmap': cmaphidec},
        'MW': {'units': 'Water Mass [g/m^3]', 'vmin': 0, 'vmax': 3, 'Nbins': 25,
               'title': 'Water Mass [g/m^3]', 'cmap': 'turbo'},
        'MI': {'units': 'Ice Mass [g/m^3]', 'vmin': 0, 'vmax': 3, 'Nbins': 25,
               'title': 'Ice Mass [g/m^3]', 'cmap': 'turbo'},
        'RC': {'units': 'HIDRO Rain Rate [mm/hr]', 'vmin': 1e-2, 'vmax': 3e2, 'Nbins': 0,
               'title': 'HIDRO Rain Rate [mm/hr]', 'cmap': check_cm('RefDiff')},
        'RP': {'units': 'PolZR Rain Rate [mm/hr]', 'vmin': 1e-2, 'vmax': 3e2, 'Nbins': 0,
               'title': 'PolZR Rain Rate [mm/hr]', 'cmap': check_cm('RefDiff')},
        'RA': {'units': 'Attenuation Rain Rate [mm/hr]', 'vmin': 1e-2, 'vmax': 3e2, 'Nbins': 0,
               'title': 'Attenuation Rain Rate [mm/hr]', 'cmap': check_cm('RefDiff')},
        'MRC': {'units': 'HIDRO Method', 'vmin': 0, 'vmax': 5, 'Nbins': 0,
                'title': 'HIDRO Method', 'cmap': cmapmeth},
        'MRC2': {'units': 'HIDRO Method', 'vmin': 0, 'vmax': 5, 'Nbins': 0,
                'title': 'HIDRO Method', 'cmap': cmapmeth},
        'DM': {'units': 'DM [mm]', 'vmin': 0.5, 'vmax': 5, 'Nbins': 8,
               'title': 'DM [mm]', 'cmap': check_cm('BlueBrown10')},
        'NW': {'units': 'Log[Nw, m^-3 mm^-1]', 'vmin': 0.5, 'vmax': 7, 'Nbins': 12,
               'title': 'Log[Nw, m^-3 mm^-1]', 'cmap': check_cm('BlueBrown10')}
    }

    # Get configuration or return defaults
    config = field_configs.get(field, {
        'units': 'Unknown', 'vmin': 0, 'vmax': 100, 'Nbins': 10,
        'title': f'Unknown Field: {field}', 'cmap': 'viridis', 'norm': None
    })
    
    # Return
    return (config['units'], config['vmin'], config['vmax'], config['cmap'], 
            config['title'], config['Nbins'], config.get('norm', None))

# ****************************************************************************************

def get_radar_info(radar, sweep):
    """Optimized radar info extraction with better site name handling"""
    
    # Extract site name with priority handling
    site = ''
    if 'site_name' in radar.metadata:
        site = radar.metadata['site_name']
    elif 'instrument_name' in radar.metadata:
        site = radar.metadata['instrument_name']
    
    # Handle bytes conversion
    if isinstance(site, bytes):
        site = site.decode().upper()
    else:
        site = str(site).upper()

    # Handle ODIM format
    if radar.metadata.get('original_container') == 'odim_h5':
        try:
            site = radar.metadata['source'].replace(',', ':').split(':')[1].upper()
        except:
            site = radar.metadata.get('site_name', '').upper()

    # Site name mapping for faster lookup
    site_mappings = {
        'NPOL1': 'NPOL', 'NPOL2': 'NPOL', 'LAVA1': 'KWAJ',
        'AN1-P': 'AL1', 'JG1-P': 'JG1', 'MC1-P': 'MC1', 'NT1-P': 'NT1',
        'PE1-P': 'PE1', 'SF1-P': 'SF1', 'ST1-P': 'ST1', 'SV1-P': 'SV1',
        'TM1-P': 'TM1', 'GUNN_PT': 'CPOL', 'REUNION': 'Reunion', 'CP2RADAR': 'CP2'
    }
    
    # Clean up byte strings in site names
    site = site.replace('\x00', '').strip()
    site = site_mappings.get(site, site)
    
    # Handle special system metadata
    if 'system' in radar.metadata:
        system_sites = {'KuD3R': 'KuD3R', 'KaD3R': 'KaD3R'}
        site = system_sites.get(radar.metadata['system'], site)

    # Get radar datetime efficiently
    radar_DT = pyart.util.datetime_from_radar(radar)
    
    # Handle special time formats for certain radars
    if radar_DT.year > 2000 and site in ['NPOL', 'KWAJ']:
        EPOCH_UNITS = "seconds since 1970-01-01T00:00:00Z"
        dtrad = num2date(0, radar.time["units"])
        epnum = date2num(dtrad, EPOCH_UNITS)
        radar_DT = num2date(epnum, EPOCH_UNITS)

    elv = radar.fixed_angle['data'][sweep]
    string_csweep = str(sweep).zfill(2)
    
    # Format date/time strings efficiently
    year = f'{radar_DT.year:04d}'
    month = f'{radar_DT.month:02d}'
    day = f'{radar_DT.day:02d}'
    hh = f'{radar_DT.hour:02d}'
    mm = f'{radar_DT.minute:02d}'
    ss = f'{radar_DT.second:02d}'
    
    mydate = f'{month}/{day}/{year}'
    mytime = f'{hh}:{mm}:{ss}'

    return site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep

# ****************************************************************************************

# Colorbar adjustment functions - optimized for repeated use
def adjust_fhc_colorbar_for_pyart(cb):
    cb.set_ticks(np.arange(0.5, 11, 1.0))
    cb.ax.set_yticklabels(['No Echo', 'Drizzle', 'Rain', 'Ice Crystals', 
                          'Aggregates', 'Wet Snow', 'Vertical Ice', 
                          'LD Graupel', 'HD Graupel', 'Hail', 'Big Drops'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

def adjust_fhw_colorbar_for_pyart(cb):
    cb.set_ticks(np.arange(0.5, 8, 1.0))
    cb.ax.set_yticklabels(['No Echo','Ice Crystals', 'Plates', 'Dendrites', 
                          'Aggregates', 'Wet Snow','Frozen Precip', 'Rain'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

def adjust_ec_colorbar_for_pyart(cb):
    cb.set_ticks(np.arange(1.4, 9, 0.9))
    cb.ax.set_yticklabels(["Aggregates", "Ice Crystals", "Light Rain", 
                          "Rimed Particles", "Rain", "Vertically Ice", 
                          "Wet Snow", "Melting Hail", "Dry Hail/High Density Graupel"])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

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

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N, 0))
    cmap_name = base.name + str(N)
    return plt.cm.colors.ListedColormap(color_list, cmap_name, N)

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                           left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)

def check_cm(cmap_name):
    """Handles old and new versions of colormaps"""
    candidates = [cmap_name, f'pyart_{cmap_name}']
    for name in candidates:
        if name in plt.colormaps():
            return name
    return candidates[1]