import os
import sys
import math
import numpy as np
import matplotlib
import pyart
os.environ['PYART_QUIET'] = '1'  # Suppress PyART citation

# Qt context (batch use)
from PyQt5.QtWidgets import QApplication
_app = QApplication.instance() or QApplication(sys.argv)

# Matplotlib backend setup
matplotlib.use("Qt5Agg")
matplotlib.rcParams["backend"] = "Qt5Agg"
matplotlib.rcParams["agg.path.chunksize"] = 0

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from PIL import Image
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from cftime import date2num, num2date
import time


# ======================================================================================
# Colormaps / Norms
# ======================================================================================

scale_rhohv = [
    (1.0, 1.0, 1.0),
    (0.7969, 0.9961, 0.9961),
    (0.0156, 0.9102, 0.9023),
    (0.0039, 0.6211, 0.9531),
    (0.0117, 0, 0.9531),
    (0.0078, 0.9883, 0.0078),
    (0.0039, 0.7695, 0.0039),
    (0, 0.5547, 0),
    (0.9883, 0.9688, 0.0078),
    (0.8945, 0.7344, 0),
    (0.9883, 0.582, 0),
    (0.9883, 0, 0),
    (0.8281, 0, 0),
    (0.7344, 0, 0),
    (0.6, 0, 0),
    (0.4, 0, 0),
    (0.2, 0, 0),
]
cbar_limits_rhohv = [
    0.0, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94,
    0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.994, 0.997, 1.00
]
rhohv_cmap = colors.LinearSegmentedColormap.from_list("rhohv_refined", scale_rhohv)
rhohv_norm = colors.BoundaryNorm(cbar_limits_rhohv, rhohv_cmap.N)

scale_zdr = [
    (0.5, 0, 0.5),
    (0.3, 0.3, 0.8),
    (0.2, 0.5, 0.9),
    (0.5, 0.7, 0.9),
    (0.8, 0.8, 0.8),
    (0.6, 0.9, 0.6),
    (0.2, 0.8, 0.2),
    (0.0, 0.6, 0.0),
    (0.9, 0.9, 0.0),
    (0.9, 0.7, 0.0),
    (0.9, 0.5, 0.0),
    (0.9, 0.2, 0.0),
    (0.8, 0.0, 0.0),
    (0.5, 0.0, 0.0),
]
cbar_limits_zdr = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
zdr_cmap = colors.LinearSegmentedColormap.from_list("zdr_refined", scale_zdr)
zdr_norm = colors.BoundaryNorm(cbar_limits_zdr, zdr_cmap.N)


# ======================================================================================
# Cache
# ======================================================================================

class PlottingCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._map_features_cache = None
        self._logos_cache = {}
        self._field_cache = {}
        self._coordinate_cache = {}
        self._initialized = True

    def get_map_features(self):
        if self._map_features_cache is None:
            self._map_features_cache = self._load_all_map_features()
        return self._map_features_cache

    def _load_all_map_features(self):
        base_dir = os.path.dirname(__file__)
        shapefile_dir = os.path.join(base_dir, "shape_files")

        # Counties
        COUNTIES = None
        county_shapefile = os.path.join(shapefile_dir, "countyl010g.shp")
        if os.path.exists(county_shapefile):
            try:
                reader = shpreader.Reader(county_shapefile)
                counties = list(reader.geometries())
                COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
            except Exception:
                COUNTIES = None

        # States
        STATES = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="10m",
            facecolor="none",
        )

        # Reefs
        REEFS = None
        reef_shapefile = os.path.join(shapefile_dir, "ne_10m_reefs.shp")
        if os.path.exists(reef_shapefile):
            try:
                reef_reader = shpreader.Reader(reef_shapefile)
                reefs = list(reef_reader.geometries())
                REEFS = cfeature.ShapelyFeature(reefs, ccrs.PlateCarree())
            except Exception:
                REEFS = None

        # Minor Islands
        MINOR_ISLANDS = None
        islands_shapefile = os.path.join(shapefile_dir, "ne_10m_minor_islands.shp")
        if os.path.exists(islands_shapefile):
            try:
                islands_reader = shpreader.Reader(islands_shapefile)
                islands = list(islands_reader.geometries())
                MINOR_ISLANDS = cfeature.ShapelyFeature(islands, ccrs.PlateCarree())
            except Exception:
                MINOR_ISLANDS = None

        return COUNTIES, STATES, REEFS, MINOR_ISLANDS

    def get_logo(self, logo_name):
        if logo_name not in self._logos_cache:
            logo_dir = os.path.dirname(__file__)
            logo_path = os.path.join(logo_dir, f"{logo_name}.png")
            self._logos_cache[logo_name] = Image.open(logo_path)
        return self._logos_cache[logo_name]

    def get_processed_field(self, radar, field_name):
        cache_key = f"{field_name}_{id(radar)}"
        if cache_key not in self._field_cache:
            self._field_cache[cache_key] = self._process_field(radar, field_name)
        return self._field_cache[cache_key]

    def _process_field(self, radar, field_name):
        if field_name == "RC":
            rc = radar.fields["RC"]["data"].copy()
            rc[rc < 0.01] = np.nan
            return {"data": rc, "units": "mm/h",
                    "long_name": "HIDRO Rainfall Rate", "_FillValue": -32767.0,
                    "standard_name": "HIDRO Rainfall Rate"}
        if field_name == "RP":
            rp = radar.fields["RP"]["data"].copy()
            rp[rp < 0.01] = np.nan
            return {"data": rp, "units": "mm/h",
                    "long_name": "Polzr_Rain_Rate", "_FillValue": -32767.0,
                    "standard_name": "Polzr_Rain_Rate"}
        if field_name == "RA":
            ra = radar.fields["RA"]["data"].copy()
            ra[ra < 0.01] = np.nan
            return {"data": ra, "units": "mm/h",
                    "long_name": "A_Rain_Rate", "_FillValue": -32767.0,
                    "standard_name": "A_Rain_Rate"}
        return None

    def get_coordinate_transform(self, radar_lat, radar_lon, max_range):
        cache_key = f"{round(float(radar_lat), 6)}_{round(float(radar_lon), 6)}_{int(max_range)}"
        if cache_key not in self._coordinate_cache:
            self._coordinate_cache[cache_key] = self._calculate_coordinates(radar_lat, radar_lon, max_range)
        return self._coordinate_cache[cache_key]

    def _calculate_coordinates(self, radar_lat, radar_lon, max_range):
        dtor = math.pi / 180.0
        maxrange_meters = max_range * 1000.0
        meters_to_lat = 1.0 / 111177.0
        meters_to_lon = 1.0 / (111177.0 * math.cos(radar_lat * dtor))

        min_lat = radar_lat - maxrange_meters * meters_to_lat
        max_lat = radar_lat + maxrange_meters * meters_to_lat
        min_lon = radar_lon - maxrange_meters * meters_to_lon
        max_lon = radar_lon + maxrange_meters * meters_to_lon

        lon_grid = np.arange(round(min_lon, 2) - 1.0, round(max_lon, 2) + 1.0, 1.0)
        lat_grid = np.arange(round(min_lat, 2) - 1.0, round(max_lat, 2) + 1.0, 1.0)

        return {
            "min_lat": min_lat, "max_lat": max_lat,
            "min_lon": min_lon, "max_lon": max_lon,
            "lon_grid": lon_grid, "lat_grid": lat_grid,
            "meters_to_lat": meters_to_lat, "meters_to_lon": meters_to_lon
        }


_cache = PlottingCache()

def configure_matplotlib():
    plt.ioff()
    plt.rcParams.update({
        "font.weight": "bold",
        "axes.facecolor": "black",
        "agg.path.chunksize": 0,
        "figure.max_open_warning": 0,
        "axes.formatter.useoffset": False,
    })

configure_matplotlib()

_plotting_warmed_up = False


# ======================================================================================
# Warmup
# ======================================================================================

def warmup_plotting_engine(radar, sweep, max_range):
    radar_lat = radar.latitude["data"][0]
    radar_lon = radar.longitude["data"][0]
    projection = ccrs.LambertConformal(radar_lon, radar_lat)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), subplot_kw={"projection": projection})
    ax.set_facecolor("black")

    display = pyart.graph.RadarMapDisplay(radar)
    field = list(radar.fields.keys())[0]

    try:
        coord_data = _cache.get_coordinate_transform(radar_lat, radar_lon, max_range)
        display.plot_ppi_map(
            field, sweep,
            resolution="50m",
            projection=projection,
            ax=ax,
            min_lon=coord_data["min_lon"], max_lon=coord_data["max_lon"],
            min_lat=coord_data["min_lat"], max_lat=coord_data["max_lat"],
            lon_lines=coord_data["lon_grid"], lat_lines=coord_data["lat_grid"],
            add_grid_lines=False,
            lat_0=radar_lat, lon_0=radar_lon,
            embellish=False,
        )
    except Exception:
        pass

    plt.close(fig)


# ======================================================================================
# Layout helpers
# ======================================================================================

def calculate_layout(num_fields):
    nrows = max(1, math.ceil(num_fields / 4))

    if num_fields <= 4:
        ncols = num_fields
        width = num_fields * 6
        height = float(nrows * 4.5)
    else:
        ncols = round(num_fields / 2)
        width = ncols * 6
        height = float(nrows * 4.5)

    positions = [(x, y) for x in range(nrows) for y in range(ncols)]
    return {"nrows": nrows, "ncols": ncols, "width": width, "height": height, "positions": positions}

def create_figure(layout):
    fig = plt.figure(figsize=[layout["width"], layout["height"]], constrained_layout=False)
    fig.set_tight_layout(False)
    return fig

def create_gridspec(layout, fig):
    if layout["ncols"] < 2:
        return plt.GridSpec(ncols=layout["ncols"], nrows=layout["nrows"], figure=fig)
    return plt.GridSpec(
        ncols=layout["ncols"], nrows=layout["nrows"], figure=fig,
        left=0.0, right=1.0, top=0.92, bottom=0.0,
        wspace=0.000000009, hspace=0.15
    )


# ======================================================================================
# Main plot dispatch
# ======================================================================================

def plot_fields(self):
    global _plotting_warmed_up

    start = time.time()

    if self.sweeps_to_plot == "all":
        sweepn = list(range(len(self.radar.sweep_number["data"][:])))
    else:
        sweepn = self.sweeps_to_plot

    if self.scan_type == 'RHI':
        print('Plotting RHI images...')
        if self.plot_multi:
            for sweep in sweepn:
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
                for sweep in sweepn:
                    plot_fields_RHI(self.radar, sweep=sweep, fields=[field],
                                   ymax=self.max_height, xmax=self.max_range,
                                   png=self.png, outdir=plot_dir,
                                   add_logos=self.add_logos,
                                   mask_outside=self.mask_outside)

    elif self.scan_type == 'PPI':
        print('Plotting PPI images...')

        if not self.plot_fast:
            COUNTIES, STATES, REEFS, MINOR_ISLANDS = _cache.get_map_features()
            if self.plot_multi and len(self.fields_to_plot) > 1 and not _plotting_warmed_up:
                warmup_plotting_engine(self.radar, sweepn[0], self.max_range)
                _plotting_warmed_up = True
        else:
            COUNTIES = STATES = REEFS = MINOR_ISLANDS = None

        if self.plot_multi:
            os.makedirs(self.plot_dir, exist_ok=True)
            for sweep in sweepn:
                if self.plot_fast:
                    plot_fields_PPI_QC(
                        self.radar, sweep=sweep, fields=self.fields_to_plot,
                        max_range=self.max_range, png=self.png, outdir=self.plot_dir,
                        add_logos=self.add_logos, mask_outside=self.mask_outside
                    )
                else:
                    plot_fields_PPI(
                        self.radar, COUNTIES, STATES, REEFS, MINOR_ISLANDS,
                        sweep=sweep, fields=self.fields_to_plot, max_range=self.max_range,
                        mask_outside=self.mask_outside, png=self.png, outdir=self.plot_dir,
                        add_logos=self.add_logos
                    )

        if self.plot_single:
            for field in self.fields_to_plot:
                print(field)
                plot_dir = os.path.join(self.plot_dir, field)
                os.makedirs(plot_dir, exist_ok=True)
                for sweep in sweepn:
                    if self.plot_fast:
                        plot_fields_PPI_QC(
                            self.radar, sweep=sweep, fields=[field],
                            max_range=self.max_range, png=self.png, outdir=plot_dir,
                            add_logos=self.add_logos, mask_outside=self.mask_outside
                        )
                    else:
                        plot_fields_PPI(
                            self.radar, COUNTIES, STATES, REEFS, MINOR_ISLANDS,
                            sweep=sweep, fields=[field], max_range=self.max_range,
                            mask_outside=self.mask_outside, png=self.png, outdir=plot_dir,
                            add_logos=self.add_logos
                        )

    end = time.time()
    print(f'plotting time: {end - start:.2f} seconds')


# ======================================================================================
# Map features (GVview style)
# ======================================================================================

def add_rings_radials_optimized_gvstyle(year, site, display, radar_lat, radar_lon, max_range,
                                        ax, COUNTIES, STATES, REEFS, MINOR_ISLANDS):
    coord_data = _cache.get_coordinate_transform(radar_lat, radar_lon, max_range)

    display.plot_cross_hair(10, npts=100)

    rings = range(20, max_range + 20, 20) if site in ("KaD3R", "KuD3R") else range(50, max_range + 50, 50)
    for rng in rings:
        display.plot_range_ring(rng, line_style="--", lw=0.5, color="white")

    add_radials_vectorized(display, radar_lat, radar_lon, max_range, coord_data)
    add_location_markers(site, year, display)

    # Fresh base layers each axis (this was part of the speedup vs cached feature objects)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#414141", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#414141", edgecolor="white", lw=0.25, zorder=0)

    ax.add_feature(cfeature.COASTLINE, color="white", linewidth=0.5, zorder=10)
    ax.add_feature(cfeature.BORDERS, color="white", linewidth=0.5, zorder=10)

    if STATES:
        ax.add_feature(STATES, facecolor="none", edgecolor="white", lw=0.5, zorder=10)
    if COUNTIES:
        ax.add_feature(COUNTIES, facecolor="none", edgecolor="white", lw=0.25, zorder=10)

    add_site_specific_features(site, ax, REEFS, MINOR_ISLANDS)
    add_grid_lines_optimized(ax)


def add_grid_lines_optimized(ax):
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", x_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 6, "color": "black", "rotation": 0, "weight": "bold", "ha": "center"}
    gl.ylabel_style = {"size": 6, "color": "black", "rotation": 90, "weight": "bold", "va": "bottom", "ha": "center"}


def add_radials_vectorized(display, radar_lat, radar_lon, max_range, coord_data):
    azimuths = np.arange(0, 360, 30)
    dtor = math.pi / 180.0
    maxrange_meters = max_range * 1000.0

    for azi in azimuths:
        azimuth = 90.0 - azi
        dazimuth = azimuth * dtor
        lon_max = radar_lon + math.cos(dazimuth) * coord_data["meters_to_lon"] * maxrange_meters
        lat_max = radar_lat + math.sin(dazimuth) * coord_data["meters_to_lat"] * maxrange_meters
        display.plot_line_geo([radar_lon, lon_max], [radar_lat, lat_max],
                              line_style="--", lw=0.5, color="white")


def add_location_markers(site, year, display):
    markers = {
        "Pad": (-75.471, 37.934),
        "PCMK": (-75.515, 38.078),
        "IMPACTS_2022": (-72.29597, 41.80966),
        "IMPACTS_D3R": (-72.25770, 41.81795),
        "IMPACTS_2024": (-72.29597, 41.80966),
        "KTBW": (-82.406635, 27.115769),
    }

    wff_list = ["KDOX", "KAKQ", "NPOL", "KuD3R", "KaD3R"]
    impacts_list = ["KOKX", "KBOX"]

    if site in wff_list:
        for name in ("Pad", "PCMK"):
            lon, lat = markers[name]
            display.plot_point(lon, lat, symbol="v", markersize=5, color="white")

    if site in impacts_list:
        y = int(year)
        if y in (2021, 2022, 2023):
            lon, lat = markers["IMPACTS_2022"]
            display.plot_point(lon, lat, symbol="v", markersize=3, color="white")
        if y in (2022, 2023, 2024, 2025):
            lon, lat = markers["IMPACTS_D3R"]
            display.plot_point(lon, lat, symbol="v", markersize=3, color="white")
            if y in (2024, 2025):
                lon, lat = markers["IMPACTS_2024"]
                display.plot_point(lon, lat, symbol="v", markersize=3, color="white")

    if site == "KTBW":
        lon, lat = markers["KTBW"]
        display.plot_point(lon, lat, symbol="v", markersize=5, color="white")


def add_site_specific_features(site, ax, REEFS=None, MINOR_ISLANDS=None):
    brazil = ["AL1", "JG1", "MC1", "NT1", "PE1", "SF1", "ST1", "SV1", "TM1"]
    if site == "KWAJ" and REEFS:
        ax.add_feature(REEFS, facecolor="none", edgecolor="white", lw=0.25)
    elif site == "RODN":
        if MINOR_ISLANDS:
            ax.add_feature(MINOR_ISLANDS, facecolor="none", edgecolor="white", lw=0.25)
        ax.coastlines(edgecolor="white", lw=0.25)
    elif site in brazil:
        ax.coastlines(edgecolor="white", lw=0.5)


# ======================================================================================
# Logos + Colorbars + Save
# ======================================================================================

def add_logo_ppi_optimized(ax, add_logos, fig, num_fields, layout):
    if not add_logos:
        return

    nasalogo = _cache.get_logo("nasa")
    gpmlogo = _cache.get_logo("gpm")

    if num_fields < 2:
        imageboxnasa = OffsetImage(nasalogo, zoom=0.06)
        imageboxgpm = OffsetImage(gpmlogo, zoom=0.03)
        imageboxnasa.image.axes = ax
        imageboxgpm.image.axes = ax

        abnasa = AnnotationBbox(imageboxnasa, [0, 0], xybox=[.095, .93],
                                xycoords="axes pixels", boxcoords="axes fraction",
                                pad=0.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm, [0, 0], xybox=[.89, .93],
                               xycoords="axes pixels", boxcoords="axes fraction",
                               pad=0.0, frameon=False)
        ax.add_artist(abnasa)
        ax.add_artist(abgpm)
        return

    ncols = layout["ncols"]

    if ncols == 2:
        imageboxnasa = OffsetImage(nasalogo, zoom=0.035 * ncols)
        imageboxgpm = OffsetImage(gpmlogo, zoom=0.018 * ncols)
        imageboxnasa.image.axes = fig
        imageboxgpm.image.axes = fig
        abnasa = AnnotationBbox(imageboxnasa, [0, 0], xybox=[0.045, 0.985],
                                xycoords="figure pixels", boxcoords="figure fraction",
                                pad=0.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm, [0, 0], xybox=[1 - .08, 0.985],
                               xycoords="figure pixels", boxcoords="figure fraction",
                               pad=0.0, frameon=False)
    elif ncols == 3:
        imageboxnasa = OffsetImage(nasalogo, zoom=0.035 * ncols)
        imageboxgpm = OffsetImage(gpmlogo, zoom=0.0185 * ncols)
        imageboxnasa.image.axes = fig
        imageboxgpm.image.axes = fig
        abnasa = AnnotationBbox(imageboxnasa, [0, 0], xybox=[0.045, 0.975],
                                xycoords="figure pixels", boxcoords="figure fraction",
                                pad=0.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm, [0, 0], xybox=[1 - .055, 0.975],
                               xycoords="figure pixels", boxcoords="figure fraction",
                               pad=0.0, frameon=False)
    elif ncols == 4:
        imageboxnasa = OffsetImage(nasalogo, zoom=0.030 * ncols)
        imageboxgpm = OffsetImage(gpmlogo, zoom=0.015 * ncols)
        imageboxnasa.image.axes = fig
        imageboxgpm.image.axes = fig
        abnasa = AnnotationBbox(imageboxnasa, [0, 0], xybox=[0.03, 0.975],
                                xycoords="figure pixels", boxcoords="figure fraction",
                                pad=0.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm, [0, 0], xybox=[1 - .08, 0.970],
                               xycoords="figure pixels", boxcoords="figure fraction",
                               pad=0.0, frameon=False)
    else:
        imageboxnasa = OffsetImage(nasalogo, zoom=0.035 * ncols)
        imageboxgpm = OffsetImage(gpmlogo, zoom=0.018 * ncols)
        imageboxnasa.image.axes = fig
        imageboxgpm.image.axes = fig
        abnasa = AnnotationBbox(imageboxnasa, [0, 0], xybox=[0.045, 0.98],
                                xycoords="figure pixels", boxcoords="figure fraction",
                                pad=0.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm, [0, 0], xybox=[1 - .055, 0.98],
                               xycoords="figure pixels", boxcoords="figure fraction",
                               pad=0.0, frameon=False)

    fig.add_artist(abnasa)
    fig.add_artist(abgpm)


def adjust_special_colorbars(field, display, index):
    adjustments = {
        "FH": adjust_fhc_colorbar_for_pyart,
        "FH2": adjust_fhc_colorbar_for_pyart,
        "MRC": adjust_meth_colorbar_for_pyart,
        "MRC2": adjust_meth_colorbar_for_pyart,
        "FS": adjust_fhc_colorbar_for_pyart,
        "FW": adjust_fhw_colorbar_for_pyart,
        "NT": adjust_fhw_colorbar_for_pyart,
        "EC": adjust_ec_colorbar_for_pyart,
    }
    if field in adjustments and hasattr(display, "cbs") and len(display.cbs) > index:
        display.cbs[index] = adjustments[field](display.cbs[index])


def save_plot(png, outdir, site, year, month, day, hh, mm, ss, string_csweep,
             fields, num_fields, plot_type, fig, azi=None):
    if not png:
        plt.show()
        return

    if outdir == "":
        outdir = os.getcwd()

    dpi = 150

    if num_fields == 1:
        field = fields[0]
        if plot_type == "RHI" and azi is not None:
            png_file = f"{site}_{year}_{month+day}_{hh+mm+ss}_{field}_{azi:2.1f}AZ_RHI.png"
        else:
            png_file = f"{site}_{year}_{month+day}_{hh+mm+ss}_{field}_sw{string_csweep}_{plot_type}.png"
        os.makedirs(outdir, exist_ok=True)
        filepath = os.path.join(outdir, png_file)
    else:
        if plot_type == "RHI" and azi is not None:
            png_file = f"{site}_{year}_{month+day}_{hh+mm+ss}_{num_fields}panel_{azi:2.1f}AZ_RHI.png"
        else:
            png_file = f"{site}_{year}_{month+day}_{hh+mm+ss}_{num_fields}panel_sw{string_csweep}_{plot_type}.png"
        os.makedirs(outdir, exist_ok=True)
        filepath = os.path.join(outdir, png_file)

    fig.savefig(filepath, dpi=dpi,bbox_inches='tight')
    print(f"  --> {filepath}")


# ======================================================================================
# Plot sizing / Field info / Radar info / misc
# ======================================================================================

def set_plot_size_parms_ppi(num_fields):
    if num_fields < 2:
        font_config = {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.titlesize": 10,
        }
    else:
        font_config = {
            "font.size": 8,
            "axes.titlesize": 14,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 14,
        }
    plt.rcParams.update(font_config)

def get_field_info(radar, field):
    """Return plotting metadata for a given field.

    Returns:
        units, vmin, vmax, cmap, title, Nbins, norm
    """
    # HID colormaps
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
                  'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    hid_colors_summer = ['White','LightBlue','MediumBlue','Darkorange','LightPink',
                         'Cyan','DarkGray', 'Lime','Yellow','Red','Fuchsia']
    hid_colors_winter = ['White','Orange', 'Purple', 'Fuchsia', 'Pink', 'Cyan',
                         'LightBlue', 'Blue']
    ec_hid_colors = ['White','LightPink','Darkorange','LightBlue','Lime','MediumBlue','DarkGray',
                     'Cyan','Red','Yellow']

    cmaphidw = colors.ListedColormap(hid_colors_winter)
    cmaphid  = colors.ListedColormap(hid_colors_summer)
    cmaphidec = colors.ListedColormap(ec_hid_colors)
    cmapmeth = colors.ListedColormap(hid_colors[0:6])

    field_configs = {
        'CZ': {'units': 'Zh [dBZ]', 'vmin': 0, 'vmax': 70, 'Nbins': 14,
               'title': 'Corrected Reflectivity [dBZ]', 'cmap': check_cm('NWSRef'), 'norm': None},
        'DZ': {'units': 'Zh [dBZ]', 'vmin': 0, 'vmax': 70, 'Nbins': 14,
               'title': 'RAW Reflectivity [dBZ]', 'cmap': check_cm('NWSRef'), 'norm': None},

        'VR': {'units': 'Velocity [m/s]', 'vmin': -20, 'vmax': 20, 'Nbins': 12,
               'title': 'Radial Velocity [m/s]', 'cmap': check_cm('NWSVel'), 'norm': None},
        'corrected_velocity': {'units': 'Velocity [m/s]', 'vmin': -20, 'vmax': 20, 'Nbins': 12,
               'title': 'Dealiased Radial Velocity [m/s]', 'cmap': check_cm('NWSVel'), 'norm': None},

        'SW': {'units': 'Spectrum Width', 'vmin': 0, 'vmax': 21, 'Nbins': 12,
               'title': 'Spectrum Width', 'cmap': check_cm('NWS_SPW'), 'norm': None},

        'DR': {'units': 'Zdr [dB]', 'vmin': -2, 'vmax': 5, 'Nbins': 0,
               'title': 'Differential Reflectivity [dB]', 'cmap': zdr_cmap, 'norm': zdr_norm},
        'RH': {'units': 'ρHV', 'vmin': 0.0, 'vmax': 1.0, 'Nbins': 0,
               'title': 'Cross-Correlation Coefficient', 'cmap': rhohv_cmap, 'norm': rhohv_norm},

        'KD': {'units': 'Kdp [deg/km]', 'vmin': -2, 'vmax': 3, 'Nbins': 10,
               'title': 'Specific Differential Phase [deg/km]', 'cmap': check_cm('HomeyerRainbow'), 'norm': None},
        'KDPB': {'units': 'Kdp [deg/km]', 'vmin': -2, 'vmax': 5, 'Nbins': 8,
                 'title': 'Specific Differential Phase [deg/km] (Bringi)', 'cmap': check_cm('HomeyerRainbow'), 'norm': None},

        'PH': {'units': 'PhiDP [deg]', 'vmin': 0, 'vmax': 360, 'Nbins': 36,
               'title': 'Differential Phase [deg]', 'cmap': check_cm('Carbone42'), 'norm': None},
        'PHM': {'units': 'PhiDP [deg]', 'vmin': 0, 'vmax': 360, 'Nbins': 36,
                'title': 'Differential Phase [deg] Marks', 'cmap': check_cm('Carbone42'), 'norm': None},
        'PHIDPB': {'units': 'PhiDP [deg]', 'vmin': 0, 'vmax': 360, 'Nbins': 36,
                   'title': 'Differential Phase [deg] Bringi', 'cmap': check_cm('Carbone42'), 'norm': None},

        'SD': {'units': 'Std(PhiDP)', 'vmin': 0, 'vmax': 70, 'Nbins': 14,
               'title': 'Standard Deviation of PhiDP', 'cmap': check_cm('NWSRef'), 'norm': None},
        'SQ': {'units': 'SQI', 'vmin': 0, 'vmax': 1, 'Nbins': 10,
               'title': 'Signal Quality Index', 'cmap': check_cm('LangRainbow12'), 'norm': None},

        'FH': {'units': 'HID', 'vmin': 0, 'vmax': 11, 'Nbins': 0,
               'title': 'Summer Hydrometeor Identification', 'cmap': cmaphid, 'norm': None},
        'FH2': {'units': 'HID', 'vmin': 0, 'vmax': 11, 'Nbins': 0,
               'title': 'Summer Hydrometeor Identification', 'cmap': cmaphid, 'norm': None},
        'FS': {'units': 'HID', 'vmin': 0, 'vmax': 11, 'Nbins': 0,
               'title': 'Summer Hydrometeor Identification', 'cmap': cmaphid, 'norm': None},
        'FW': {'units': 'HID', 'vmin': 0, 'vmax': 8, 'Nbins': 0,
               'title': 'Winter Hydrometeor Identification', 'cmap': cmaphidw, 'norm': None},
        'NT': {'units': 'HID', 'vmin': 0, 'vmax': 8, 'Nbins': 0,
               'title': 'No TEMP Winter Hydrometeor Identification', 'cmap': cmaphidw, 'norm': None},
        'EC': {'units': 'HID', 'vmin': 0, 'vmax': 9, 'Nbins': 0,
               'title': 'Radar Echo Classification', 'cmap': cmaphidec, 'norm': None},

        'MW': {'units': 'Water Mass [g/m^3]', 'vmin': 0, 'vmax': 3, 'Nbins': 25,
               'title': 'Water Mass [g/m^3]', 'cmap': 'turbo', 'norm': None},
        'MI': {'units': 'Ice Mass [g/m^3]', 'vmin': 0, 'vmax': 3, 'Nbins': 25,
               'title': 'Ice Mass [g/m^3]', 'cmap': 'turbo', 'norm': None},

        'RC': {'units': 'HIDRO Rain Rate [mm/hr]', 'vmin': 1e-2, 'vmax': 3e2, 'Nbins': 0,
               'title': 'HIDRO Rain Rate [mm/hr]', 'cmap': check_cm('RefDiff'), 'norm': None},
        'RP': {'units': 'PolZR Rain Rate [mm/hr]', 'vmin': 1e-2, 'vmax': 3e2, 'Nbins': 0,
               'title': 'PolZR Rain Rate [mm/hr]', 'cmap': check_cm('RefDiff'), 'norm': None},
        'RA': {'units': 'Attenuation Rain Rate [mm/hr]', 'vmin': 1e-2, 'vmax': 3e2, 'Nbins': 0,
               'title': 'Attenuation Rain Rate [mm/hr]', 'cmap': check_cm('RefDiff'), 'norm': None},

        'MRC': {'units': 'HIDRO Method', 'vmin': 0, 'vmax': 5, 'Nbins': 0,
                'title': 'HIDRO Method', 'cmap': cmapmeth, 'norm': None},
        'MRC2': {'units': 'HIDRO Method', 'vmin': 0, 'vmax': 5, 'Nbins': 0,
                'title': 'HIDRO Method', 'cmap': cmapmeth, 'norm': None},

        'DM': {'units': 'DM [mm]', 'vmin': 0.5, 'vmax': 5, 'Nbins': 8,
               'title': 'DM [mm]', 'cmap': check_cm('BlueBrown10'), 'norm': None},
        'NW': {'units': 'Log[Nw, m^-3 mm^-1]', 'vmin': 0.5, 'vmax': 7, 'Nbins': 12,
               'title': 'Log[Nw, m^-3 mm^-1]', 'cmap': check_cm('BlueBrown10'), 'norm': None},
    }

    cfg = field_configs.get(field, None)
    if cfg is None:
        return ("Unknown", 0, 100, "viridis", f"Unknown Field: {field}", 10, None)

    return (cfg["units"], cfg["vmin"], cfg["vmax"], cfg["cmap"], cfg["title"], cfg["Nbins"], cfg.get("norm", None))

def get_radar_info(radar, sweep):
    site = ""
    if "site_name" in radar.metadata:
        site = radar.metadata["site_name"]
    elif "instrument_name" in radar.metadata:
        site = radar.metadata["instrument_name"]

    if isinstance(site, bytes):
        site = site.decode().upper()
    else:
        site = str(site).upper()

    if radar.metadata.get("original_container") == "odim_h5":
        try:
            site = radar.metadata["source"].replace(",", ":").split(":")[1].upper()
        except Exception:
            site = radar.metadata.get("site_name", "").upper()

    site_mappings = {
        "NPOL1": "NPOL", "NPOL2": "NPOL", "LAVA1": "KWAJ",
        "AN1-P": "AL1", "JG1-P": "JG1", "MC1-P": "MC1", "NT1-P": "NT1",
        "PE1-P": "PE1", "SF1-P": "SF1", "ST1-P": "ST1", "SV1-P": "SV1",
        "TM1-P": "TM1", "GUNN_PT": "CPOL", "REUNION": "Reunion", "CP2RADAR": "CP2",
    }
    site = site.replace("\x00", "").strip()
    site = site_mappings.get(site, site)

    if "system" in radar.metadata:
        site = {"KuD3R": "KuD3R", "KaD3R": "KaD3R"}.get(radar.metadata["system"], site)

    radar_DT = pyart.util.datetime_from_radar(radar)

    if radar_DT.year > 2000 and site in ("NPOL", "KWAJ"):
        EPOCH_UNITS = "seconds since 1970-01-01T00:00:00Z"
        dtrad = num2date(0, radar.time["units"])
        epnum = date2num(dtrad, EPOCH_UNITS)
        radar_DT = num2date(epnum, EPOCH_UNITS)

    elv = radar.fixed_angle["data"][sweep]
    string_csweep = str(sweep).zfill(2)

    year = f"{radar_DT.year:04d}"
    month = f"{radar_DT.month:02d}"
    day = f"{radar_DT.day:02d}"
    hh = f"{radar_DT.hour:02d}"
    mm = f"{radar_DT.minute:02d}"
    ss = f"{radar_DT.second:02d}"
    mydate = f"{month}/{day}/{year}"
    mytime = f"{hh}:{mm}:{ss}"

    return site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep

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
    cb.ax.set_yticklabels(['No Echo', 'Ice Crystals', 'Plates', 'Dendrites',
                          'Aggregates', 'Wet Snow', 'Frozen Precip', 'Rain'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

def adjust_ec_colorbar_for_pyart(cb):
    cb.set_ticks(np.arange(1.4, 9, 0.9))
    cb.ax.set_yticklabels(['Aggregates', 'Ice Crystals', 'Light Rain',
                          'Rimed Particles', 'Rain', 'Vertically Ice',
                          'Wet Snow', 'Melting Hail', 'Dry Hail/High Density Graupel'])
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
    """Create an N-bin discrete colormap from the specified input map."""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N, 0))
    cmap_name = base.name + str(N)
    return plt.cm.colors.ListedColormap(color_list, cmap_name, N)

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.0]
        return np.ma.masked_array(np.interp(value, x, y, left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.0]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)

def check_cm(cmap_name):
    """Handle old/new PyART colormap naming."""
    candidates = [cmap_name, f'pyart_{cmap_name}']
    for name in candidates:
        if name in plt.colormaps():
            return name
    return candidates[-1]

# ****************************************************************************************
# RHI helpers
# ****************************************************************************************

def calculate_layout_rhi(num_fields):
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

    return {'nrows': nrows, 'ncols': ncols, 'width': width,
            'height': height, 'positions': positions}

def create_figure_rhi(layout):
    if layout['ncols'] < 2:
        return plt.figure(figsize=[layout['width'], layout['height']],
                         constrained_layout=False)
    else:
        return plt.figure(figsize=[layout['width'], layout['height']],
                         constrained_layout=True)

def create_gridspec_rhi(layout, fig):
    return plt.GridSpec(ncols=layout['ncols'], nrows=layout['nrows'], figure=fig)

def add_range_rings_fast(display, max_range):
    for rng in range(50, max_range + 50, 50):
        display.plot_range_ring(rng, col='white', ls='-', lw=0.5)

def set_plot_size_parms_rhi(num_fields):
    if num_fields < 2:
        font_config = {
            'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 8,
            'xtick.labelsize': 6, 'ytick.labelsize': 6,
            'legend.fontsize': 6, 'figure.titlesize': 10
        }
    else:
        font_config = {
            'font.size': 12, 'axes.titlesize': 20, 'axes.labelsize': 12,
            'xtick.labelsize': 12, 'ytick.labelsize': 12,
            'legend.fontsize': 12, 'figure.titlesize': 20
        }
    plt.rcParams.update(font_config)

def annotate_plot_rhi_optimized(ax, fig, num_fields, layout):
    nasalogo = _cache.get_logo('nasa')
    gpmlogo = _cache.get_logo('gpm')

    if num_fields < 2:
        imageboxnasa = OffsetImage(nasalogo, zoom=0.07)
        imageboxgpm = OffsetImage(gpmlogo, zoom=0.03)
        imageboxnasa.image.axes = ax
        imageboxgpm.image.axes = ax

        abnasa = AnnotationBbox(imageboxnasa, [0, 0], xybox=[.065, .915],
                                xycoords='axes pixels', boxcoords='axes fraction',
                                pad=-10.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm, [0, 0], xybox=[.93, .925],
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

        abnasa = AnnotationBbox(imageboxnasa, [0, 0], xybox=[140, 245 * nrows],
                                xycoords='figure points', boxcoords='figure points',
                                pad=0.0, frameon=False)
        abgpm = AnnotationBbox(imageboxgpm, [0, 0], xybox=[1550, 245 * nrows],
                               xycoords='figure points', boxcoords='figure points',
                               pad=0.0, frameon=False)
        fig.add_artist(abnasa)
        fig.add_artist(abgpm)

# ======================================================================================
# PPI (Map / Cartopy) - GVview-style axis replacement and fresh display per field
# ======================================================================================

def plot_fields_PPI(
    radar, COUNTIES, STATES, REEFS, MINOR_ISLANDS,
    sweep=0, fields=("CZ",), max_range=150,
    mask_outside=True, png=False, outdir="", add_logos=True
):
    site, mydate, mytime, elv, year, month, day, hh, mm, ss, string_csweep = get_radar_info(radar, sweep)
    radar_lat = radar.latitude["data"][0]
    radar_lon = radar.longitude["data"][0]
    coord_data = _cache.get_coordinate_transform(radar_lat, radar_lon, max_range)

    num_fields = len(fields)
    layout = calculate_layout(num_fields)
    set_plot_size_parms_ppi(num_fields)

    fig = create_figure(layout)
    spec = create_gridspec(layout, fig)

    axes = []
    positions = []
    for i in range(num_fields):
        ax0 = fig.add_subplot(spec[layout["positions"][i]])
        axes.append(ax0)
        positions.append(ax0.get_position())

    projection = ccrs.LambertConformal(radar_lon, radar_lat)

    for i, field in enumerate(fields):
        ax = axes[i]
        pos = positions[i]
        ax.remove()
        ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height], projection=projection)
        ax.set_facecolor("black")
        axes[i] = ax

        display = pyart.graph.RadarMapDisplay(radar)

        units, vmin, vmax, cmap, title, Nbins, norm = get_field_info(radar, field)
        if num_fields < 2:
            title = f"{site} {field} {mydate} {mytime} UTC PPI Elev: {elv:2.1f} deg"
        else:
            mytitle = f"{site} {mydate} {mytime} UTC PPI {elv:2.1f} deg"

        if Nbins > 0:
            cmap = discrete_cmap(Nbins, base_cmap=cmap)

        if field in ("RC", "RP", "RA"):
            processed = _cache.get_processed_field(radar, field)
            plot_name = f"{field}_plot"
            radar.add_field(plot_name, processed, replace_existing=True)
            levels = [0, 5, 10, 15, 20, 25, 100, 150, 200, 250, 300]
            midnorm = MidpointNormalize(vmin=0, vcenter=25, vmax=300)
            display.plot_ppi_map(
                plot_name, sweep, vmin=vmin, vmax=vmax,
                resolution="50m", title=title, projection=projection, ax=ax,
                cmap=cmap, norm=midnorm, ticks=levels, colorbar_label=units,
                min_lon=coord_data["min_lon"], max_lon=coord_data["max_lon"],
                min_lat=coord_data["min_lat"], max_lat=coord_data["max_lat"],
                lon_lines=coord_data["lon_grid"], lat_lines=coord_data["lat_grid"],
                add_grid_lines=False, lat_0=radar_lat, lon_0=radar_lon,
                embellish=False, mask_outside=mask_outside,
            )
        elif field == "DR" and norm is not None:
            display.plot_ppi_map(
                field, sweep, vmin=vmin, vmax=vmax,
                resolution="50m", title=title, projection=projection, ax=ax,
                cmap=cmap, norm=norm, ticks=cbar_limits_zdr, colorbar_label=units,
                min_lon=coord_data["min_lon"], max_lon=coord_data["max_lon"],
                min_lat=coord_data["min_lat"], max_lat=coord_data["max_lat"],
                lon_lines=coord_data["lon_grid"], lat_lines=coord_data["lat_grid"],
                add_grid_lines=False, lat_0=radar_lat, lon_0=radar_lon,
                embellish=False, mask_outside=mask_outside,
            )
        elif field == "RH" and norm is not None:
            display.plot_ppi_map(
                field, sweep, vmin=vmin, vmax=vmax,
                resolution="50m", title=title, projection=projection, ax=ax,
                cmap=cmap, norm=norm, ticks=cbar_limits_rhohv, colorbar_label=units,
                min_lon=coord_data["min_lon"], max_lon=coord_data["max_lon"],
                min_lat=coord_data["min_lat"], max_lat=coord_data["max_lat"],
                lon_lines=coord_data["lon_grid"], lat_lines=coord_data["lat_grid"],
                add_grid_lines=False, lat_0=radar_lat, lon_0=radar_lon,
                embellish=False, mask_outside=mask_outside,
            )
        else:
            display.plot_ppi_map(
                field, sweep, vmin=vmin, vmax=vmax,
                resolution="50m", title=title, projection=projection, ax=ax,
                cmap=cmap, colorbar_label=units,
                min_lon=coord_data["min_lon"], max_lon=coord_data["max_lon"],
                min_lat=coord_data["min_lat"], max_lat=coord_data["max_lat"],
                lon_lines=coord_data["lon_grid"], lat_lines=coord_data["lat_grid"],
                add_grid_lines=False, lat_0=radar_lat, lon_0=radar_lon,
                embellish=False, mask_outside=mask_outside,
            )

        add_rings_radials_optimized_gvstyle(
            year, site, display, radar_lat, radar_lon, max_range,
            ax, COUNTIES, STATES, REEFS, MINOR_ISLANDS
        )

        adjust_special_colorbars(field, display, 0)

    if num_fields >= 2:
        plt.suptitle(mytitle, fontsize=8 * layout["ncols"], weight="bold", y=0.99)

    add_logo_ppi_optimized(axes[-1], add_logos, fig, num_fields, layout)

    save_plot(
        png, outdir, site, year, month, day, hh, mm, ss,
        string_csweep, list(fields), num_fields, "PPI", fig
    )
    plt.close(fig)

def plot_fields_PPI_QC(radar, sweep=0, fields=('CZ',), max_range=150,
                       mask_outside=True, png=False, outdir='', add_logos=True):
    """Fast PPI plotting without Cartopy."""
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

        if Nbins > 0:
            cmap = discrete_cmap(Nbins, base_cmap=cmap)

        ax = fig.add_subplot(spec[layout['positions'][index]])
        ax.set_facecolor('black')

        if norm is not None:
            display.plot_ppi(field, sweep=sweep, vmin=vmin, vmax=vmax, cmap=cmap,
                            norm=norm, colorbar_label=units, mask_outside=mask_outside,
                            title=title, axislabels_flag=False)
        else:
            display.plot_ppi(field, sweep=sweep, vmin=vmin, vmax=vmax, cmap=cmap,
                            colorbar_label=units, mask_outside=mask_outside,
                            title=title, axislabels_flag=False)

        display.set_limits(xlim=[-max_range, max_range], ylim=[-max_range, max_range])
        add_range_rings_fast(display, max_range)
        display.plot_grid_lines(col='white', ls=':')
        display.set_aspect_ratio(aspect_ratio=1.0)

        ax.set_xticklabels('', rotation=0)
        ax.set_yticklabels('', rotation=90)
        ax.set_xlabel('')
        ax.set_ylabel('')

        if num_fields >= 2:
            plt.suptitle(mytitle, fontsize=8 * layout['ncols'], weight='bold',
                        y=(1.0 + (layout['nrows'] * 0.055)))

        adjust_special_colorbars(field, display, index)

    save_plot(png, outdir, site, year, month, day, hh, mm, ss, string_csweep,
             list(fields), num_fields, 'PPI', fig)
    plt.close(fig)

def plot_fields_RHI(radar, sweep=0, fields=('CZ',), ymax=10, xmax=150,
                   png=False, outdir='', add_logos=True, mask_outside=True):
    """RHI plotting."""
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

        if Nbins > 0:
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

        if field in ('RC', 'RP'):
            processed_field = _cache.get_processed_field(radar, field)
            plot_name = f"{field}_plot"
            radar.add_field(plot_name, processed_field, replace_existing=True)
            levels = [0, 5, 10, 15, 20, 25, 100, 150, 200, 250, 300]
            midnorm = MidpointNormalize(vmin=0, vcenter=25, vmax=300)
            display.plot_rhi(plot_name, sweep, vmin=vmin, vmax=vmax, cmap=cmap,
                            title=title, mask_outside=mask_outside,
                            colorbar_label=units, norm=midnorm, ticks=levels)
        elif norm is not None:
            ticks = cbar_limits_zdr if field == 'DR' else cbar_limits_rhohv
            display.plot_rhi(field, sweep, vmin=vmin, vmax=vmax, cmap=cmap,
                            title=title, mask_outside=mask_outside,
                            colorbar_label=units, norm=norm, ticks=ticks)
        else:
            display.plot_rhi(field, sweep, vmin=vmin, vmax=vmax, cmap=cmap,
                            title=title, mask_outside=mask_outside,
                            colorbar_label=units)

        display.set_limits(xlim, ylim, ax=ax)
        display.plot_grid_lines(col='white')

        if add_logos:
            annotate_plot_rhi_optimized(ax, fig, num_fields, layout)

        adjust_special_colorbars(field, display, index)

    if num_fields >= 2:
        plt.suptitle(mytitle, fontsize=28, weight='bold')

    save_plot(png, outdir, site, year, month, day, hh, mm, ss, string_csweep,
             list(fields), num_fields, 'RHI', fig, azi)
    plt.close(fig)