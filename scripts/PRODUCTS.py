from gvradar import GVradar
import pyart
import sys
import glob

import warnings
warnings.filterwarnings("ignore")

# ***************************************************************************************

def get_kwargs():

    kwargs_product = {}

    # Select if you want to output a cf file and what fields to write
    kwargs_product.update({'output_cf': True, 'cf_dir': './cf/',
                           'output_fields': ['DZ', 'CZ', 'VR', 'DR', 'KD',
                                             'PH', 'RH', 'SD', 'FS', 'FW',
                                             'RC', 'DM', 'NW', 'SQ']})
    kwargs_product.update({'output_grid': False, 'grid_dir': './grid/',
                           'output_fields': ['DZ', 'CZ', 'VR', 'DR', 'KD',
                                             'PH', 'RH', 'SD', 'FS',
                                             'RC', 'DM', 'NW']})

    # Select which products to produce.
    kwargs_product.update({'do_HID_summer': True,
                           'do_HID_winter': True,
                           'do_mass': True,
                           'do_RC': True,
                           'do_tokay_DSD': True,
                           'dsd_loc': 'wff'})

    # Select plots ranges, type, and fields
    kwargs_product.update({'plot_images': True, 'plot_single': True, 'plot_multi': False,
                           'max_range': 150, 'max_height': 14, 'sweeps_to_plot': [0],
                           'fields_to_plot': ['DZ', 'CZ', 'VR', 'DR', 'KD',
                                             'PH', 'RH', 'SD', 'FS', 'FW',
                                             'RC', 'DM', 'NW'],
                           'plot_dir': './plots/', 'add_logos': True})

    # A Sounding is needed for DP products, sounding type can be; uwy, ruc, ruc_archive)
    kwargs_product.update({'use_sounding': True, 'sounding_type': 'ruc_archive',
                           'sounding_dir': '/Users/jpippitt/sounding/'})

    return kwargs_product

# *******************************************  M  A  I  N  **************************************************

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit("Usage: python run_GVradar.py in_dir")

    in_dir = sys.argv[1]

    # Get full filelist for this day
    wc = in_dir + '*'
    files = sorted(glob.glob(wc))
    numf = len(files)

    kwargs_product = get_kwargs()

    for file in files:
        radar = []
        #file = 'QC_radar'
        d = GVradar.DP_products(file, radar, **kwargs_product)
        d.run_DP_products()
