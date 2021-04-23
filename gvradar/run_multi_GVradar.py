'''

Program to run GVradar with multi-process.
Written by Jason Pippitt NASA/SSAI, GPM-GV group.

'''
# ******************************************************************************************************

from __future__ import print_function
import numpy as np
import math
import os, sys, glob, ast, itertools
import datetime
from gvradar import GVradar
import pathlib
import concurrent.futures
import time
import argparse

# ******************************************************************************************************
 
def dpqc(x, kwargs):
    q = GVradar.QC(files[x], **kwargs)
    qc_radar = q.run_dpqc()
    return qc_radar

# ******************************************************************************************************

def products(x, kwargs_product):
    radar = []
    d = GVradar.DP_products(files[x], radar, **kwargs_product)
    d.run_DP_products()
   
# ******************************************************************************************************

def products_radars(x, kwargs_product):
    print(radars[x])
    file = 'QC_radar'
    d = GVradar.DP_products(file, radars[x], **kwargs_product)
    d.run_DP_products()
   
# ******************************************************************************************************

if(__name__ == "__main__"):

    runargs = argparse.ArgumentParser(description='User information')
    runargs.add_argument('in_dir', type=str, help='Input Directory')
    runargs.add_argument('--stime', dest= 'stime', nargs="+", 
                         help='Year Month Day Hour Minute ex: 2020 1 1 0 59')
    runargs.add_argument('--etime', dest= 'etime', nargs="+", 
                         help='Year, Month, Day, Hour, Minute ex: 2020 1 1 23 59')
    runargs.add_argument('--thresh_dict', dest='thresh_dict', type=str, help='Threshold dictionary')
    runargs.add_argument('--product_dict', dest='product_dict', type=str, help='DP product dictionary')
    runargs.add_argument('--do_qc', action="store_true", help='Run QC')
    runargs.add_argument('--dp_products', action="store_true", help='Create DP products')

    args = runargs.parse_args()

    if args.do_qc and args.thresh_dict == None: 
        print('No threshold dictionary, applying default thresholds.', '', sep='\n')
    if args.dp_products and args.product_dict == None:
        print('No product dictionary, applying defaults.', '', sep='\n' )
    if args.do_qc == False and args.dp_products == False: 
        sys.exit('Please specify actions: --do_qc --dp_products at least 1 must be selected.')
    
    if args.do_qc:
        if args.thresh_dict:
            kwargs = ast.literal_eval(open(args.thresh_dict).read())
        else:
            kwargs = {}
    
    if args.dp_products:
        if args.product_dict:
            kwargs_product = ast.literal_eval(open(args.product_dict).read())
        else:
            kwargs_product = {}

    stime = args.stime
    etime = args.etime
    year = stime[0]
    month = stime[1]
    day = stime[2]

    in_dir = args.in_dir + '/' + year + '/' + month + day + '/ppi/'

# Get full file list for this day
    wc = in_dir + '*'
    all_files = sorted(glob.glob(wc))
    nf = len(all_files)
    if(nf == 0):
        print("No files found in " + wc)
        sys.exit("Bye.")

    DT_beg = datetime.datetime(*map(int, stime))
    DT_end = datetime.datetime(*map(int, etime))

# Get date/time from filename
    files = []
    for file in all_files:
        fileb = os.path.basename(file)
        cfy = pathlib.Path(file).suffix
        if cfy == '.cf':
            x = fileb.split('_')
            year  = int(x[1])
            month = int(x[2][0:2])
            day   = int(x[2][2:4])
            hour  = int(x[3][0:2])
            mint  = int(x[3][2:4])
            sec   = int(x[3][4:6])
        elif cfy == '.gz':
            x = fileb.split('.')
            year  = int(x[0][3:5])
            year = year + 2000
            month = int(x[0][5:7])
            day   = int(x[0][7:9])
            hour  = int(x[0][9:11])
            mint  = int(x[0][11:13])
            sec   = int(x[0][13:15])
        elif cfy == '':
            x = fileb.split('_')
            year  = int(x[0][4:8])
            month = int(x[0][8:10])
            day   = int(x[0][10:12])
            hour  = int(x[1][0:2])
            mint  = int(x[1][2:4])
            sec   = int(x[1][4:6])

# Grab files that fall within stime and etime            
        DT = datetime.datetime(year, month, day, hour, mint, sec)
        if (DT >= DT_beg) & (DT < DT_end):
            files.append(file)

    print(files) 

    n_files= len(files)
 
# Use multi-process 
    beg_time = time.perf_counter()
    lst = range(n_files)
    radars = []

    if args.do_qc:
        with concurrent.futures.ProcessPoolExecutor() as qc:
            for num in qc.map(dpqc, lst, itertools.repeat(kwargs, len(lst))):
                radars.append(num)
       
    if args.dp_products and args.do_qc == False:
        with concurrent.futures.ProcessPoolExecutor() as dp:
            results = dp.map(products, lst, itertools.repeat(kwargs_product, len(lst)))

    if args.dp_products and args.do_qc:
        with concurrent.futures.ProcessPoolExecutor() as dp_radars:
            results = dp_radars.map(products_radars, lst, itertools.repeat(kwargs_product, len(lst)))
              
    end_time = time.perf_counter()
    print(f'Run time was {round(((end_time-beg_time)/60), 1)} minute(s)')
    print()
    print('Done.')

