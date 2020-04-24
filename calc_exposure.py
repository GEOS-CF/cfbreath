#!/bin/python

# Requirements
import logging
import argparse
import numpy as np
import xarray as xr
import math
import sys
import pandas as pd
import datetime as dt
import os
import glob
from multiprocessing.pool import ThreadPool
import dask


# Definition of model variables and corresponding scale factors
MODELVARS = {
 'DUSMASS': 1.0,
 'OCSMASS': 1.0,
 'BCSMASS': 1.0,
 'SSSMASS': 1.0,
 'SO4SMASS': 132.14/96.06,
}


def calc_exposure(args):
    '''
    Calculate population weighted PM2.5 exposure for user-provided areas.
    The PM2.5 exposure is calculated from three data sources:
    1. Gridded PM2.5 concentration data (e.g. GEOS-CF, GEOS-FP, or MERRA-2).
    2. Gridded region masks, to determine the area over which the concentration
       averaged shall be calculated.
    3. Gridded population density, to provide the weights given to the individual 
       cells within an area.
    The values are calculated for a sequence of times, with the start and end date
    as well as the time frequency being supplied via the argument list.
    The mask file is expected to be a netCDF file. Script 'shp2mask.py' can be used
    to generate a netCDF file from shape files.
    '''
    log = logging.getLogger(__name__)
    dask.config.set(pool=ThreadPool(10))
    # read masks
    log.info('Reading {}'.format(args.mask_file))
    masks = xr.open_dataset(args.mask_file)
    # read population density file 
    log.info('Reading {}'.format(args.popfile))
    pop = xr.open_dataset(args.popfile)
    # prepare output dataframe
    dat = pd.DataFrame()
    startdate = dt.datetime(args.year1,args.month1,1,0,0,0)
    enddate = dt.datetime(args.year2,args.month2,1,0,0,0)
    times = pd.date_range(start=startdate,end=enddate,freq=args.read_freq).tolist()
    # loop over all time stamps, exclude last time stamp
    for i,itime in enumerate(times):
        if i==len(times)-1:
            continue
        exp = _get_exposure(args,times,i,masks,pop)
        dat = dat.append(exp)
    # write to file
    if os.path.isfile(args.output_file) and args.append==1:
        orig = pd.read_csv(args.output_file,parse_dates=['datetime'],date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m'))
        dat = orig.append(dat)
        dat = dat.sort_values(by='datetime')
    dat.to_csv(args.output_file,date_format=args.date_format,index=False,float_format='%.1f')
    log.info('Written to {}'.format(args.output_file))
    return


def _get_exposure(args,times,i,masks,pop):
    '''Calculate the population weighted average PM2.5 exposure for each mask region and the given time'''
    log = logging.getLogger(__name__)
    t = times[i]
    starttime = dt.datetime(t.year,t.month,t.day,t.hour,t.minute,t.second)
    t = times[i+1]
    endtime = dt.datetime(t.year,t.month,t.day,t.hour,t.minute,t.second)
    # get PM2.5 data
    pm25 = _get_pm25(args,starttime,endtime)
    # pick population density for that time
    poparr = pop[args.popvar].sel(time=starttime,method='nearest')
    if poparr.shape != pm25.shape:
        log.info('interpolating population array...')
        poparr = poparr.interp_like(pm25)
        log.info('interpolation done!')
    poparr = poparr.values[:,:]
    poparr[np.isnan(poparr)] = 0.0
    # calculate population-weighted PM2.5 exposure for each masked area
    exp = pd.DataFrame()
    exp['date'] = [starttime]
    pm25arr = pm25.values[:,:]
    for imask in masks:
        if imask=='mask':
            continue
        mask = masks[imask].mean(dim='time')
        if mask.shape != pm25.shape:
            log.info('interpolating population array...')
            mask = mask.interp_like(pm25)
            log.info('interpolation done!')
        mask = mask.values[:,:]
        mask[np.isnan(mask)] = 0.0
        wgt = mask[:,:]*poparr[:,:]
        if np.sum(wgt) > 0.0:
            ival = np.sum(pm25arr[:,:]*wgt[:,:]) / np.sum(wgt)
        else: 
            ival = 0.0
        ivar = imask.replace('\\',"")
        exp[imask] = [ival*1.0e9]
    return exp


def _get_pm25(args,starttime,endtime):
    '''Read model data and return PM2.5 (model) concentrations'''
    log = logging.getLogger(__name__)
    ifile = starttime.strftime(args.model_template)
    log.info('Reading {}'.format(ifile))
    # Open file
    if 'opendap' in ifile:
        ds = xr.open_dataset(ifile).sel(time=slice(starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))).mean(dim='time')
    else:
        files = glob.glob(ifile)
        if len(files)==0:
            log.warning('No files found: {}'.format(ifile))
        if len(files)==1:
            ds = xr.open_dataset(files).mean(dim='time')
        else:
            ds = xr.open_mfdataset(files).sel(time=slice(starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))).mean(dim='time')
    # Read all variables and apply scale factors
    vars = list(MODELVARS.keys())
    scal = list(MODELVARS.values())
    pm25 = ds[vars[0]]*scal[0]
    if len(vars)>1:
        for v,s in zip(vars[1:],scal[1:]):
            pm25 = pm25 + ds[v]*s
    ds.close()
    return pm25


def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-y1', '--year1',type=int,help='start year',default=2016)
    p.add_argument('-y2', '--year2',type=int,help='end year',default=2016)
    p.add_argument('-m1', '--month1',type=int,help='start month',default=1)
    p.add_argument('-m2', '--month2',type=int,help='end month',default=2)
    p.add_argument('-m', '--mask-file',type=str,help='file with region masks. A separate PM2.5 value will be calculated for every variable on the file',default='outputs/masks/cb_2015_us_zcta510_500k.nc4')
    p.add_argument('-pf', '--popfile',type=str,help='population file',default="inputs/templates/gpw-v4-population-density-rev10.01x01.nc")
    p.add_argument('-pv', '--popvar',type=str,help='population file variable',default="Population Density, v4.10 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes")
    p.add_argument('-o', '--output-file',type=str,help='output file',default="PM25_NY_MERRA2.txt")
    p.add_argument('-a', '--append',type=int,help='append to existing file (1=yes, 0=no)?',default=1)
    p.add_argument('-t', '--model-template',type=str,help='model file template',default='/discover/nobackup/projects/gmao/merra2/data/products/d5124_m2_jan10/Y%Y/M%m/MERRA2_400.tavg1_2d_aer_Nx.%Y%m%d.nc4')
    p.add_argument('-rf', '--read-freq',type=str,help='model file reading frequency, use "1D" for daily, "MS" for monthly',default='1D')
    p.add_argument('-df', '--date-format',type=str,help='datetime format in output file',default='%Y-%m-%d')
    return p.parse_args()


if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    log.addHandler(handler)
    calc_exposure(parse_args())
