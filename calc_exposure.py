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


def main(args):
    log = logging.getLogger(__name__)
    dask.config.set(pool=ThreadPool(10))
    log.info('Reading {}'.format(args.mask_file))
    masks = xr.open_dataset(args.mask_file)
    log.info('Reading {}'.format(args.popfile))
    pop = xr.open_dataset(args.popfile)
    dat = pd.DataFrame()
    startdate = dt.datetime(args.year1,args.month1,1,0,0,0)
    enddate = dt.datetime(args.year2,args.month2,1,0,0,0)
    times = pd.date_range(start=startdate,end=enddate,freq=args.read_freq).tolist()
    for itime in times:
        exp = _get_exposure(args,itime,masks,pop)
        dat = dat.append(exp)
    if os.path.isfile(args.output_file) and args.append==1:
        orig = pd.read_csv(args.output_file,parse_dates=['datetime'],date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m'))
        dat = orig.append(dat)
        dat = dat.sort_values(by='datetime')
    dat.to_csv(args.output_file,date_format=args.date_format,index=False,float_format='%.1f')
    log.info('Written to {}'.format(args.output_file))
    return


def _get_exposure(args,itime,masks,pop):
    log = logging.getLogger(__name__)
    exp = pd.DataFrame()
    exp['datetime'] = [itime]
    poparr = pop[args.popvar].sel(time=itime,method='nearest')
    pm25 = _get_pm25(args,itime,poparr)
    poparr = poparr.values[:,:]
    poparr[np.isnan(poparr)] = 0.0
    for imask in masks:
        if imask=='mask':
            continue
        mask = masks[imask].values[0,:,:]
        mask[np.isnan(mask)] = 0.0
        wgt = mask[:,:]*poparr[:,:]
        if np.sum(wgt) > 0.0:
            ival = np.sum(pm25[:,:]*wgt[:,:]) / np.sum(wgt)
        else: 
            ival = 0.0
        ivar = imask.replace('\\',"")
        exp[imask] = [ival*1.0e9]
    return exp


def _get_pm25(args,itime,poparr):
    log = logging.getLogger(__name__)
    ifile = args.merra2_dir.replace('$p',args.merra2_prefix)
    ifile = itime.strftime(ifile)
    log.info('Reading {}'.format(ifile))
    files = glob.glob(ifile)
    if len(files)==0:
        log.warning('No files found: {}'.format(ifile))
    if len(files)==1:
        ds = xr.open_dataset(files).mean(dim='time')
    else:
        ds = xr.open_mfdataset(files).mean(dim='time')
    if ds['DUSMASS'].shape != poparr.shape:
        log.info('interpolating...')
        pm25 = ds['DUSMASS'] \
             + ds['OCSMASS'] \
             + ds['BCSMASS'] \
             + ds['SSSMASS'] \
             + ( ds['SO4SMASS'] * (132.14/96.06) )
        pm25 = pm25.interp_like(poparr)
        pm25 = pm25.values[:,:]
        log.info('interpolation done!')
    else:
        pm25 = ds['DUSMASS'].values[:,:] \
             + ds['OCSMASS'].values[:,:] \
             + ds['BCSMASS'].values[:,:] \
             + ds['SSSMASS'].values[:,:] \
             + ( ds['SO4SMASS'].values[:,:] * (132.14/96.06) )
    ds.close()
    return pm25


def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-y1', '--year1',type=int,help='start year',default=2016)
    p.add_argument('-y2', '--year2',type=int,help='end year',default=2016)
    p.add_argument('-m1', '--month1',type=int,help='start month',default=1)
    p.add_argument('-m2', '--month2',type=int,help='end month',default=2)
    p.add_argument('-m', '--mask-file',type=str,help='mask file',default='outputs/masks/cb_2015_us_zcta510_500k.nc4')
    p.add_argument('-pf', '--popfile',type=str,help='population file',default="inputs/templates/gpw-v4-population-density-rev10.merra2.nc")
    p.add_argument('-pv', '--popvar',type=str,help='population file variable',default="Population Density, v4.10 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes")
    p.add_argument('-o', '--output-file',type=str,help='output file',default="PM25_NY_MERRA2.txt")
    p.add_argument('-a', '--append',type=int,help='append to existing file',default=1)
    #p.add_argument('-md', '--merra2-dir',type=str,help='merra2 data directory',default='/discover/nobackup/projects/gmao/merra2/data/products/$p/Y%Y/M%m/$p.tavg1_2d_aer_Nx.monthly.%Y%m.nc4')
    p.add_argument('-md', '--merra2-dir',type=str,help='merra2 data directory',default='/discover/nobackup/projects/gmao/merra2/data/products/d5124_m2_jan10/Y%Y/M%m/MERRA2_400.tavg1_2d_aer_Nx.%Y%m%d.nc4')
    #p.add_argument('-md', '--merra2-dir',type=str,help='merra2 data directory',default='/discover/nobackup/projects/gmao/gmao_ops/pub/fp/das/Y%Y/M%m/D%d/GEOS.fp.asm.tavg3_2d_aer_Nx.*.nc4')
    p.add_argument('-mp', '--merra2-prefix',type=str,help='merra2 file prefix',default='d5124_m2_jan00')
    p.add_argument('-rf', '--read-freq',type=str,help='model file reading frequency',default='1D')
    p.add_argument('-df', '--date-format',type=str,help='datetime format in output file',default='%Y-%m-%d')
    return p.parse_args()



if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    log.addHandler(handler)
    main(parse_args())
