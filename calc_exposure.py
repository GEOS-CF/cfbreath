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


def main(args):
    log = logging.getLogger(__name__)
    log.info('Reading {}'.format(args.mask_file))
    masks = xr.open_dataset(args.mask_file)
    log.info('Reading {}'.format(args.popfile))
    pop = xr.open_dataset(args.popfile)
    dat = pd.DataFrame()
    startdate = dt.datetime(args.year1,args.month1,1,0,0,0)
    enddate = dt.datetime(args.year2,args.month2,1,0,0,0)
    times = []
    idate = startdate
    while idate <= enddate:
        times.append(idate)
        idate = idate + dt.timedelta(days=35)
        idate = dt.datetime(idate.year,idate.month,1,0,0,0)
    for itime in times:
        exp = _get_exposure(args,itime,masks,pop)
        dat = dat.append(exp)
    if os.path.isfile(args.output_file) and args.append==1:
        orig = pd.read_csv(args.output_file,parse_dates=['datetime'],date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m'))
        dat = orig.append(dat)
        dat = dat.sort_values(by='datetime')
    dat.to_csv(args.output_file,date_format='%Y-%m',index=False,float_format='%.1f')
    log.info('Written to {}'.format(args.output_file))
    return


def _get_exposure(args,itime,masks,pop):
    log = logging.getLogger(__name__)
    exp = pd.DataFrame()
    exp['datetime'] = [itime]
    pm25 = _get_pm25(args,itime)
    poparr = pop[args.popvar].sel(time=itime,method='nearest').values[:,:]
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
        exp[imask] = [ival*1.0e9]
    return exp


def _get_pm25(args,itime):
    log = logging.getLogger(__name__)
    ifile = args.merra2_dir.replace('$p',args.merra2_prefix)
    ifile = itime.strftime(ifile)
    log.info('Reading {}'.format(ifile))
    ds = xr.open_dataset(ifile)
    pm25 = ds['DUSMASS'].values[0,:,:] \
         + ds['OCSMASS'].values[0,:,:] \
         + ds['BCSMASS'].values[0,:,:] \
         + ds['SSSMASS'].values[0,:,:] \
         + ( ds['SO4SMASS'].values[0,:,:] * (132.14/96.06) )
    ds.close()
    return pm25


def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-y1', '--year1',type=int,help='start year',default=2000)
    p.add_argument('-y2', '--year2',type=int,help='end year',default=2010)
    p.add_argument('-m1', '--month1',type=int,help='start month',default=1)
    p.add_argument('-m2', '--month2',type=int,help='end month',default=12)
    p.add_argument('-m', '--mask-file',type=str,help='mask file',default='outputs/masks/mask_hlabisa.nc4')
    p.add_argument('-pf', '--popfile',type=str,help='population file',default="inputs/templates/gpw-v4-population-density-rev10.merra2.nc")
    p.add_argument('-pv', '--popvar',type=str,help='population file variable',default="Population Density, v4.10 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes")
    p.add_argument('-o', '--output-file',type=str,help='output file',default="pm25.txt")
    p.add_argument('-a', '--append',type=int,help='append to existing file',default=1)
    p.add_argument('-md', '--merra2-dir',type=str,help='merra2 data directory',default='/discover/nobackup/projects/gmao/merra2/data/products/$p/Y%Y/M%m/$p.tavg1_2d_aer_Nx.monthly.%Y%m.nc4')
    p.add_argument('-mp', '--merra2-prefix',type=str,help='merra2 file prefix',default='d5124_m2_jan00')
    return p.parse_args()


if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    log.addHandler(handler)
    main(parse_args())
