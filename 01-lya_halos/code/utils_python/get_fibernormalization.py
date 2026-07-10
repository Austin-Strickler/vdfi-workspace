#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:52:44 2023

@author: gregz
"""
import numpy as np
import argparse as ap
import logging
import warnings
import tables
import sys
from astropy.table import Table
sys.path.append("..") 

import matplotlib.pyplot as plt
import seaborn as sns
from skimage.registration import phase_cross_correlation
from astropy.time import Time
from astropy.io import fits
from multiprocessing import Pool
from scipy.interpolate import interp1d
from virusraw import VIRUSRaw


def setup_logging(logname='input_utils'):
    '''Set up a logger for shuffle with a name ``input_utils``.

    Use a StreamHandler to write to stdout and set the level to DEBUG if
    verbose is set from the command line
    '''
    log = logging.getLogger('input_utils')
    if not len(log.handlers):
        fmt = '[%(levelname)s - %(asctime)s] %(message)s'
        fmt = logging.Formatter(fmt)

        level = logging.INFO

        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler.setLevel(level)

        log = logging.getLogger('input_utils')
        log.setLevel(logging.DEBUG)
        log.addHandler(handler)
    return log

warnings.filterwarnings("ignore")


parser = ap.ArgumentParser(add_help=True)

parser.add_argument('object_table', type=str,
                    help='''name of the output file''')

parser.add_argument('hdf5file', type=str,
                    help='''name of the hdf5 file''')

parser.add_argument('outname', type=str,
                    help='''name appended to shift output''')
args = parser.parse_args(args=None)
args.log = setup_logging('get_object_table')

def_wave = np.linspace(3470, 5540, 1036)

sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams["font.family"] = "Times New Roman"

basedir = '/work/03946/hetdex/maverick'
hdf5file = args.hdf5file
h5file = tables.open_file(hdf5file, mode='r')
h5table = h5file.root.Cals
ifuslots = list(np.unique(['%03d' % i for i in h5table.cols.ifuslot[:]]))
ifuslots = ifuslots
T = Table.read(args.object_table, format='ascii.fixed_width_two_line')

keys = list([str(t) for t in T['Exposure']])
values = list(T['Description'])

twi_obs = [key for key, value in zip(keys, values) if value == 'skyflat']

twi_list = twi_obs

def get_fibernorm(twi, bins=25):
    date = twi[:8]
    obs = int(twi[8:15])
    exp = int(twi[15:])
    virus = VIRUSRaw(date, obs, h5table, basepath=basedir, exposure_number=exp,
                     ifuslots=ifuslots)
    allspec = np.ones((448 * len(ifuslots), 1036)) * np.nan
    twi_dictionary = {}
    cnt = 0
    for ifuslot in ifuslots:
        twi_dictionary[ifuslot] = np.zeros((448, 1036))
        try:
            current_observation = virus.info[ifuslot].data
            allspec[cnt:cnt+448] = current_observation
            cnt += 448
        except:
            args.log.warning('Could not get data for %s_%s' % (twi, ifuslot))
            cnt += 448
            continue
    wave = np.linspace(3470, 5540, 1036)
    avg = np.nanmedian(allspec, axis=0)
    avgi = np.array([np.nanmedian(xi) for xi in np.array_split(avg, bins)])
    w = np.array([np.nanmedian(xi) for xi in np.array_split(wave, bins)])
    binned_spec = np.ones((448 * len(ifuslots), bins)) * np.nan
    ftf = np.nan * allspec
    for i in np.arange(allspec.shape[0]):
        s = np.array([np.nanmedian(xi) for xi in np.array_split(allspec[i], bins)])
        sel = np.isfinite(s)
        if sel.sum() > (bins-5):
            ftf[i] = interp1d(w[sel], s[sel] / avgi[sel], kind='quadratic', fill_value='extrapolate')(wave)
    cnt = 0
    for ifuslot in ifuslots:
        twi_dictionary[ifuslot] = ftf[cnt:cnt+448]
        cnt += 448
    timeobs = Time(virus.info[ifuslot].header['DATE'])
    rho = virus.info[ifuslot].header['RHO_STRT']
    the = virus.info[ifuslot].header['THE_STRT']
    phi = virus.info[ifuslot].header['PHI_STRT']
    hum = virus.info[ifuslot].header['HUMIDITY']
    temp = virus.info[ifuslot].header['AMBTEMP']
    virus.log.info('Shifts finished %s_%07d_exp%02d' % (date, obs, exp))
    return twi_dictionary, timeobs, rho, the, phi, hum, temp


P = Pool(16)
res = P.map(get_fibernorm, twi_list)
P.close()
twi_dictionary = {}
for ifuslot in ifuslots:
    twi_dictionary[ifuslot] = np.nan * np.ones((len(twi_list), 448, 1036))
for ifuslot in ifuslots:  
    twi_dictionary[ifuslot] = [r[0][ifuslot] for r in res]
time_list = [r[1] for r in res]
rho_list = [r[2] for r in res]
the_list = [r[3] for r in res]
phi_list = [r[4] for r in res]
hum_list = [r[5] for r in res]
temp_list = [r[6] for r in res]
for ifuslot in ifuslots:
    name = 'fibernormalization_model_%s_%s.fits' % (ifuslot, args.outname)
    f  = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(twi_dictionary[ifuslot]),
                       fits.ImageHDU(np.array([t.mjd for t in time_list])),
                       fits.ImageHDU(np.array([t for t in rho_list])),
                       fits.ImageHDU(np.array([t for t in the_list])),
                       fits.ImageHDU(np.array([t for t in phi_list])),
                       fits.ImageHDU(np.array([t for t in hum_list])),
                       fits.ImageHDU(np.array([t for t in temp_list]))])
    f.writeto(name, overwrite=True)
