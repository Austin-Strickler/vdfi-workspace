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
LDLS_obs = [key for key, value in zip(keys, values) if value == 'ldls_long']

x_list = [100., 300., 500., 700., 900.]
thresh = 150.

flt_list = LDLS_obs

def get_shift(flt):
    date = flt[:8]
    obs = int(flt[8:15])
    exp = int(flt[15:])
    virus = VIRUSRaw(date, obs, h5table, basepath=basedir, exposure_number=exp,
                     ifuslots=ifuslots, lightmode=False)
    shift_dictionary = {}
    for ifuslot in ifuslots:
        shift_dictionary[ifuslot] = np.zeros((448, len(x_list)))
    for ifuslot in ifuslots:
        try:
            monthly_average = virus.info[ifuslot].masterflt * 1.
            current_observation = virus.info[ifuslot].image * 1.
        except:
            virus.log.warning("Can't calculate shift for %s" % ifuslot)
            continue
        current_observation[np.isnan(current_observation)] = 0.0
        trace = virus.info[ifuslot].trace * 1.
        shifts = np.zeros((current_observation.shape[0], len(x_list)))
        yran = np.arange(current_observation.shape[0])
        xran = np.arange(current_observation.shape[1])
        for fiber in np.arange(448):
            fit_waves = [np.abs(xran - line) <= 20. for line in x_list]
            for j, waverange in enumerate(fit_waves):
                trace_range = np.abs(trace[fiber, int(x_list[j])] - yran) < 4.
                X = np.nanmedian(current_observation[:, waverange], axis=1)
                Y = np.nanmedian(monthly_average[:, waverange], axis=1)
                if np.nanmax(X[trace_range]) > thresh:
                    FFT = phase_cross_correlation(X[trace_range][:, np.newaxis],
                                            Y[trace_range][:, np.newaxis], 
                                            normalization=None, upsample_factor=100)
                    shifts[fiber, j] = FFT[0][0]
        shift_dictionary[ifuslot] = shifts
    timeobs = Time(virus.info[ifuslot].header['DATE'])
    hum = virus.info[ifuslot].header['HUMIDITY']
    temp = virus.info[ifuslot].header['AMBTEMP']
    virus.log.info('Shifts finished %s_%07d_exp%02d' % (date, obs, exp))
    return shift_dictionary, timeobs, hum, temp


P = Pool(16)
res = P.map(get_shift, flt_list)
P.close()
shift_dictionary = {}
for ifuslot in ifuslots:
    shift_dictionary[ifuslot] = np.zeros((len(flt_list), 448, len(x_list)))
for ifuslot in ifuslots:  
    shift_dictionary[ifuslot] = [r[0][ifuslot] for r in res]
time_list = [r[1] for r in res]
hum_list = [r[2] for r in res]
temp_list = [r[3] for r in res]
for ifuslot in ifuslots:
    name = 'trace_shifts_%s_%s.fits' % (ifuslot, args.outname)
    try:
        f  = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(shift_dictionary[ifuslot]),
                           fits.ImageHDU(np.array([t.mjd for t in time_list])),
                           fits.ImageHDU(np.array([t for t in hum_list])),
                           fits.ImageHDU(np.array([t for t in temp_list]))])
        f.writeto(name, overwrite=True)
    except:
        print('Failed making %s' %name)