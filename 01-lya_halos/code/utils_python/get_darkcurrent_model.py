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

Dark_obs = [key for key, value in zip(keys, values) if value == 'dark']


line_list = [3610.508, 4046.565, 4358.335, 4678.149, 4799.912,
                      4916.068, 5085.822, 5460.750]

thresh = 150.

dark_list = Dark_obs

def get_dark(dark):
    date = dark[:8]
    obs = int(dark[8:15])
    exp = int(dark[15:])
    virus = VIRUSRaw(date, obs, h5table, basepath=basedir, exposure_number=exp,
                     ifuslots=ifuslots)
    dark_dictionary = {}
    for ifuslot in ifuslots:
        dark_dictionary[ifuslot] = np.zeros((448, 1036))
    for ifuslot in ifuslots:
        try:
            current_observation = virus.info[ifuslot].data
        except:
            args.log.warning('Could not get dark current measurement for %s_%s' % (dark, ifuslot))
            continue
        nchunks = 14
        nfib, ncols = virus.info[ifuslot].data.shape
        Z = np.zeros((nfib, nchunks))
        xi = [np.mean(x) for x in np.array_split(np.arange(ncols), nchunks)]
        x = np.arange(ncols)
        i = 0
        for chunk in np.array_split(current_observation, nchunks, axis=1):
            Z[:, i] = np.nanmedian(chunk, axis=1)
            i += 1
        image = current_observation * 0.
        for ind in np.arange(Z.shape[0]):
            p0 = np.polyfit(xi, Z[ind], 4)
            model = np.polyval(p0, x)
            image[ind] = model
        mult1 = (virus.info[ifuslot].area *
                 virus.info[ifuslot].transparency *
                 virus.info[ifuslot].exptime)
        dark_dictionary[ifuslot] = image * mult1
    timeobs = Time(virus.info[ifuslot].header['DATE'])
    hum = virus.info[ifuslot].header['HUMIDITY']
    temp = virus.info[ifuslot].header['AMBTEMP']
    virus.log.info('Shifts finished %s_%07d_exp%02d' % (date, obs, exp))
    return dark_dictionary, timeobs, hum, temp


P = Pool(16)
res = P.map(get_dark, dark_list)
P.close()
dark_dictionary = {}
for ifuslot in ifuslots:
    dark_dictionary[ifuslot] = np.nan * np.ones((len(dark_list), 448, 1036))
for ifuslot in ifuslots:  
    dark_dictionary[ifuslot] = [r[0][ifuslot] for r in res]
time_list = [r[1] for r in res]
hum_list = [r[2] for r in res]
temp_list = [r[3] for r in res]
for ifuslot in ifuslots:
    name = 'darkcurrent_model_%s_%s.fits' % (ifuslot, args.outname)
    f  = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(dark_dictionary[ifuslot]),
                       fits.ImageHDU(np.array([t.mjd for t in time_list])),
                       fits.ImageHDU(np.array([t for t in hum_list])),
                       fits.ImageHDU(np.array([t for t in temp_list]))])
    f.writeto(name, overwrite=True)
