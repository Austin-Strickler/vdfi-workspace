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
from scipy.signal import medfilt

def qe_from_ldls(self, ifuslot, niter=3, filter_length=11,
                       sigma=5, badcolumnthresh=300):
    '''
    

    Parameters
    ----------
    niter : TYPE, optional
        DESCRIPTION. The default is 3.
    filter_length : TYPE, optional
        DESCRIPTION. The default is 11.
    sigma : TYPE, optional
        DESCRIPTION. The default is 5.
    badcolumnthresh : TYPE, optional
        DESCRIPTION. The default is 300.

    Returns
    -------
    None.

    '''
    image = self.info[ifuslot].data
    qe = image * 0.
    for ind in np.arange(image.shape[0]):
        y = image[ind] * 1.
        for i in np.arange(niter):
            m = medfilt(y, filter_length)
            dev = (image[ind] - m) / np.sqrt(np.where(m<25, 25, m))
            flag = np.abs(dev) > sigma
            y[flag] = m[flag]
        qe[ind] = (image[ind] - m) / m 
    self.info[ifuslot].qe = qe

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

ldls_obs = [key for key, value in zip(keys, values) if value == 'ldls_long']

ldls_list = ldls_obs

def get_qe(ldls, bins=25):
    date = ldls[:8]
    obs = int(ldls[8:15])
    exp = int(ldls[15:])
    virus = VIRUSRaw(date, obs, h5table, basepath=basedir, exposure_number=exp,
                     ifuslots=ifuslots)
    ldls_dictionary = {}
    for ifuslot in ifuslots:
        try:
            qe_from_ldls(virus, ifuslot)
            ldls_dictionary[ifuslot] = virus.info[ifuslot].qe
            virus.log.warning('Failed to get QE for %s' % ifuslot)
        except:
            ldls_dictionary[ifuslot] = np.zeros((448, 1036))
    timeobs = Time(virus.info[ifuslot].header['DATE'])
    virus.log.info('Shifts finished %s_%07d_exp%02d' % (date, obs, exp))
    return ldls_dictionary, timeobs

P = Pool(16)
res = P.map(get_qe, ldls_list)
P.close()
ldls_dictionary = {}
for ifuslot in ifuslots:
    ldls_dictionary[ifuslot] = np.nan * np.ones((len(ldls_list), 448, 1036))
for ifuslot in ifuslots:  
    ldls_dictionary[ifuslot] = [r[0][ifuslot] for r in res]
time_list = [r[1] for r in res]
for ifuslot in ifuslots:
    name = 'badpixels_model_%s_%s.fits' % (ifuslot, args.outname)
    f  = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(ldls_dictionary[ifuslot]),
                       fits.ImageHDU(np.array([t.mjd for t in time_list]))])
    f.writeto(name, overwrite=True)
