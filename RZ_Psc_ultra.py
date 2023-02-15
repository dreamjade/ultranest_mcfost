#!/usr/bin/env python
# coding: utf-8

import mcfost
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterExponent
from numpy import unravel_index
import astropy.units as unit
import pysynphot
import matplotlib.ticker as mticker
import os
from scipy.interpolate import CubicSpline
from scipy.stats import norm
import time
import random
from multiprocessing import Pool
import signal
import shutil
from pathlib import Path
import ultranest
Path("/tmp/mcmc").mkdir(parents=True, exist_ok=True)

#obs_data
phot_wav=np.array([1.2375,1.662,2.159])
phot_flux=np.array([132,119,87.7])
visir_wav=np.array([8.30301,9.00134,9.50446,10.005,10.5013,11.0053,11.5002,12.0027,12.5017,12.797])
visir_flux=np.array([0.0839141,0.136693,0.203797,0.212987,0.207156,0.203014,0.163458,0.121283,0.0925295,0.105856])*1000
v_err=np.array([0.00839141,0.0136693,0.0203797,0.0212988,0.0207156,0.0203014,0.0163458,0.0121283,0.00925295,0.0105856])*1000
Wise_wav=np.array([3.4, 4.6, 12, 22])
Wise_flux=np.array([56.7436,47.7518,107.821,83.6299])
w_err=np.array([1.18940,0.914726,1.57726,2.12915])
Alma_flux=np.array([41.6*10**-3])
Alma_wav=1260
a_err=np.array([8*10**-3])

#record best
best_sol_dir='best_sol.npy'
#best_sol = np.array([-np.inf,0,0,0,0,0,0,0])
#np.save(best_sol_dir, best_sol)

def lnlike(theta, original=False):
    inc, surfd_exp, r_in, dr, log_dust_mass, scale_height, aexp = theta
    #surfd_exp, log_dust_mass, scale_height, aexp = theta
    #read paramfile
    file_dir='RZ_psc/'
    par = mcfost.Paramfile(file_dir+'RZ_psc.para')
    
    #change parameters
    par.RT_imin = inc #40~90
    par.RT_imax = inc #40~90
    #par.density_zones[0]['flaring_exp'] = flar_exp #0.0
    par.density_zones[0]['surface_density_exp'] = surfd_exp #-2.5~0.0
    par.density_zones[0]['r_in'] = r_in #0~2
    par.density_zones[0]['r_out'] = r_in+dr #r_in+0.1~10.0
    par.density_zones[0]['dust_mass'] = 10**log_dust_mass #-10.5 -7.5
    par.density_zones[0]['scale_height'] = scale_height #0.01~1.5
    par.density_zones[0]['dust'][0]['aexp'] = aexp #2~5
    
    #make_dir to write new paramfile
    file_dir = '/tmp/'
    rint = str(random.randint(0,9999999999999))
    os.mkdir(file_dir+'mcmc/'+rint)
    file_dir = file_dir+'mcmc/'+rint+'/'
    par.writeto(file_dir+'RZ_psc.para', log_show=False)
    par = mcfost.Paramfile(file_dir+'RZ_psc.para')
    
    #run mcfost
    mcfost.run_one_file(file_dir+'RZ_psc.para', wavelengths=[], move_to_subdir=False, log_show=False, timeout = 300)

    #IR Excess
    filename=file_dir+'data_th/sed_rt.fits.gz'
    os.system("gunzip {}".format(filename))

    try:sed_model=fits.open(file_dir+'data_th/sed_rt.fits')
    except:return -999999999.0
    wav=sed_model[1].data

    #total result
    modelt_flux=sed_model[0].data[0]
    modeltotal = pysynphot.ArraySpectrum(wav, modelt_flux.flatten(), name='total-model',fluxunits='flam',waveunits='microns')
    c=2.99792*10**14
    mt_jy=10**26*modeltotal.flux*modeltotal.wave*1000/c

    #likelihood
    mt_jy_cs = CubicSpline(wav, mt_jy)
    try:output = (-np.sum(((visir_flux-mt_jy_cs(visir_wav))/v_err)**2)-np.sum(((Wise_flux-mt_jy_cs(Wise_wav))/w_err)**2)-np.sum(((Alma_flux-mt_jy_cs(Alma_wav))/a_err)**2))/2 #chi^2/2
    except:return -999999999.0
    shutil.rmtree(file_dir, ignore_errors=True)
    global best_sol_dir
    best_sol = np.load(best_sol_dir,)
    if output>best_sol[0]:
        best_sol[0]=output
        best_sol[1:]=theta
        print(best_sol)
        np.save(best_sol_dir, best_sol)
        par.writeto('RZ_psc_ultra.para', log_show=False)
    return(output)

def prior_transform(unit_cube):
    inc = 65.8 + unit_cube[0]*6.0
    surfd_exp = -2.03 + unit_cube[1]*0.1
    r_in = 0.0 + unit_cube[2]*0.13
    dr = 7.2 + unit_cube[3]*0.52
    log_dust_mass = -8.8 + unit_cube[4]*0.80
    scale_height = 1.2 + unit_cube[5]*0.28
    aexp  = 3.2 + unit_cube[6]*0.16
    return inc, surfd_exp, r_in, dr, log_dust_mass, scale_height, aexp

#run ultranest
import os
os.environ["OMP_NUM_THREADS"] = "32"
param_names = ["inc", "surfd_exp","r_in","dr","log_dust_mass", "scale_height", "aexp"]
sampler = ultranest.ReactiveNestedSampler(
    param_names,
    lnlike, 
    prior_transform,
    log_dir='ultra_0215',
    resume='resume')
result = sampler.run(min_num_live_points=100, max_num_improvement_loops=0)
sampler.print_results()

#cornerplot
from ultranest.plot import cornerplot
cornerplot(result)
sampler.plot_corner()
