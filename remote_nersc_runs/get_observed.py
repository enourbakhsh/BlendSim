from __future__ import print_function

# TODO: I could assign a core to each magnitude band in each cell, meaning 6 cores per cell

## imports
import sys
import numpy as np
import os
import scipy
import datetime
import time
import healpy as hp
import fitsio as fio
import datetime
from itertools import accumulate
import yamlplus as yp
import psutil
from inspect import currentframe
import gc # garbage collector
import resource
from GCR import GCRQuery
import pandas as pd

# https://stackoverflow.com/questions/53037758/leverage-python-f-strings-with-yaml-files
# works with yaml.safe_load
# def format_constructor(loader, node):
# 	return loader.construct_scalar(node).format(**dictionary)
# yaml.SafeLoader.add_constructor(u'tag:yaml.org,2002:str', format_constructor)

# yaml.add_constructor('!join', join)
# yaml.add_constructor('!len', length)

config_file = 'config.yaml'
params = yp.load(open(config_file))
globals().update(params) # converts all of the keys in cfg to global variables to be used everywhere in this code

if parallel:
	try:
		#import nonsense # to provoke python to run in serial
		from mpi4py import MPI
		# for parallel jobs
		comm = MPI.COMM_WORLD
		size = comm.Get_size()
		rank = comm.Get_rank()
		# parallel = True
	except(ImportError):
		print("*** Error in importing/utilizing MPI, all jobs will be executing on a single core...")
		size = 1
		rank = 0
		parallel = False
		# pass
else:
	size = 1
	rank = 0

paths = [f'{package_dir}/gcr-catalogs-buzzard-1.9',
         f'{package_dir}/util',
         f'{package_dir}/PhotoZDC1/src',
         f'{package_dir}/footprinter',
         f'{package_dir}/epsnoise']

for path in paths:
	sys.path.insert(0, path)

import GCRCatalogs
import util
import photErrorModel as ephot
import footprinter as fp
import epsnoise 


# -------------------
# functions
# -------------------

def delete(varname):
	del globals()[varname]
	gc.collect()

def magerr2snr(magerr):
	# dm = 2.5 log10 ( 1 + N/S )
	# N/S = 10^(dm/2.5) - 1
	SNR = 1./(10**(magerr/2.5) - 1)
	return SNR

def radec2pix(ra,dec,pixinfo=None):
	# pixinfo = {'nside': nside, 'nest': nest}
	pixnum = hp.ang2pix(pixinfo['nside'],ra,dec,nest=pixinfo['nest'],lonlat=True) # longitude (~ra) and latitude (~dec) in degrees
	return pixnum.tolist()

def radeclim2cell(ra_lim, dec_lim,num_points=5000,pixinfo=None):
	rand_ra, rand_dec = fp.uniform_rect(ra_lim, dec_lim, num_points=num_points)
	cell_ids_touched = radec2pix(rand_ra,rand_dec,pixinfo=pixinfo)
	return list(set(cell_ids_touched)) # deletes duplicates; won't preserve the order

def sum_cumulative(thelist):
	#return list(np.cumsum(thelist)) # works in py2 as well as py3
	return list(accumulate(thelist)) #only works in py3

def fair_division(nbasket,nball):
	''' fair distribution of jobs b/w available cores for an efficient mpi '''
	basket= [0]*nbasket
	while True:
		for i in range(nbasket):
			basket[i]+=1
			if sum(basket)==nball:
				return [0]+list(sum_cumulative(basket)) # [0]+`a list` (not an array!)
def parse(config_file, **params):    # pass in variable numbers of 
	# use the ** syntax to pass the key/values into a function. This way you don't have to know the keys beforehand
	# https://stackoverflow.com/questions/35590381/input-yaml-config-values-into-keyword-arguments-for-python
	if verbose: print('\nInput parameters loaded from %s:' % config_file)
	if verbose: print('------------------------------------------')
	for key, value in params.items():
		if verbose: print('%s: %s' % (key, value))
	if verbose: print('------------------------------------------\n')

def memory_percent_pid():
	# return the memory usage in percentage like top
	process = psutil.Process(os.getpid())
	memper = process.memory_percent()
	return round(memper,1)

class check_memory: 
	def __init__(self, logfname='check_memory.log'):
		self.logfname = logfname
		if rank==0: # write the header only once
			with open(self.logfname, 'w') as f:
				f.write("<tag> \t line \t core \t time \t total (GB) \t maxrss \t used \t PID_usage \t CPU\n")
	def log(self, tag, linenumber):
		with open(self.logfname, 'a') as f: # using different definitions to interpret them later
			maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss # peak memory usage (kilobytes on Linux, bytes on OS X)
			mem = psutil.virtual_memory()
			cpu_percent = psutil.cpu_percent()
			f.write(f"{tag} \t {linenumber} \t {rank} \t {time.strftime('%H:%M:%S')} \t {mem.total >> 30} \t {maxrss*1000} \t {mem.used >> 30} ({mem.percent}%) \t {memory_percent_pid()}% \t {cpu_percent}%\n")

# https://stackoverflow.com/questions/3056048/filename-and-line-number-of-python-script
def get_linenumber():
	cf = currentframe()
	return cf.f_back.f_lineno


def shearMasked(): # when g>1 # added by Erfan :: eq 3.2 in http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1997A%26A...318..687S&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf
	return ( 1-g_complex[_mask_g]*np.conj(eps_true[_mask_g]) )/( np.conj(eps_true[_mask_g])-np.conj(g_complex[_mask_g]) ) # ! this is not WL limit since g>1 ! it happens very very rarely like 1 per 52 sq. deg. cell, it gave e.g. e=1.08 after shearing in one case; had to fix it bc addNoise chokes with e>=1
	# Any sign convention in teh above ??? cf. eps_sheared = (eps + g)/(1 + eps*conj(g)) in https://github.com/pmelchior/epsnoise/blob/9e556ed5019a01010149fda0a0e9ba0a6226fa84/epsnoise.py#L124

def mask_of_mask(mask,submask):
	if sum(mask) != len(submask):
		raise ValueError('Not the submask of this mask.')
	nth = 0
	mask_final = np.zeros_like(mask)
	for m, mval in enumerate(mask):
		mask_final[m] = True if mval and submask[nth] else False
		if mval: nth+=1
	if sum(mask_final) != sum(submask):
		raise AssertionError('Final mask should have the same number of `True` values as the submask.')
	return mask_final


# https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
# https://www.mikulskibartosz.name/how-to-reduce-memory-usage-in-pandas/
def reduce_df_mem_usage(df, verbose=True):
	start_mem = df.memory_usage().sum() / 1024**2
	for col in df.columns:
	    col_type = df[col].dtype
	    if col_type != object:
	            c_min = df[col].min()
	            c_max = df[col].max()
	            if str(col_type)[:3] == 'int':
	                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
	                    df[col] = df[col].astype(np.int8)
	                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
	                    df[col] = df[col].astype(np.uint8)
	                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
	                    df[col] = df[col].astype(np.int16)
	                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
	                    df[col] = df[col].astype(np.uint16)
	                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
	                    df[col] = df[col].astype(np.int32)
	                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
	                    df[col] = df[col].astype(np.uint32)                    
	                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
	                    df[col] = df[col].astype(np.int64)
	                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
	                    df[col] = df[col].astype(np.uint64)
	            else:
	#                     if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and df[col].equals(df[col].astype(np.float16)):
	#                         df[col] = df[col].astype(np.float16)
	                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: # and df[col].equals(df[col].astype(np.float32)):
	                    df[col] = df[col].astype(np.float32)
	                else:
	                    df[col] = df[col].astype(np.float64)
	end_mem = df.memory_usage().sum() / 1024**2
	if verbose: print(f'Memory usage decreased from {start_mem :5.2f} Mb to {end_mem :5.2f} Mb ({100 * (start_mem - end_mem) / start_mem :.1f}% reduction)')
	return df

# ------------------
# PSF functions
# ------------------

def get_covmat(e1,e2,galhlr): #,already_projected=False): #slow?
	"""
	Returns the covariance matrix of the lensing shears given the two components
	Author: Erfan Nourbakhsh

	[assuming N galaxies in the blended system]

	e1 : the array of the first component of the shape for N galaxies, epsilon ellipticity
	e2 : the array of the second component of the shape for N galaxies, epsilon ellipticity
	"""
	e = np.sqrt(e1**2+e2**2)
	if np.isscalar(e):
		if not np.isfinite(e):
			print(f'Warning: |e| is not finite.\n|e|={e}\ne1={e1}\ne2={e2}')
	else:
		if any(not np.isfinite(e_abs) for e_abs in e):
			print(f'Warning: |e| is not finite.\n|e|={e}\ne1={e1}\ne2={e2}')

	if np.isscalar(e):
		if e>=1:
			print(f'Warning: |e|>=1 which will lead to negative value under sqrt.\n|e|={e}\ne1={e1}\ne2={e2}')
	else:
		if any(e_abs>=1 for e_abs in e):
			print(f'Warning: |e|>=1 which will lead to negative value under sqrt.\n|e|={e}\ne1={e1}\ne2={e2}')

	sigma_round = galhlr/3600/np.sqrt(2.*np.log(2)) # degrees # galhlr = FLUX_RADIUS in arcsec

	# a and b are deviations from a circle of radius r=sigma_round
	a = sigma_round * np.sqrt((1+e)/(1-e)) # wrong: a = sigma_round /(1-gamma)
	b = sigma_round * np.sqrt((1-e)/(1+e)) # wrong: b = sigma_round /(1+gamma)

# merge_blends.py:317: RuntimeWarning: invalid value encountered in sqrt
#   a = sigma_round * np.sqrt((1+e)/(1-e)) # wrong: a = sigma_round /(1-gamma)
# merge_blends.py:318: RuntimeWarning: invalid value encountered in sqrt
#   b = sigma_round * np.sqrt((1-e)/(1+e)) # wrong: b = sigma_round /(1+gamma)
# MAYBE do a cut on e < maxe | yep, I think I solved it with _mask_e? I don't think so; here
# it says under sqrt is negative (or nan mabe)

	theta = 0.5*np.arctan2(e2,e1) # radians
	
	if np.isscalar(theta):
		R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
		Sigma_0 = np.array([[a**2,0],[0,b**2]])
		Sigma   = np.dot(R,np.dot(Sigma_0,R.T))
	else:
		N = theta.size
		R = [np.array([[np.cos(theta[k]),-np.sin(theta[k])],[np.sin(theta[k]),np.cos(theta[k])]]) for k in range(N)]
		Sigma_0 = [np.array([[a[k]**2,0],[0,b[k]**2]]) for k in range(N)]
		Sigma   = [np.dot(R[k],np.dot(Sigma_0[k],R[k].T)) for k in range(N)]
	return np.array(Sigma)


#vget_covmat=np.vectorize(get_covmat)


def Qij(i,j,A,mu,c,Sigma,individual=False,axis=None): #slow?
	"""
	Returns second moments assuming extended gaussian profiles
	Author: Erfan Nourbakhsh

	[assuming N galaxies in the blended system]

	A     :: the array of total NORMALIZED fluxes
	mu    :: the array (N vectors) of galaxy centers (i.e. the peaks of the Gaussians)
	c     :: the vector pointing to the luminosity center of the blended system
	Sigma :: the array of N covariance matrices (2 by 2)
	individual :: True if we are interested in second moments of individual galaxies (non-blends), False sums over everything (good for blending)
	"""

	delta = np.radians(c[1]) # central declination in radians
	cosd = np.cos(delta)
	cf = cosd if i!=j else cosd**2 if i==j==1 else 1.0   
	i,j = i-1, j-1

	if individual:
	    Qij = Sigma.T[i][j]+(mu[i]-c[i])*(mu[j]-c[j])*cf
	else: # sum over galaxies in a blend
	    Qij = np.sum( A*(Sigma.T[:][i][j]+(mu[i]-c[i])*(mu[j]-c[j])*cf), axis=axis )
	    Qij = Qij/np.sum(A, axis=axis)

	return Qij


def convolve_with_PSF(shape1,shape2,galhlr,PSF_FWHM=0):
	if np.isscalar(galhlr):
		if galhlr<=0: raise ValueError("galhlr can't be negative!")
	else:
		if any(gh<=0 for gh in galhlr): raise ValueError("galhlr can't be negative!")
	PSF_size = PSF_FWHM/2 # only true for Gaussians
	Sigma = get_covmat(shape1,shape2,galhlr)
	Sigma_PSF = get_covmat(0,0,PSF_size) # 0, 0 for circular
	# N = A.size
	mu, c = np.array([0,0]), np.array([0,0]) #[np.array([0,0]) for ar_ in range(N)], [np.array([0,0]) for ar_ in range(N)] #np.array([0,0]), np.array([0,0]) 
	A_gal = 1 # it does not affect my Qij calculation since I will normalize and IT IS ONLY ONE OBJECT! we do not have different weights through A
	Q11 = Qij(1,1,A_gal,mu,c,Sigma,individual=True) # A_gal is irrelevant here
	Q22 = Qij(2,2,A_gal,mu,c,Sigma,individual=True)
	Q12 = Qij(1,2,A_gal,mu,c,Sigma,individual=True)
	A_psf = 1 # apparently it does not affect my Qij calculation since I will normalize and IT IS ONLY ONE OBJECT! we do not have different weights through A
	P11 = Qij(1,1,A_psf,mu,c,Sigma_PSF,individual=True) 
	P22 = Qij(2,2,A_psf,mu,c,Sigma_PSF,individual=True)
	P12 = Qij(1,2,A_psf,mu,c,Sigma_PSF,individual=True) # 0 circular!
	# print('P12,P22,P12',P12,P22,P12)    
	Q11 += P11        
	Q22 += P22        
	Q12 += P12       
	# epsilon-ellipticity, not chi-ellipticity!
	e1_sys = (Q11-Q22)/(Q11+Q22+2*(Q11*Q22-Q12**2)**0.5)
	e2_sys = 2.0*Q12/(Q11+Q22+2*(Q11*Q22-Q12**2)**0.5)
	galhlr_sys = 3600*(Q11*Q22-Q12**2)**0.25 * np.sqrt(2.*np.log(2)) # in arcsec 
	return e1_sys, e2_sys, galhlr_sys


# -------------------
# main program
# -------------------

# def get_cells(): # realization: 0, 1, 2 ,3 ,4, 5

# globals celli, cellf, outputfname

if rank==0: parse(config_file, **params)

if log_memory: memory = check_memory(logfname=logfname)

#print('-->>> entered core %d' % rank)
#realization = int(realization)
# celli = None if celli is None else int(celli)
cellf = cellf+1 if cellf is not None else None
# if celli is None and cellf is not None: celli = 0
# if cellf is None and celli is not None: cellf = -1

## adjustments
# nyear = 5
# magcut_i_lsst = 26.0 #24.0 # I went 2 mags fainter because Blendsim might displace some very faint galaxies and combined with other galaxies they can have imag < final_cut or so
# maxNaN = 2

# if rank==0: t0=time.process_time()
t0=time.process_time()

hpinfo = {'nside': nside, 'nest': nest}

# generate randoms such that at least one random position falls into each healpix cell for the iven resolution nside = 8 -> so 1000 is fine
cell_ids_in_a_quarter = radeclim2cell([rai,raf], [deci,decf],num_points=numrand,pixinfo=hpinfo) # Buzzard galaxies are all in a quarter of sky

# find the edge cells to remove them later
cell_ids_in_stripe = radeclim2cell([rai,rai+bezel], [deci,decf],num_points=numrand,pixinfo=hpinfo) 
cell_ids_in_stripe.extend( radeclim2cell([rai,raf], [deci,deci+bezel],num_points=numrand,pixinfo=hpinfo) )
cell_ids_in_stripe.extend( radeclim2cell([raf-bezel,raf], [deci,decf],num_points=numrand,pixinfo=hpinfo) )
cell_ids_in_stripe.extend( radeclim2cell([rai,raf], [decf-bezel,decf],num_points=numrand,pixinfo=hpinfo) )

blacklist = list(set(cell_ids_in_stripe)) # avoid cells that live around the edges

cell_ids = [aa for aa in cell_ids_in_a_quarter if aa not in blacklist]
cell_ids.sort() # to have identical sorted lists for all cores

cell_ids = cell_ids[celli:cellf] #if cellf is None else cell_ids[celli:cellf_+1] # since it should include celli and cellf as well
# print(f"\ncell_ids = {cell_ids}\n")

if log_memory: memory.log('A1', get_linenumber())
## load 'buzzard' catalog
# realizations = ['buzzard_v1.6', 'buzzard_v1.6_1', 'buzzard_v1.6_2', 'buzzard_v1.6_3', 'buzzard_v1.6_5', 'buzzard_v1.6_21']
# catalogs = ['buzzard_v1.9.2', 'buzzard_v1.9.2_1', 'buzzard_v1.9.2_2', 'buzzard_v1.9.2_3'] # and more - 18 realizations
gcr = GCRCatalogs.load_catalog(catalog)
gcr.healpix_pixels = cell_ids # !NEST ordering!
gcr.check_healpix_pixels()
ncell = len(gcr.healpix_pixels)

## some hacks to get lsst magnitudes in older versions of GCRCatalogs:
# _abs_mask_func = lambda x: np.where(x==99.0, np.nan, x + 5 * np.log10(gcr.cosmology.h))
# _mask_func = lambda x: np.where(x==99.0, np.nan, x)
# for i, b in enumerate('ugrizY'):
# 	gcr._quantity_modifiers['Mag_true_{}_lsst_z0'.format(b)] = (_abs_mask_func, 'lsst/AMAG/{}'.format(i))
# 	gcr._quantity_modifiers['mag_true_{}_lsst'.format(b)] = (_mask_func, 'lsst/TMAG/{}'.format(i)) #-----> TMAG to LMAG !!!!

# if ncell % size != 0 the last processor will
# do more than njob_per_core jobs 
njob = ncell # TODO: I can define more jobs by assigning one band per core


if log_memory: memory.log('A2', get_linenumber())

if rank<njob:

	mpi_index_list = fair_division(size,njob) # if parallel else fair_division(1,njob) # last core reserved for the progress bar and saving the outputs
	starting_index = mpi_index_list[rank] if parallel else 0
	ending_index = mpi_index_list[rank+1] if parallel else None #len(cell_ids) # not -1 since it ignores the last one

	if rank==0: # last core in parallel or just the only core in serial
		if size==njob+1:
			print("Warning: %d core is idle and has no job assigned to it. Use maximum %d cores.\n" % (size-njob,njob))
		elif size>njob:
			print("Warning: %d cores are idle and have no jobs assigned to them. Use maximum %d cores.\n" % (size-njob,njob))
		if verbose: print("Total cells (HEALPix pixels): %d" % njob)
		if verbose: print("Total cores to use: %d\nAn optimum number of cores is %d for this configuration.\n\njob #" % (size,njob)) # one core reserved for output
		range_rank = range(size)
		for tempRank in range_rank:
			indi = mpi_index_list[tempRank]
			indf = mpi_index_list[tempRank+1]-1
			if indi==indf:
				arrow = " \t"+str(indf)+"\t-------------> to core "+str(tempRank)
			elif indi>indf:
				arrow = " \tNA"+"\t-------------> to core "+str(tempRank) # no job is avalible for this core
			else:
				arrow = str(indi)+"\t"+str(indf)+"\t-------------> to core "+str(tempRank)
			if verbose: print( arrow.expandtabs(len(str(mpi_index_list[-1]))+1) ) # some formatting hacks
		arrow = "prints"+"\t-------------> to core 0"	
		if verbose: print(arrow)


	# nedded later for PSF convolution as well
	theta_eff = theta_eff_zenith.copy()
	for _key in theta_eff.keys():
		theta_eff[_key] *= airMass**0.6 # to account for the increased column density of air away from zenith (median airMass is a good representative)

	FWHM_gal_median = 2*HLR_gal_median # arcsec, assumes gaussian galaxies (not convolved)

	# if nYrObs is None:
	# 	tvis_total = 30 * ( 10.**((4./5.)*(m5-Cm_shifted[refBand]-0.5*(msky[refBand]-21.)-2.5*np.log10(0.7/np.sqrt(theta_eff[refBand]**2+FWHM_gal_median**2))+km[refBand]*(airMass-1))) )
	# 	# here median of FWHM_gal needed, can't do individually
	# 	nvis_total = tvis_total/tvis
	# 	nYrObs = nvis_total/nVisYr[refBand]
	# 	print(f'set nYrObs to {nYrObs:.2f} to satisfy requested m5={m5}')

	# if rank==0: print(f"\nGetting LSST Y{nYrObs:.2f} observed magnitudes and errors and healing non-detections:")

	# set up LSST error model from PhotozDC1 codebase
	errmodel = ephot.LSSTErrorModel()
	errmodel.setRandomSeed(RandomSeed)

	# new values, May 2018 paper, https://arxiv.org/pdf/0805.2366.pdf
	# errmodel.nYrObs   = nYrObs
	errmodel.nVisYr   = nVisYr
	errmodel.tvis     = tvis
	errmodel.msky     = msky
	errmodel.theta    = theta_eff
	errmodel.gamma    = gamma_LSST
	errmodel.Cm       = Cm_shifted
	errmodel.km       = km
	errmodel.airMass  = airMass
	errmodel.sigmaSys = sigmaSys
	#errmodel.extendedSource=extendedSource # 0.1-0.3 ; I implemented a new method, do no use this!

	def depth2nyear(m5_,return_delta_snr=False): # for point sources by definition
		tvis_total = 30 * ( 10.**((4./5.)*(m5_-Cm_shifted[refBand]-0.5*(msky[refBand]-21.)-2.5*np.log10(0.7/theta_eff[refBand])+km[refBand]*(airMass-1))) )
		nvis_total = tvis_total/tvis
		nYrObs = nvis_total/nVisYr[refBand]
		if return_delta_snr:
			errmodel.nYrObs = nYrObs
			mag_error = errmodel.getMagError(depth['mag'],refBand,FWHM_gal=0) # make sure it is for the point sources
			SNR = 1./(10**(mag_error/2.5) - 1)
			return SNR-depth['nsigma'] #nYrObs, SNR
		else:
			return nYrObs #, SNR

	def nyear2m5(nyear,refBand='LSST_i',FWHM_gal_median=0): # 0 for point sources
		# here median of FWHM_gal needed, can't do individually
		tvis_total = nyear*nVisYr[refBand]*tvis
		return Cm_shifted[refBand] + 0.5*(msky[refBand]-21.) + 2.5*np.log10(0.7/np.sqrt(theta_eff[refBand]**2+FWHM_gal_median**2)) + 1.25*np.log10(tvis_total/30.) - km[refBand]*(airMass-1.)

	def nyear2snr(nyear,at_mag=None,refBand='LSST_i',FWHM_gal_median=0):
		errmodel.nYrObs = nyear
		mag_error = errmodel.getMagError(at_mag,refBand,FWHM_gal=FWHM_gal_median)
		SNR = 1./(10**(mag_error/2.5) - 1)
		return SNR

	if nYrObs is None:
		m5_guess = depth['mag']+2.5*np.log10(depth['nsigma']/5.0)
		m5 = scipy.optimize.brentq(depth2nyear, m5_guess-0.5, m5_guess+0.5, args=(True), maxiter=maxiter)
		nYrObs = depth2nyear(m5)
		m5_ext = nyear2m5(nYrObs,refBand=refBand,FWHM_gal_median=FWHM_gal_median)
		if rank==0 and verbose:
			# achieved_snr = depth2nyear(m5,return_delta_snr=True)+depth['nsigma']
			achieved_snr     = nyear2snr(nYrObs,at_mag=depth['mag'],refBand=refBand,FWHM_gal_median=0) #depth2nyear(m5,return_delta_snr=True)+depth['nsigma']
			achieved_snr_ext = nyear2snr(nYrObs,at_mag=depth['mag'],refBand=refBand,FWHM_gal_median=FWHM_gal_median)
			print(f"Set nYrObs to {nYrObs:.4f} to satisfy:\n{refBand} m5 ~ {m5:.4f} for point sources \t\t\t --> \t {refBand} m5 ~ {m5_ext:.4f} for median sized (FWHM={FWHM_gal_median}) extended sources.\nS/N ~ {achieved_snr:.4f} at {refBand} of {depth['mag']} for point sources \t --> \t S/N ~ {achieved_snr_ext:.4f} at {refBand} of {depth['mag']} for median sized (FWHM={FWHM_gal_median}) extended sources.\n")

	errmodel.nYrObs = nYrObs # do not forget!

	if rank==0 and verbose:
		msg_non_det = 'healing non-detections with S/N=1 values' if heal_non_det else 'flagging non-detections as '+str(flag_non_det)
		print(f"\nGetting LSST Y{nYrObs:.2f} observed magnitudes and errors and {msg_non_det}:")

	def magerr2mag(_magerr,_FWHM_gal=0,_band='LSST_i'):
		quadratic_pars = [gamma_LSST[_band],(0.04-gamma_LSST[_band]),(sigmaSys**2-_magerr**2)*nVisYr[_band]*nYrObs]
		root = np.roots(quadratic_pars)[1] # the 0th root is negative
		m5_ = Cm_shifted[_band] + 0.5*(msky[_band]-21.) + 2.5*np.log10(0.7/np.sqrt(theta_eff[_band]**2+_FWHM_gal**2)) + 1.25*np.log10(tvis/30.) - km[_band]*(airMass-1.)
		return m5_+np.log10(root)/0.4

	# Non-numpy functions like math.sqrt() don't play nicely with numpy arrays
	# So, I had to vectorize the function using numpy to speed it up
	# getObsMag = np.vectorize(errmodel.getObs)
	# getMagError = np.vectorize(errmodel.getMagError)

	# speed-up by making everything numpy-friendly inside the functions
	getObsMag = errmodel.getObs
	getMagError = errmodel.getMagError

	# del errmodel
	gc.collect()


	# def funcsn1(mag,FWHM_gal=0):
	# 	# gives magnitude for the magnitude error where S/N = 1
	# 	# dMag = 2.5 log10 ( 1 + N/S ) = 0.7526
	# 	# _ , magerr = getObsMag(mag,band)-magerr_SN1
	# 	delta_magerr = getMagError(mag,on_band,FWHM_gal=FWHM_gal)-magerr_SN1 #
	# 	return delta_magerr

	# brentq = np.vectorize(scipy.optimize.brentq)

	# mag_SN1 = {}
	# for band in 'ugrizy':
	# 	mag_SN1.update({band: scipy.optimize.brentq(funcsn1, maglim[0], maglim[1], args=(f'LSST_{band}'), maxiter=maxiter)})
	# # final sol: brentq(funcsn1, maglim[0], maglim[1], args=(FWHM_gal), maxiter=maxiter)


	##output_dir = '/global/cscratch1/sd/erfanxyz/projects/blending/buzzard_v1.6_lsst_y1/zsnb/'
	##output_dir = '/global/cscratch1/sd/tyson1/projects/blending/buzzard_v1.9.2_lsst/zsnb/'
	# print('starting_index,ending_index',starting_index,ending_index)
	# print('gcr.healpix_pixels[starting_index:ending_index]', gcr.healpix_pixels[starting_index:ending_index])
	for cell in gcr.healpix_pixels[starting_index:ending_index]:
		#print('cell', cell)
		#print('in1d',np.in1d(cell, np.concatenate([[cell], hp.get_all_neighbours(nside, cell, nest=nest)])))
		#output_fname = output_dir+'zsnb.'+str(cell)+'_r'+str(realization)+'.fit' 
		output_fname = output_fname.format(**locals()) # fstring(output_fname)

		filters = [(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell, 'ra', 'dec')]
		native_filters = [(lambda p: np.in1d(p, np.concatenate([[cell], hp.get_all_neighbours(nside, cell, nest=nest)])), 'healpix_pixel')]

		if log_memory: memory.log('B1', get_linenumber())

		# since Buzzard is split in healpix pixels before lensing is applied, if we need lensed galaxies that are in a specific healpix pixel, we have to load all neighboring pixels
		data = gcr.get_quantities([f'mag_true_{band}_lsst' for band in 'ugrizy']+['shear_1', 'shear_2', 'convergence', 'size'],
		                           filters=filters, native_filters=native_filters, return_iterator=load_one_chunk)

		if load_one_chunk and nrows:
			raise ValueError(f"`load_one_chunk` and `nrows` in {config_file} cannot be applied at the same time.")

		if load_one_chunk: data=next(data)
		if nrows:
			nrows_total = len(data['shear_1'])
			if nrows>nrows_total:
				raise ValueError(f"`nrows` in {config_file} cannot be more than the total row number.")
			_mask_rows = np.array([True]*nrows+[False]*(nrows_total-nrows))
			data = GCRQuery._mask_table(data, _mask_rows)

		imag = data['mag_true_i_lsst']
		imagfinite = np.isfinite(imag)
		imag[~imagfinite] = 99.0 # just a placeholder to avoid runtime warning in the next line
		imagcut = imag < magcut_i_lsst
		data = GCRQuery._mask_table(data, imagcut)

		del imag, imagfinite
		#gc.collect()

		if log_memory: memory.log('B2', get_linenumber())

		mag_true_u_lsst = data['mag_true_u_lsst']
		mag_true_g_lsst = data['mag_true_g_lsst']
		mag_true_r_lsst = data['mag_true_r_lsst']
		mag_true_i_lsst = data['mag_true_i_lsst']
		mag_true_z_lsst = data['mag_true_z_lsst']
		mag_true_y_lsst = data['mag_true_y_lsst']

		gamma1 = data['shear_1']
		gamma2 = data['shear_2']
		kappa  = data['convergence']
		galhlr_lensed_only = data['size']
		# e1_lensed_only = data['ellipticity_1']
		# e2_lensed_only = data['ellipticity_2']

		# # [+] PSF convolution ------------------------------------
		# # get the convolved values, no noise yet
		# e1_lensed_convolved, e2_lensed_convolved, galhlr_lensed_convolved = np.array([convolve_with_PSF(e1_lensed_only[item_],e2_lensed_only[item_],galhlr_lensed_only[item_],PSF_FWHM=theta_eff[refBand]) for item_ in range(GroupSize_candidates)]).T # TODO: vectorization! Didn't work the first time I tried.
		# del e1_lensed_only, e2_lensed_only
		# # [-] PSF convolution -------------------------------------

		# convolved galaxy FWHM is needed to introduce noise to the magnitudes
		FWHM_gal = 2*galhlr_lensed_only # assumes gaussian galaxies | needed later not right here
		# FWHM_gal should not be convolved!!! based on the way getObs(...) function is written

		# FWHM_gal_median = 2*np.median(np.ma.masked_invalid(galhlr_lensed_only)) # no! make it a fixed value for all cells

		notfinite_FWHM_gal_idx = ~np.isfinite(FWHM_gal)  #|(FWHM_gal<=0) nope, if there is any not-finite it says <= is invalid
		nf_fwhmg = sum(notfinite_FWHM_gal_idx)
		if nf_fwhmg>0: print(f'Warning in rank {rank}, cell {cell}: {nf_fwhmg} galaxies have not-finite sizes, they will be filtered and therefore treated as NaN mags in all bands...')
		FWHM_gal[notfinite_FWHM_gal_idx] = FWHM_gal_median # will get rid of them anyway

		bad_FWHM_gal_idx = (FWHM_gal<=galfwhmlim[0])|(FWHM_gal>=galfwhmlim[1])  #|(FWHM_gal<=0) nope, if there is any not-finite it says <= is invalid
		bad_fwhmg = sum(bad_FWHM_gal_idx)
		if bad_fwhmg>0: print(f'Warning in rank {rank}, cell {cell}: {bad_fwhmg} galaxies have abnormal FWHM i.e. not in ({galfwhmlim[0]},{galfwhmlim[1]}) arcsec, they will be thrown out of the sample...')
		FWHM_gal[bad_FWHM_gal_idx] = FWHM_gal_median # will get rid of them anyway


		if log_memory: memory.log('B3', get_linenumber())

		del data

		if log_memory: memory.log('B4', get_linenumber())

		gc.collect() # force the Garbage Collector to release unreferenced memory immediately

		if log_memory: memory.log('B5', get_linenumber())

		gamma = np.sqrt(gamma1**2+gamma2**2)
		# lensed magnitude = true_magnitude - 2.5 * log10(magnification)
		magnification = 1./( (1-kappa)**2-gamma**2  )

		del gamma
		# gc.collect()

		negative_mu_idx = magnification<=0
		nnegmu = sum(negative_mu_idx)
		if nnegmu>0: print(f'Warning in rank {rank}, cell {cell}: {nnegmu} galaxies have mu<=0, throwing them away without even flagging...')
		magnification[negative_mu_idx] = 1.0 # we are going to delete those later anaway

		mag_u_lsst_lensed_only = mag_true_u_lsst - 2.5*np.log10(magnification)
		mag_g_lsst_lensed_only = mag_true_g_lsst - 2.5*np.log10(magnification)
		mag_r_lsst_lensed_only = mag_true_r_lsst - 2.5*np.log10(magnification)
		mag_i_lsst_lensed_only = mag_true_i_lsst - 2.5*np.log10(magnification)
		mag_z_lsst_lensed_only = mag_true_z_lsst - 2.5*np.log10(magnification)
		mag_y_lsst_lensed_only = mag_true_y_lsst - 2.5*np.log10(magnification)
        
        # negative magnifications rarely exist in Buzzard and they give us NaN values for the magnitudes above
        # we have to get rid of them in the following

		if verbose==3: print(f"rank {rank}: first stage done in {datetime.timedelta(seconds=round(time.process_time()-t0))}")

		# u_finite = np.isfinite(mag_u_lsst_lensed_only) #& (~notfinite_FWHM_gal_idx) & (~bad_FWHM_gal_idx) # also filtering galaxies with magnification<=0? : no, wrong, now all mu's are positive, I'll delete them
		# g_finite = np.isfinite(mag_g_lsst_lensed_only) #& (~notfinite_FWHM_gal_idx) & (~bad_FWHM_gal_idx) # will filter them in accepted
		# r_finite = np.isfinite(mag_r_lsst_lensed_only) #& (~notfinite_FWHM_gal_idx) & (~bad_FWHM_gal_idx) 
		# i_finite = np.isfinite(mag_i_lsst_lensed_only) #& (~notfinite_FWHM_gal_idx) & (~bad_FWHM_gal_idx) 
		# z_finite = np.isfinite(mag_z_lsst_lensed_only) #& (~notfinite_FWHM_gal_idx) & (~bad_FWHM_gal_idx) 
		# y_finite = np.isfinite(mag_y_lsst_lensed_only) #& (~notfinite_FWHM_gal_idx) & (~bad_FWHM_gal_idx) 

		# store all mags in a big matrix
		mags_lsst_lensed_only = np.vstack((mag_u_lsst_lensed_only,
		                                   mag_g_lsst_lensed_only,
		                                   mag_r_lsst_lensed_only,
		                                   mag_i_lsst_lensed_only,
		                                   mag_z_lsst_lensed_only,
		                                   mag_y_lsst_lensed_only))

		mags_finite = np.isfinite(mags_lsst_lensed_only)
		mags_lsst_lensed_only[~mags_finite] = 20.0 # does the same thing as the next two lines
		# mags_lsst_lensed_only = np.ma.masked_invalid(mags_lsst_lensed_only) # mask nan and inf
		# mags_lsst_lensed_only = mags_lsst_lensed_only.filled(20.0) # will get rid of them anyway!

		# this will eventually replace -inf, +inf and nan values with 99.0 as a flag for non-detections (i.e. treated like zero or negative flux)
		# u,g,r,i,z,y, eu,eg,er,ei,ez,ey = [np.repeat(99.0,len(mag_u_lsst_lensed_only)) for _ in range(len('ugrizy')*2)]
		# they are all the same size
		# u[u_finite],eu[u_finite] = getObsMag(mag_u_lsst_lensed_only[u_finite],'LSST_u',FWHM_gal=FWHM_gal[u_finite]) # for now we assumed the same galaxy size in all bands but it is not accurate
		# g[g_finite],eg[g_finite] = getObsMag(mag_g_lsst_lensed_only[g_finite],'LSST_g',FWHM_gal=FWHM_gal[g_finite])
		# r[r_finite],er[r_finite] = getObsMag(mag_r_lsst_lensed_only[r_finite],'LSST_r',FWHM_gal=FWHM_gal[r_finite])
		# i[i_finite],ei[i_finite] = getObsMag(mag_i_lsst_lensed_only[i_finite],'LSST_i',FWHM_gal=FWHM_gal[i_finite])
		# z[z_finite],ez[z_finite] = getObsMag(mag_z_lsst_lensed_only[z_finite],'LSST_z',FWHM_gal=FWHM_gal[z_finite])
		# y[y_finite],ey[y_finite] = getObsMag(mag_y_lsst_lensed_only[y_finite],'LSST_y',FWHM_gal=FWHM_gal[y_finite])

		# print('mags_lsst_lensed_only.shape, FWHM_gal.shape =', mags_lsst_lensed_only.shape, FWHM_gal.shape)
		mags_lsst_observed, mags_lsst_error = getObsMag(mags_lsst_lensed_only,'LSST_ugrizy',FWHM_gal=FWHM_gal)
		# for now we assumed the same galaxy size in all bands but it is not accurate, FWHM_gal (ngal)
		# FWHM_gal can be of shape (6,ngal) to account for that
		del mags_lsst_lensed_only

		if verbose==3: print(f"rank {rank}: stage 1.25 done in {datetime.timedelta(seconds=round(time.process_time()-t0))}")

		mags_lsst_observed[~mags_finite] = 99.0
		mags_lsst_error[~mags_finite]    = 99.0

		# zero/negative observed fluxes are flagged as ugrizy magnitudes ~>99 in PhotoZDC1
		#mags_obs_lsst = np.array([u,g,r,i,z,y])
		NaN_index = mags_lsst_observed>=98.9 # True for infinite mags, so it does not ignore them, good!
		nNaN = np.sum(NaN_index, axis=0) # number of NaN bands per object e.g. [0 0 0 ..., 0 2 1]
		del NaN_index

		# print('mags_lsst_observed.shape, mags_lsst_error.shape =',mags_lsst_observed.shape,mags_lsst_error.shape)
		u,g,r,i,z,y = mags_lsst_observed #[mags_lsst_observed[filtObs] for filtObs in range(len('ugrizy'))]
		eu,eg,er,ei,ez,ey = mags_lsst_error #[mags_lsst_error[filtObs] for filtObs in range(len('ugrizy'))]

		del errmodel, mags_lsst_observed, mags_lsst_error
		gc.collect()

		# u[~u_finite],g[~g_finite],r[~r_finite],i[~i_finite],z[~z_finite],y[~y_finite] = [99.0 for _b_ in range(len('ugrizy'))]
		# eu[~u_finite],eg[~g_finite],er[~r_finite],ei[~i_finite],ez[~z_finite],ey[~y_finite] = [99.0 for _b_ in range(len('ugrizy'))]

		if verbose==3: print(f"rank {rank}: stage 1.5 done in {datetime.timedelta(seconds=round(time.process_time()-t0))}")


		# # zero/negative observed fluxes are flagged as ugrizy magnitudes ~>99 in PhotoZDC1
		# #mags_obs_lsst = np.array([u,g,r,i,z,y])
		# NaN_index = np.array([u,g,r,i,z,y])>=98.9 # True for infinite mags, so it does not ignore them, good!
		# nNaN = np.sum(NaN_index, axis=0) # number of NaN bands per object e.g. [0 0 0 ..., 0 2 1]
		
		# del NaN_index
		# gc.collect()

		magerr_SN1 = 2.5*np.log10(2)

		# I could vectorize this for the filters too but it was already super fast
		u[u>=98.9] = magerr2mag(magerr_SN1,_FWHM_gal=FWHM_gal[u>=98.9],_band='LSST_u') if heal_non_det else flag_non_det # mag>98.9 is True for inf mags so it heals them too, if any, good!
		g[g>=98.9] = magerr2mag(magerr_SN1,_FWHM_gal=FWHM_gal[g>=98.9],_band='LSST_g') if heal_non_det else flag_non_det
		r[r>=98.9] = magerr2mag(magerr_SN1,_FWHM_gal=FWHM_gal[r>=98.9],_band='LSST_r') if heal_non_det else flag_non_det
		i[i>=98.9] = magerr2mag(magerr_SN1,_FWHM_gal=FWHM_gal[i>=98.9],_band='LSST_i') if heal_non_det else flag_non_det
		z[z>=98.9] = magerr2mag(magerr_SN1,_FWHM_gal=FWHM_gal[z>=98.9],_band='LSST_z') if heal_non_det else flag_non_det
		y[y>=98.9] = magerr2mag(magerr_SN1,_FWHM_gal=FWHM_gal[y>=98.9],_band='LSST_y') if heal_non_det else flag_non_det

		if verbose==3: print(f"rank {rank}: stage 1.75 done in {datetime.timedelta(seconds=round(time.process_time()-t0))}")

		eu[eu>=98.9] = magerr_SN1 if heal_non_det else flag_non_det
		eg[eg>=98.9] = magerr_SN1 if heal_non_det else flag_non_det
		er[er>=98.9] = magerr_SN1 if heal_non_det else flag_non_det
		ei[ei>=98.9] = magerr_SN1 if heal_non_det else flag_non_det
		ez[ez>=98.9] = magerr_SN1 if heal_non_det else flag_non_det
		ey[ey>=98.9] = magerr_SN1 if heal_non_det else flag_non_det

		if log_memory: memory.log('B6', get_linenumber())

		if maxNaN<6:
			# galaxies with maximum `maxNaN` bands with flagged (i.e. 99) or healed values are accepted
			# limiting magnitude on i is also applied before # no galaxy with i=99.0 will survive
			accepted = (nNaN <= maxNaN) & (~negative_mu_idx) & (~notfinite_FWHM_gal_idx) & (~bad_FWHM_gal_idx) # `i < ...` is False for not finite i values not the other bands
		else:
			accepted = (~negative_mu_idx) & (~notfinite_FWHM_gal_idx) & (~bad_FWHM_gal_idx) # `i < ...` is False for not finite i values not the other bands

		del negative_mu_idx, FWHM_gal, notfinite_FWHM_gal_idx, bad_FWHM_gal_idx
		#del mags_obs_lsst
		#gc.collect()

		mag_u_lsst_lensed_only = mag_u_lsst_lensed_only[accepted]
		mag_g_lsst_lensed_only = mag_g_lsst_lensed_only[accepted]
		mag_r_lsst_lensed_only = mag_r_lsst_lensed_only[accepted]
		mag_i_lsst_lensed_only = mag_i_lsst_lensed_only[accepted]
		mag_z_lsst_lensed_only = mag_z_lsst_lensed_only[accepted]
		mag_y_lsst_lensed_only = mag_y_lsst_lensed_only[accepted]

		u = u[accepted]
		g = g[accepted]
		r = r[accepted]
		i = i[accepted]
		z = z[accepted]
		y = y[accepted]

		eu = eu[accepted]
		eg = eg[accepted]
		er = er[accepted]
		ei = ei[accepted]
		ez = ez[accepted]
		ey = ey[accepted]

		nNaN = nNaN[accepted]

		gamma1 = gamma1[accepted]
		gamma2 = gamma2[accepted]
		kappa  = kappa[accepted]

		galhlr_lensed_only = galhlr_lensed_only[accepted]

		# e1_lensed_convolved = e1_lensed_convolved[accepted]
		# e2_lensed_convolved = e2_lensed_convolved[accepted]
		# galhlr_lensed_convolved = galhlr_lensed_convolved[accepted]

		if verbose==3: print(f"rank {rank}: second stage done in {datetime.timedelta(seconds=round(time.process_time()-t0))}")

		if log_memory: memory.log('B7', get_linenumber())

		# now get some more quantities
		filters = [(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell, 'ra', 'dec')]
		native_filters = [(lambda p: np.in1d(p, np.concatenate([[cell], hp.get_all_neighbours(nside, cell, nest=nest)])), 'healpix_pixel')]
		data = gcr.get_quantities(['galaxy_id', 'redshift_true', 'redshift', 'ra', 'dec', 'size', 'ellipticity_1_true', 'ellipticity_2_true', 'ellipticity_1', 'ellipticity_2', 'Mag_true_g_des_z01', 'Mag_true_r_des_z01'],
		                           filters=filters, native_filters=native_filters, return_iterator=load_one_chunk)

		# if load_one_chunk: data=next(data) # will do it directly below

		if log_memory: memory.log('B8', get_linenumber())

		if nrows:
			imagcut = mask_of_mask(_mask_rows, imagcut)
			del _mask_rows

		accepted = mask_of_mask(imagcut, accepted)
		# if nrows: accepted = mask_of_mask(_mask_rows, accepted) # old one w/o imagcut

		data = GCRQuery._mask_table(next(data), accepted) if load_one_chunk else GCRQuery._mask_table(data, accepted)
		
		del imagcut
		# gc.collect()

		if log_memory: memory.log('B9', get_linenumber())

		galaxy_id = np.int64(data['galaxy_id']).astype('U11')
		redshift_true = data['redshift_true']
		zs = data['redshift']
		ra = data['ra']
		dec = data['dec']
		e1_true = data['ellipticity_1_true']
		e2_true = data['ellipticity_2_true']
		e1_lensed_only = data['ellipticity_1']
		e2_lensed_only = data['ellipticity_2']
		Mag_true_g_des_z01 = data['Mag_true_g_des_z01']
		Mag_true_r_des_z01 = data['Mag_true_r_des_z01']
		galhlr_lensed_only_copy = data['size'] # we load it just to check if they match with the old one (important)

		if not np.array_equal(galhlr_lensed_only, galhlr_lensed_only_copy):
			raise AssertionError('A fatal mismatch in quantities after masking.')

		if verbose==3: print(f'rank {rank}, cell {cell}: number of galaxies: {len(ra)}')
		if log_memory: memory.log('B10', get_linenumber())

		del data, galhlr_lensed_only_copy

		# theta_eff = theta_eff_zenith.copy()
		# for _key in theta_eff.keys():
		# 	theta_eff[_key] *= airMass**0.6 # to account for the increased column density of air away from zenith (median airMass is a good representative)

		# [+] PSF convolution ------------------------------------
		# get the convolved values, no noise yet
		e1_lensed_convolved, e2_lensed_convolved, galhlr_lensed_convolved = convolve_with_PSF(e1_lensed_only,e2_lensed_only,galhlr_lensed_only,PSF_FWHM=theta_eff[refBand])  
		del e1_lensed_only, e2_lensed_only, galhlr_lensed_only
		# [-] PSF convolution ------------------------------------

		if log_memory: memory.log('B11', get_linenumber())

		gc.collect()

		if log_memory: memory.log('B12', get_linenumber())

		Color_true_gr_des_z01 = Mag_true_g_des_z01 - Mag_true_r_des_z01
		# g1, g2 = gamma1/(1-kappa), gamma2/(1-kappa) # assuming shear_{1,2} is not the reduced shear!
		g_complex = gamma1/(1-kappa) + 1j * gamma2/(1-kappa)

		# # - - - Here I use COSMOS ellipticity distribution to assign realistic ellipticities myself

		# # mask = np.isfinite(Color_true_gr_des_z01)
		numrows = sum(accepted)
		del accepted

		red = eval(red_cut) #(Color_true_gr_des_z01 > red_coeffs[0] - red_coeffs[1] * Mag_true_r_des_z01)
		# numred = sum(red)
		# numblue = sum(~red)
		# numrb = [numred,numblue] 
		#eps_noisy = np.zeros(numrows, dtype=complex)
		eps_lensed_convolved = e1_lensed_convolved + 1j*e2_lensed_convolved
		# shape_noise = np.zeros(numred+numblue, dtype=float)

		# # Johnson SB model parameters for red (index=0) and blue (index=1) population:
		# JSB_a = [1.91567133, -0.13654868] 
		# JSB_b = [1.10292468, 0.94995499 ]
		# JSB_loc = [0.00399965, -0.02860337]

		# for rb, gtype in enumerate([red,~red]):
		# 	eps_intrinsic = epsnoise.sampleEllipticity(numrb[rb], model='JohnsonSB', a=JSB_a[rb], b=JSB_b[rb], loc=JSB_loc[rb]) # I added JohnsonSB model to epsnoise
		# 	ok = np.isfinite(g_complex) & np.isfinite(g_complex)
		# 	print('rb',rb,'bad_e g_complex values :',sum(~ok))
		# 	print('rb',rb,'min max g_complex :', np.min(g_complex),np.max(g_complex))
		# 	eps_lensed_only[gtype] = epsnoise.addShear(eps_intrinsic, g_complex[gtype])
		# 	nu = magerr2snr(ei[gtype]) # nu: significance of image i.e. S/N ratio based on imag_err
		# 	try:
		# 		eps_noisy[gtype] = epsnoise.addNoise(eps_lensed_only[gtype], nu, True) # I fixed this function in epsnoise to get rid of an excess number of galaxies arounf e~1 but Peter said you better remove them
		# 	except Exception as exc:
		# 		print('Failure in rank',rank,'cell',cell,'rb',rb)
		# 		raise exc

		nu = magerr2snr(ei) # nu: significance of image i.e. S/N ratio based on imag_err

		_mask_g = np.abs(g_complex) > 1 
		if sum(_mask_g)>0: print(f'Warning in rank {rank}, cell {cell}: {sum(_mask_g)} bad reduced shear, g_complex \n |g|[{np.where(_mask_g)[0]}] = {np.abs(g_complex)[_mask_g]} \n will take care of it...')

		eps_true = e1_true + 1j* e2_true
		eps_lensed_convolved[_mask_g] = shearMasked() # ! this is not WL limit since g>1 ! it happens very very rarely; addNoise chokes with e>=1

		# after g>1 correction, let's see if there's still a problem?
		_mask_e = (np.abs(eps_lensed_convolved) > maxe) | (~ np.isfinite(eps_lensed_convolved))
		if sum(_mask_e)>0: print(f'Warning in rank {rank}, cell {cell}: {sum(_mask_e)} bad eps_lensed_convolved \n |e|[{np.where(_mask_e)[0]}] = {np.abs(eps_lensed_convolved)[_mask_e]} \n will take care of it...')
		eps_lensed_convolved[_mask_e] = 0 + 1j * 0 # just a place-holder not to choke epsnoise

		eps_noisy = epsnoise.addNoise(eps_lensed_convolved, nu, transform_eps=transform_eps) #, rseed=cell+1958) # rseed just for completeness # since we pass the data as a whole to epsnoise, the initial seed at the top of epsnoise is enough to generate reproducible results # I fixed this function in epsnoise to get rid of an excess number of galaxies around e~1 but Peter said you better remove them

		# replace problematic shapes with (sheared+convolved)-only shapes without noise provided by buzzard to be flagged as a bad shapes later
		eps_noisy[_mask_e] = eps_lensed_convolved[_mask_e]

		del nu, g_complex
		#gc.collect()

		# e1_lensed_only = np.real(eps_lensed_only)
		# e2_lensed_only = np.imag(eps_lensed_only)

		# good = np.abs(eps_noisy) <= 0.999999999999 # not actually needed, just to print
		# bad_e = np.abs(eps_noisy) > 0.999999999999
		# print('rank %i: %f percent of shapes is useless' % (rank,100.*sum(bad_e)/len(eps_noisy)) )

		e1 = np.real(eps_noisy) # final observed
		e2 = np.imag(eps_noisy)

		# shape_noise_ = {'red':np.std(eps_noisy[(red)&(~bad_e)]), 'blue':np.std(eps_noisy[(~red)&(~bad_e)])}
		# print('sn red, blue',shape_noise_['red'],shape_noise_['blue'])

		# shape_noise[red] = shape_noise_['red'] 
		# shape_noise[~red] = shape_noise_['blue'] 

		# print('g_complex.shape', g_complex.shape)
		# print('num good',sum(good),'num bad_e',sum(bad_e),'N-bad_e',numred+numblue-sum(bad_e),'N-good',numred+numblue-sum(good))
		# print('eps_lensed_only.shape', eps_lensed_only.shape)
		# print('eps_noisy.shape', eps_noisy.shape)
		de1 = np.abs( e1_lensed_convolved - e1 ) # np.abs( np.real(eps_lensed_only) - e1 )
		de2 = np.abs( e2_lensed_convolved - e2 )  #np.abs( np.imag(eps_lensed_only) - e2 )
		de = np.abs( eps_lensed_convolved - eps_noisy ) #np.abs( np.abs(eps_lensed_only) - np.abs(eps_noisy) )
		bad_e = np.abs(eps_noisy) > maxe
		if sum(bad_e):
			print(f'Warning in rank {rank}, cell {cell}: {100.*sum(bad_e)/len(eps_noisy):.2f} percent of shapes have e > {maxe}, i.e. useless!')

		#shape_weight = 1. / (shape_noise**2 + de**2)

		del eps_true, eps_lensed_convolved
		#gc.collect()

		red = red.astype(np.int8) # {1: red, 0: blue}
		#bad_e = bad_e.astype(int)
		bad_e = bad_e + _mask_g*2 + _mask_e*10 # {0: useful shape, 1: useless shape with e>1, 2: e<1 but sheared with g>1 not WL, 3: 1 & 2 happened at the same time, 10: sheared-only e is still illegal after all the corrections, >10: 10 combined with 1 and/or 2}

		del _mask_g, _mask_e
		#gc.collect()

		shape_noise = np.std(eps_noisy[~bad_e.astype(bool)]) # exclude problematic shapes in this calculation
		del eps_noisy
		# gc.collect()

		# - - - end reassigning ellipticities

		# ug,gr,ri,iz,zy = u-g, g-r, r-i, i-z, z-y
		# eug = np.sqrt(eu**2+eg**2)
		# egr = np.sqrt(eg**2+er**2)
		# eri = np.sqrt(er**2+ei**2)
		# eiz = np.sqrt(ei**2+ez**2)
		# ezy = np.sqrt(ez**2+ey**2)

		if verbose==3: print(f"rank {rank}: third stage done in {datetime.timedelta(seconds=round(time.process_time()-t0))}. Waiting to write the {save_format} file... the columns have the shape: {galaxy_id.shape}")

		if log_memory: memory.log('C1', get_linenumber())

		# 'f' is the shorthand for 'float32'. 'f4' also means 'float32' because it has 4 bytes and each byte has 8 bits.
		# Similarly, 'f8' means 'float64' because 8*8 = 64. For the difference between '>f4' and ' <f4 ',
		# it is related to how the 32 bits are stored in 4 bytes
		# ('>')Big Endian Byte Order: The most significant byte (the "big end") of the data is placed at the byte 
		# with the lowest address. The rest of the data is placed in order in the next three bytes in memory.
		# ('<')Little Endian Byte Order: The least significant byte (the "little end") of the data is placed
		# at the byte with the lowest address. The rest of the data is placed in order in the next three bytes in memory.

		# _keys = ['id','u_lensed_only','g_lensed_only','r_lensed_only','i_lensed_only','z_lensed_only','y_lensed_only',
		#  'u','g','r','i','z','y',
		#  'eu','eg','er','ei','ez','ey',
		#  'redshift_true','zs','ra','dec','galhlr','galhlr_lensed_convolved','gamma1','gamma2','kappa','e1','e2',
		#  'delta_e','delta_e1','delta_e2','e1_lensed_convolved','e2_lensed_convolved','nNaN','isred','bad_shape']

		_values = [galaxy_id, mag_u_lsst_lensed_only,mag_g_lsst_lensed_only,mag_r_lsst_lensed_only,
		 mag_i_lsst_lensed_only, mag_z_lsst_lensed_only,mag_y_lsst_lensed_only,
		 u,g,r,i,z,y, eu,eg,er,ei,ez,ey, 
		 redshift_true,zs, ra, dec, galhlr_lensed_convolved, gamma1, gamma2, kappa,
		 e1, e2, de, de1, de2, e1_lensed_convolved, e2_lensed_convolved, nNaN, red, bad_e]

		type_dict = {'id':'U11','u_lensed_only':np.float32,'g_lensed_only':np.float32,'r_lensed_only':np.float32,'i_lensed_only':np.float32,'z_lensed_only':np.float32,'y_lensed_only':np.float32,
		 'u':np.float32,'g':np.float32,'r':np.float32,'i':np.float32,'z':np.float32,'y':np.float32,
		 'eu':np.float32,'eg':np.float32,'er':np.float32,'ei':np.float32,'ez':np.float32,'ey':np.float32,
		 'redshift_true':np.float32,'zs':np.float32,'ra':np.float64,'dec':np.float64,'galhlr_lensed_convolved':np.float32,'gamma1':np.float32,'gamma2':np.float32,'kappa':np.float32,'e1':np.float32,'e2':np.float32,
		 'delta_e':np.float32,'delta_e1':np.float32,'delta_e2':np.float32,'e1_lensed_convolved':np.float32,'e2_lensed_convolved':np.float32,'nNaN':np.uint8,'isred':np.uint8,'bad_shape':np.uint8}

		*_keys, = type_dict

		del galaxy_id, mag_u_lsst_lensed_only,mag_g_lsst_lensed_only,mag_r_lsst_lensed_only, \
		    mag_i_lsst_lensed_only, mag_z_lsst_lensed_only,mag_y_lsst_lensed_only, \
		    u,g,r,i,z,y, eu,eg,er,ei,ez,ey, \
		    redshift_true,zs, ra, dec, galhlr_lensed_convolved, gamma1, gamma2, kappa, \
		    e1, e2, de, de1, de2, e1_lensed_convolved, e2_lensed_convolved, nNaN, red, bad_e
		# del galhlr
		
		# ugriz, e1,e2, galhlr with their errors --> are all observed: lensed+convolved+noisy
		# dt = np.dtype('u1'); dt.name # 'uint8' # a11 --> bytes88

		if save_format=='fit':
			util.gen_write(output_fname,_keys,_values,dtypes=list(type_dict.values())) # u1: 0 to 255 (unsigned integer)
			# util.gen_write(output_fname,_keys,_values,dtypes=['U11']+(len(_keys)-3)*['f4']+3*['u1']) # u1: 0 to 255 (unsigned integer)
		else:
			df_dict = dict(zip(_keys, _values))
			df = pd.DataFrame.from_dict(df_dict).astype(type_dict)
			del df_dict
			if reduce_mem_usage: df = reduce_df_mem_usage(df)
			getattr(df, f'to_{save_format}')(output_fname)
			if rank==0 and verbose:
				df_mem = df.memory_usage().sum() / 1024**2
				print(f'The typical size of the dataframes: {df_mem :5.2f} Mb')
				print(f'As an example, the dataframe for rank {rank} is stored in {output_fname}')
			del df

		if verbose==3: print(f"Core {rank} is done writing the output catalog ({save_format} format) in {datetime.timedelta(seconds=round(time.process_time()-t0))}")
		
		del _keys, _values

		# gc.collect()

		# # append some more info -- Cori got stuck here and never finished # why don't I put this in gen_write?
		# fits = fio.FITS(output_fname, mode='rw', clobber=False) #, extname='catalog')
		# created = datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S')
		# # fits.write(data)
		# fits[1].write_history(f'Created from {catalog} by Erfan Nourbakhsh on {created}')
		# for comment in comments:
		# 	fits[1].write_comment(comment.format(**locals())) # 'Shape noise: {shape_noise}'
		# if log_memory: memory.log('C2', get_linenumber())
		# fits.close() # del fits?
		# if verbose==3: print(f"Core {rank} is done adding the comments to the catalog in {datetime.timedelta(seconds=round(time.process_time()-t0))}")

		if log_memory: memory.log('C3', get_linenumber())

if rank==0 and verbose: print(f"Waiting for all processors to write their outputs... so far {datetime.timedelta(seconds=round(time.process_time()-t0))} passed.")

# when we use MPI_barrier() for synchronization, we are guaranteed that no process 
# will leave the barrier until all processes have entered it
if parallel: comm.Barrier()
time.sleep(2)

#END: This need to print after all MPI_Send/MPI_Recv has been completed
if rank==0: print( f"\nSuccess! All done! Elapsed time: {datetime.timedelta(seconds=round(time.process_time()-t0))}" ) # time is underestimated?
# os.system("echo -e 'attached are out and err files (inetrnally from the python code)' | mailx -s 'Batch job COMPLETED' -a get_observed_0_0_r3.out -a get_observed_0_0_r3.err 'erfan@ucdavis.edu'")



# if __name__== "__main__":
# 	get_cells() # realization: 0, 1, 2 ,3 ,4, 5
