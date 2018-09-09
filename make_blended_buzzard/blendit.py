import sys

## adding a rich env
sys.path.insert(0, '/global/common/software/lsst/common/miniconda/py3-4.2.12/lib/python3.6/site-packages')

# sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/footprinter')
# import footprinter as fp

import numpy as np
import fitsio as fio
import time
import datetime
from itertools import accumulate, chain
from astropy.coordinates import SkyCoord, Angle, search_around_sky
from numpy.lib.recfunctions import append_fields
import pandas as pd
#import numba
import six 
import os
import multiprocessing
import dask.dataframe as dd # memory manager / helps avoid oom error / TODO: use dask more extensively
# from dask.diagnostics import ProgressBar
from dask.multiprocessing import get as dget
import scipy

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/Jupytermeter') # loading bar
from jupytermeter import *

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/smallestenclosingcircle')
import smallestenclosingcircle as smec

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/PhotoZDC1/src') # LSST error model
import photErrorModel as ephot

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/epsnoise') # my version of epsnoise (I edited and added some functions)
import epsnoise 


try:
	# import nonesense
	import mpi4py
	# https://bitbucket.org/mpi4py/mpi4py/issues/54/example-mpi4py-code-not-working
	mpi4py.rc.recv_mprobe = False # otherwise it won't complete the recv for large data
	from mpi4py import MPI
	# for parallel jobs
	COMM = MPI.COMM_WORLD
	size = COMM.Get_size()
	rank = COMM.Get_rank()
	#status = MPI.Status()
	parallel = True
except(ImportError):
	print("*** Error in importing/utilizing MPI, all jobs will be executing on a single core...")
	size = 1
	rank = 0
	parallel = False
	pass


# ------------
# functions
# ------------

def usedir(mydir):
# https://stackoverflow.com/questions/12468022/python-fileexists-error-when-making-directory	
	if not mydir.endswith('/'): mydir += '/' # important
	try:
		if not os.path.exists(os.path.dirname(mydir)):
			os.makedirs(os.path.dirname(mydir)) # sometimes b/w the line above and this line another core already made this directory and it leads to [Errno 17]
			print('-- made dir:',mydir)
	except OSError as err:
		pass # print('[handled error]',err)


def _apply_df(args): # 20X slower than _apply_df_gbf
	df, groupby, dask, func = args
	if dask:
		print('dask groupby apply started')
		return df.groupby(groupby).apply(func).compute() # dask idea: https://stackoverflow.com/questions/50051210/avoiding-memory-issues-for-groupby-on-large-pandas-dataframe
	else:
		return df.groupby(groupby).apply(func)

def _filter_df_gbf(args):
	df, groupby, limiting_mag_i_lsst = args
	return df.groupby(groupby).filter(lambda q: -2.5*np.log10( np.sum(10**(-0.4*q['i']) ) ) <= limiting_mag_i_lsst) # takes ~ 12 min (some of them 20 min??)

# implemented thanks to: https://gist.github.com/yong27/7869662
# only worked for apply - not for groupby.apply so I had to edit it (groupby obj -> df) and lots of other modifications in what to split etc.
# if order matters: https://gist.github.com/tejaslodaya/562a8f71dc62264a04572770375f4bba
# also look at a different one https://github.com/josepm/MP_Pandas/blob/master/mp_generic.py
def apply_by_multiprocessing(df, brange, groupby, dask, workers, func, **kwargs):
	if dask[0]: npartitions = dask[1]
	pool = multiprocessing.Pool(processes=workers)
	branges = np.array_split(brange, workers)
	dfs = [df.loc[df[groupby].astype(int).isin(br)] for br in branges]
	del df
	if dask[0]: dfs = [dd.from_pandas(d,npartitions=dask[1]) for d in dfs]
	result = pool.map(_apply_df, [(d, groupby, dask[0], func) for d in dfs]) 
	# https://stackoverflow.com/questions/26784556/pandas-python-np-array-splitdf-x-throws-an-error-dataframe-object-has-no
	pool.close()
	return pd.concat(list(result))

def filter_by_multiprocessing(df, filter_func, brange, groupby, limiting_mag_i_lsst, **kwargs):
	workers = kwargs.pop('workers')
	pool = multiprocessing.Pool(processes=workers)
	branges = np.array_split(brange, workers)
	dfs = [df.loc[df[groupby].astype(int).isin(br)] for br in branges]
	del df
	result = pool.map(filter_func, [(d, groupby, limiting_mag_i_lsst) for d in dfs]) 
	pool.close()
	return pd.concat(list(result))


def Qij_points(i,j,A,mu,c,axis=None):
	"""
    Returns second moments assuming point sources
    Author: Erfan Nourbakhsh
    
    [assuming N galaxies in the blended system]

    A     :: the array of total fluxes
    mu    :: the array (N vectors) of galaxy centers (i.e. the peaks of the Gaussians)
    c     :: the vector pointing to the luminosity center of the blended system
    """

	delta = np.radians(c[1]) # central declination in radians
	cosd = np.cos(delta)
	cf = cosd if i!=j else cosd**2 if i==j==1 else 1.0        
	i,j = i-1, j-1
	Qij = np.sum( A*((mu[i]-c[i])*(mu[j]-c[j])*cf), axis=axis )
	Qij = Qij/np.sum(A, axis=axis)

	return Qij

def Qij(i,j,A,mu,c,Sigma,axis=None): #slow?
	"""
	Returns second moments assuming extended gaussian profiles
	Author: Erfan Nourbakhsh

	[assuming N galaxies in the blended system]

	A     :: the array of total fluxes
	mu    :: the array (N vectors) of galaxy centers (i.e. the peaks of the Gaussians)
	c     :: the vector pointing to the luminosity center of the blended system
	Sigma :: the array of N covariance matrices (2 by 2)
	"""
	
	delta = np.radians(c[1]) # central declination in radians
	cosd = np.cos(delta)
	cf = cosd if i!=j else cosd**2 if i==j==1 else 1.0   
	i,j = i-1, j-1
	Qij = np.sum( A*(Sigma.T[:][i][j]+(mu[i]-c[i])*(mu[j]-c[j])*cf), axis=axis )
	Qij = Qij/np.sum(A, axis=axis)

	return Qij

def get_covmat(gamma1,gamma2,kappa,gsize, reduced=True): #slow?
	"""
	Returns the covariance matrix of the lensing shears given the two components
	Author: Erfan Nourbakhsh

	[assuming N galaxies in the blended system]

	gamma1 : the array of the first component of the shears for N galaxies 
	gamma2 : the array of the second component of the shears for N galaxies        
	"""

	# http://gravitationallensing.pbworks.com/w/page/15553259/Weak%20Lensing
	gamma = np.sqrt(gamma1**2+gamma2**2)

	# \text{FWHM}=2\sqrt{2\sigma^2\ln 2}=\sigma\sqrt{8\ln 2}
	FWHM = 2.*gsize/3600 # gsize = FLUX_RADIUS in arcsec
	sigma_round = FWHM/np.sqrt(8.*np.log(2)) # degrees

	# Acting on a circular background source with radius {\displaystyle R~} R~, lensing generates an ellipse with major and minor axes
	# {\displaystyle a={\frac {R}{1-\kappa -\gamma }}} a={\frac {R}{1-\kappa -\gamma }}
	# {\displaystyle b={\frac {R}{1-\kappa +\gamma }}} b={\frac {R}{1-\kappa +\gamma }}

	kappa = 1.*kappa if reduced is True else 0 # not sure!!!
	# a and b are deviations from a circle of radius r=sigma_round
	a = sigma_round/(1-kappa-gamma)
	b = sigma_round/(1-kappa+gamma)

	theta = 0.5*np.arctan2(gamma2,gamma1) # radians
	N = theta.size
	R = [np.array([[np.cos(theta.iloc[k]),-np.sin(theta.iloc[k])],[np.sin(theta.iloc[k]),np.cos(theta.iloc[k])]]) for k in range(N)]

	Sigma_0 = [np.array([[a.iloc[k]**2,0],[0,b.iloc[k]**2]]) for k in range(N)]
	Sigma   = [np.dot(R[k],np.dot(Sigma_0[k],R[k].T)) for k in range(N)]

	return np.array(Sigma)

def get_shear(A,mu,c,gamma1,gamma2,kappa,gsize,reduced=True):
	"""        
	Returns the combined shear of the blended system
	Author: Erfan Nourbakhsh

	[assuming N galaxies in the blended system]

	A      : the array of total fluxes
	mu     : the array (N vectors) of galaxy centers (i.e. the peaks of the Gaussians)
	c      : the vector pointing to the luminosity center of the blended system
	gamma1 : the array of the first component of the shears for N galaxies 
	gamma2 : the array of the second component of the shears for N galaxies        
	"""

	Sigma = get_covmat(gamma1,gamma2,kappa,gsize,reduced=reduced)

	# compute second moments
	Q11 = Qij(1,1,A,mu,c,Sigma) #,axis=-1)
	Q12 = Qij(1,2,A,mu,c,Sigma) #,axis=-1)
	Q22 = Qij(2,2,A,mu,c,Sigma) #,axis=-1)

	# WL9 in http://gravitationallensing.pbworks.com/w/page/15553259/Weak%20Lensing
	g1_sys = (Q11-Q22)/(Q11+Q22) #+2(Q11+Q22-Q12**2)**0.5 )
	g2_sys = 2.0*Q12/(Q11+Q22) #+2(Q11+Q22-Q12**2)**0.5 )

	return g1_sys, g2_sys


def sum_cumulative(thelist):
	if six.PY2:
		return list(np.cumsum(thelist)) # works for py2 and py3
	else:
		return list(accumulate(thelist)) # faster but only works in py3


# fair distribution of jobs for an efficient mpi
def fair_division(nbasket,nball):
	basket= [0]*nbasket
	while True:
		for i in range(nbasket):
			basket[i]+=1
			if sum(basket)==nball:
				return [0]+sum_cumulative(basket)


def add_field(data,newkey,values):
	if newkey in data.dtype.names:
		data[newkey] = values
	else:
		data = append_fields(data, newkey, values).filled()
	return data

def to_float(x):
	return [np.float(val) for val in x]

# def magtot(mags):
# 	return -2.5 * np.log10( np.sum(10**(-0.4*mags)) )

def magerr2snr(magerr):
	# dm = 2.5 log ( 1 + N/S )
	# N/S = 10^(dm/2.5) - 1
	SNR = 1./(10**(magerr/2.5) - 1)
	return SNR

# I initially liked the code described in the following link from DES and adopted a modified version of it here
# https://github.com/DarkEnergySurvey/easyaccess/blob/master/easyaccess/eautils/fileio.py
# However, I figured it is very slow for my purpose and I chose a better approach
def write_df2fits(filename, df, mode='w', comment='', extname=None, clobber=True):
	"""
	Write a pandas DataFrame to a FITS binary table using fitsio.
	It is necessary to convert the pandas.DataFrame to a numpy.array
	before writing, which leads to some hit in performance.

	Parameters:
	-----------
	filename :  FITS filename
	             <should have fit/fits at the end>
	df       :  DataFrame object
	mode     :  write mode: 'w'=write, 'a'=append
	comment  :  additional comment to put in the fits file
	clobber  :  if True, it over-writes the file if already exists
	extname  :  extension name

	Returns:
	--------
	None
	"""

	# TODO: with fits[1] we will have a problem with the flexibility to have multiple extensions for our fits files

	np_names = df.columns
	np_arr = df.values #.T # why transposing here and wasting memory

	# https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array-preserving-index
	if six.PY2: # py2
		types = [(np_names[i].encode(), df[name].dtype.type) for (i, name) in enumerate(np_names)]
	else:       # py3
		types = [(np_names[i], df[name].dtype.type) for (i, name) in enumerate(np_names)]

	data = np.empty(len(df.index), dtype=np.dtype(types)) 

	for (i, k) in enumerate(data.dtype.names):
		data[k] = np_arr[:, i]

	# write or append...
	if mode == 'w': # write a new file/extension
		if os.path.exists(filename) and clobber: os.remove(filename)
		fits = fio.FITS(filename, mode='rw', extname=extname, clobber=clobber)
		created = datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S')
		fits.write(data)
		fits[1].write_history('Created by BlendSim on ' + created)
		fits[1].write_comment('Comment ' + comment)
		fits.close()
	elif mode == 'a': # only append # slow and line by line :(
		fits = fio.FITS(filename, mode='rw', extname=extname) # clobber = obviously False
		fits[1].append(data)
		fits.close()
	else:
		msg = "Illegal write mode!"
		raise Exception(msg)


def mix_blends(grouped): # kw_param some extra keys to pass

# `grouped` is our input: sub dataframe

	GroupID = int(grouped['GroupID'].iloc[0]) # note: it has more than one (they are blends)
	GroupSize = int(grouped['GroupSize'].iloc[0])

	idnum = grouped['id'].iloc[0] # note: considered the first one's id for now for the sys
	ra, dec = grouped['ra'], grouped['dec'] # use absmag difference
	gamma1, gamma2 = grouped['gamma1'], grouped['gamma2']
	g1, g2 = grouped['g1'], grouped['g2']
	# e1, e2 = grouped['e1'], grouped['e2'] # nope, combine shapes without noise first and add a new noise after combining
	e1, e2 = grouped['e1_lensed_only'], grouped['e2_lensed_only'] 	
	gsize = grouped['size']
	kappa = grouped['kappa']
	ug_rest = grouped['ug_rest']
	u, g, r, i, z, y = grouped['u'], grouped['g'], grouped['r'], grouped['i'], grouped['z'], grouped['y']
	eu, eg, er, ei, ez, ey = grouped['eu'], grouped['eg'], grouped['er'], grouped['ei'], grouped['ez'], grouped['ey']    
	eug, egr, eri, eiz, ezy = grouped['eu-g'], grouped['eg-r'], grouped['er-i'], grouped['ei-z'], grouped['ez-y']    

	mag = np.array([ [u.iloc[j], g.iloc[j], r.i
	loc[j], i.iloc[j], z.iloc[j], y.iloc[j]] for j in range(GroupSize) ]) # 6 by GroupSize
	magerr = np.array([ [ eu.iloc[j], eg.iloc[j], er.iloc[j], ei.iloc[j], ez.iloc[j], ey.iloc[j] ] for j in range(GroupSize) ]) 

	mag_sys = -2.5 * np.log10( np.sum(10**(-0.4*mag), axis=0) ) # axis=0?? check it!!! I checked it; it was fine
	magerr_sys = np.sqrt(  np.sum((magerr*10**(-0.4*mag))**2, axis=0)  ) / ( np.sum(10**(-0.4*mag), axis=0) )


	col_sys = [ mag_sys[c]-mag_sys[c+1] for c in range(5) ]         
	ug_rest_sys = grouped['ug_rest'][ ug_rest==np.min(ug_rest) ].iloc[0]
	# we put the bluest one as a representative in the catalog
	zs_sys = grouped['zs'][ ug_rest==np.min(ug_rest) ].iloc[0]


	# using i mag as a ref. to find the centroid :: grouped['ra'][0] is ra_m1
	# the separation angle is so small that ra*cos(dec) won't make any difference
	ra_sys  = np.sum(ra *10**(-0.4*mag[:,3])) / np.sum(10**(-0.4*mag[:,3]))
	dec_sys = np.sum(dec*10**(-0.4*mag[:,3])) / np.sum(10**(-0.4*mag[:,3]))

	# this one is fast
	points = [(x,y) for (x,y) in zip(ra,dec)] # pairing up two lists
	gsize_sys = smec.make_circle(points)[2]*3600 # the radius of the smallest enclosing circle (in arcsec)

	A  = 10**(-0.4*mag[:,3]) # actual flux needs a factor because of the zero point, here we just use them as weights
	mu = [ra,dec] #np.array(points).T
	c  = [ra_sys,dec_sys]
	e1_sys, e2_sys = get_shear(A,mu,c,e1,e2,kappa,gsize,reduced=False) # actually no kappa needed fo e!
	gamma1_sys, gamma2_sys = gamma1.mean(), gamma2.mean() #get_shear(A,mu,c,gamma1,gamma2,kappa,gsize,reduced=False)
	g1_sys, g2_sys = g1.mean(), g2.mean() #get_shear(A,mu,c,gamma1,gamma2,kappa,gsize) # reduced by default
	kappa_sys = kappa.mean() # should be checked if it is ok?

	nGoldmem = sum(i<limiting_mag_i_lsst)

#----

	magerr_sys[0] = vec_getMagError(mag_sys[0],'LSST_u')
	magerr_sys[1] = vec_getMagError(mag_sys[1],'LSST_g')
	magerr_sys[2] = vec_getMagError(mag_sys[2],'LSST_r')
	magerr_sys[3] = vec_getMagError(mag_sys[3],'LSST_i')
	magerr_sys[4] = vec_getMagError(mag_sys[4],'LSST_z')
	magerr_sys[5] = vec_getMagError(mag_sys[5],'LSST_y')

	colerr_sys = [ np.sqrt(magerr_sys[c]**2 + magerr_sys[c+1]**2) for c in range(5) ] 

	e_sys = e1_sys + 1j*e2_sys # only sheared not noised yet
	nu = magerr2snr(magerr_sys[3]) # imag as ref.
	e_sys_noisy = epsnoise.addNoise(e_sys, nu, True) #[0]
	e1_sys_noisy = np.real(e_sys_noisy)
	e2_sys_noisy = np.imag(e_sys_noisy)

	bad = np.abs(e_sys_noisy) > 0.999999999999
	bad = bad.astype(int) # {1: useless shape, 0: useful shape}

	de1 = np.abs( e1_sys_noisy - e1_sys )
	de2 = np.abs( e2_sys_noisy - e2_sys )
	de = np.abs( e_sys_noisy - e_sys )

	shape_noise = np.mean(grouped['shape_noise']) # the whole shape noise based on redblue is iffy - will use the same shape noise for the whole thing in corr func
	shape_weight = 1. / (shape_noise**2 + de**2) #!!! TODO change this weight for the blended sample based on the new shape noise

	bdf = pd.DataFrame({'id':idnum,'u':mag_sys[0],'g':mag_sys[1],'r':mag_sys[2],'i':mag_sys[3],'z':mag_sys[4],'y':mag_sys[5],'ug_rest':ug_rest_sys,'u-g':col_sys[0], \
	'g-r':col_sys[1],'r-i':col_sys[2],'i-z':col_sys[3],'z-y':col_sys[4],'eu':magerr_sys[0],'eg':magerr_sys[1],'er':magerr_sys[2],'ei':magerr_sys[3],'ez':magerr_sys[4],'ey':magerr_sys[5], \
	'eu-g':colerr_sys[0],'eg-r':colerr_sys[1],'er-i':colerr_sys[2],'ei-z':colerr_sys[3],'ez-y':colerr_sys[4], \
	'zs':zs_sys,'ra':ra_sys,'dec':dec_sys,'size':gsize_sys,'gamma1':gamma1_sys,'gamma2':gamma2_sys,'kappa':kappa_sys, \
	'g1':g1_sys,'g2':g2_sys,'e1':e1_sys_noisy,'e2':e2_sys_noisy,'delta_e':de,'delta_e1':de1,'delta_e2':de2,'e1_lensed_only':e1_sys, \
	'e2_lensed_only':e1_sys,'shape_noise':shape_noise,'shape_weight':shape_weight,'isred':5,'badshape':bad, \
	'GroupID':GroupID,'GroupSize':GroupSize,'nGoldmem':nGoldmem}, index=[0])

	return bdf



# -----------------------
# making blended catalogs
# -----------------------

t0 = datetime.datetime.now() #time.process_time()

#maxsep = 3.3 # maximum separation for blends in arcsec
zregime = 'zs'
blend_percent = 5 # it turns out that you should make zsb_20percent folder otherwise `fits: file already exist?`
realization_number = 0
limiting_mag_i_lsst = 23.0 # we haven't cut yet for the blended systems - it is actually a global variable, no need to pass to the functions

input_dir  = '/global/cscratch1/sd/erfanxyz/projects/blending/buzzard_v1.6_lsst_y1/'
output_dir = '/global/cscratch1/sd/erfanxyz/projects/blending/buzzard_v1.6_lsst_y1/'

# cell numbers in NEST ordering - all ids in a quarter of sky excluding the edge cells
cell_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 120, 121, 122, 123, 124, 126, 285, 286, 287, 308, 309, 311, 349, 350, 351, 365, 366, 367, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 429, 430, 431, 440, 442, 443]

ncell = len(cell_ids)
nest = True
# nside = 8 # not used here

write_with_last_core = False # writing all the fits file with te last core is slow
last_core_o = 1 if parallel and write_with_last_core else 0 # whether the last core should be `exclusively` for outputs/prints/synchornization {1:yes, 0:no}

# njob_per_cell=0 means some (or all) cores have to do more than one iteration 
njob_per_cell = (size-last_core_o)//ncell # being conservative by giving cores the number of jobs the can definitely do
njob = ncell*njob_per_cell if parallel else ncell
jobs = range(njob)


mpi_index_list = fair_division(size-last_core_o,njob) if parallel else fair_division(1,njob) 
starting_index = mpi_index_list[rank] if parallel and rank<size-last_core_o else 0
ending_index = mpi_index_list[rank+1] if parallel and rank<size-last_core_o else 1

#-----

# set up LSST error model from PhotozDC1 codebase
errmodel = ephot.LSSTErrorModel()

# manually customize parameters
errmodel.nYrObs = 1.0 #nyear !!! important - same as get_tiles.py
errmodel.sigmaSys = 0.005
# https://docushare.lsstcorp.org/docushare/dsweb/Get/LPM-17
# Nv1 (design spec.) { 56 (2.2) 80 (2.4) 184 (2.8) 184 (2.8) 160 (2.8) 160 (2.8) } / 10 = per year
errmodel.nVisYr = {'LSST_u':5.6,'LSST_g':8,'LSST_r':18.4,'LSST_i':18.4,'LSST_z':16,'LSST_y':16}
# Non-numpy functions like math.sqrt() don't play nicely with numpy arrays
# So, I had to vectorize the function using numpy to speed it up
# vec_getObs = np.vectorize(errmodel.getObs) #getMagError(self, mag, filtObs)
vec_getMagError = np.vectorize(errmodel.getMagError)

#print('-->>> st3 %d' % rank)

def funcsn1(mag,band):
	# gives magnitude for the magnitude error where S/N = 1
	# dMag = 2.5 log ( 1 + N/S ) = 0.7526
	# _ , magerr = vec_getObs(mag,band)-2.5*np.log10(2)
	magerr = vec_getMagError(mag,band)-2.5*np.log10(2)
	return magerr

u_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_u'), maxiter=1000)
g_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_g'), maxiter=1000)
r_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_r'), maxiter=1000)
i_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_i'), maxiter=1000)
z_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_z'), maxiter=1000)
y_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_y'), maxiter=1000)

if rank==size-1: # last core in parallel or just the only core in serial
	print("Total cells (HEALPix pixels): %d" % ncell)
	if size==njob+1+last_core_o:
		print("Warning: %d core is idle and has no job assigned to it. Use maximum %d cores." % (size-(njob+last_core_o),njob+last_core_o))
	elif size>njob+last_core_o:
		print("Warning: %d cores are idle and have no jobs assigned to them. Use maximum %d cores." % (size-(njob+last_core_o),njob+last_core_o))
	isopt = 'optimum' if size%ncell==last_core_o else 'not optimum'
	extra = '+1' if write_with_last_core else ''
	print("Total cores to use: %d (%s)\nAn optimum number of cores are [factors of %d]%s for this configuration.\n\njob #" % (size,isopt,ncell,extra)) # one core reserved for output? y/n
	#print("Separation angle threshold for blends: %s arcsec\n\njob #" % maxsep)
	range_rank = range(size-last_core_o) if parallel else range(size)
	for tempRank in range_rank:
		indi = mpi_index_list[tempRank]
		indf = mpi_index_list[tempRank+1]-1
		if indi==indf:
			arrow = " \t"+str(indf)+"\t-------------> to core "+str(tempRank)
		elif indi>indf:
			arrow = " \tNA"+"\t-------------> to core "+str(tempRank) # no job is avalible for this core
		else:
			arrow = str(indi)+"\t"+str(indf)+"\t-------------> to core "+str(tempRank)
		print( arrow.expandtabs(len(str(mpi_index_list[-1]))+1) ) # some formatting hacks
	arrow = "outputs"+"\t-------------> to core "+str(size-1)	
	if last_core_o==1: print(arrow)

	explained = {'zs':'spectroscopic', 'zp':'photometric'}
	if rank==size-1: print("\n"+time.strftime('%a %H:%M:%S')+" :: Making blended catalog for the sample with %s redshifts:\n" % explained[zregime])

# -------------
# main process
# -------------

header = ['id','u','g','r','i','z','y','ug_rest','u-g','g-r','r-i','i-z','z-y',
	      'eu','eg','er','ei','ez','ey','eu-g','eg-r','er-i','ei-z','ez-y',
	      'zs','ra','dec','size','gamma1','gamma2','kappa','g1','g2','e1','e2',
	      'delta_e','delta_e1','delta_e2','e1_lensed_only','e2_lensed_only',
	      'shape_noise','shape_weight','isred','badshape','GroupID','GroupSize','nGoldmem']

meta  = pd.DataFrame({'id':'a11','u':'f8','g':'f8','r':'f8','i':'f8','z':'f8','y':'f8','ug_rest':'f8','u-g':'f8','g-r':'f8','r-i':'f8','i-z':'f8','z-y':'f8',
	      'eu':'f8','eg':'f8','er':'f8','ei':'f8','ez':'f8','ey':'f8','eu-g':'f8','eg-r':'f8','er-i':'f8','ei-z':'f8','ez-y':'f8',
	      'zs':'f8','ra':'f8','dec':'f8','size':'f8','gamma1':'f8','gamma2':'f8','kappa':'f8','g1':'f8','g2':'f8','e1':'f8','e2':'f8',
	      'delta_e':'f8','delta_e1':'f8','delta_e2':'f8','e1_lensed_only':'f8','e2_lensed_only':'f8',
	      'shape_noise':'f8','shape_weight':'f8','isred':'f8','badshape':'f8','GroupID':'i8','GroupSize':'int','nGoldmem':'int'}, index=[0])


for irank in jobs[starting_index:ending_index]: # turns into only one iteration in serial! irank=0
	if rank==size-1 and parallel and write_with_last_core: break # last core should or should not make any catalog

	cell_ids_used = [cell_ids[irank//njob_per_cell]] if parallel else cell_ids
	if not parallel: progress = progress_bar(ncell,'catalogs',per=5, html=False)
	
	for m, cell in enumerate(cell_ids_used):
	
		input_file_name = input_dir+zregime+'nb/'+zregime+'nb.'+str(cell)+'.grouped_'+str(blend_percent)+'percent_r'+str(realization_number)+'.fit'
		out_root = output_dir+zregime+'b_'+str(blend_percent)+'percent/'
		usedir(out_root) # creates it if it does not exist
		output_file_name = out_root+zregime+'b_'+str(blend_percent)+'percent.'+str(cell)+'_r'+str(realization_number)+'.fit'

		npcat = fio.read(input_file_name) #, extname='blends_stat') extname is useless bc no matter what that is it reads the first ext
		nblendsys = int(max(npcat['GroupID']))
		npcat = npcat.byteswap().newbyteorder()
		df = pd.DataFrame.from_records(npcat) 
		del npcat

		if parallel and njob_per_cell>1:
			blendsys_idx = fair_division(njob_per_cell,nblendsys) # in this case we need a +1 later
			nth = irank%njob_per_cell # nth chunk of our blend systems (min: 0, max: njob_per_cell-1)
			brange = range(blendsys_idx[nth]+1,blendsys_idx[nth+1]+1)
		else:
			brange = range(1,nblendsys+1)

		singles = (df['GroupID'].astype(int)==0)
		# df_singles = df[singles] # not here (to save memory)
		df_blends  = df[~singles] # only blends (still not merged)
		
		print('df_blends.shape = ',df_blends.shape)

		if parallel and njob_per_cell>1:
			if not write_with_last_core:
				if nth==njob_per_cell-1: # last core that works on this cell wraps things up
					df_singles = df[singles]
		else:
			df_singles = df[singles]	


		del df # memory considerations
		del singles

		ncpu = multiprocessing.cpu_count()
		nworkers = int(ncpu/2.5) #int(ncpu/2.5)

		# https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
		df_groups = df_blends.loc[df_blends['GroupID'].astype(int).isin(brange)] # exclude 0 because it is for isolated galaxies
		print('df_blends_inbrange.shape = ',df_groups.shape)
		del df_blends


		#df_groups = df[df['GroupID']!=0] # only blends # no need; last line did that already
		print(time.strftime('%a %H:%M:%S')+" :: start grouping/filtering") 
		df_groups = filter_by_multiprocessing(df_groups, _filter_df_gbf, brange, 'GroupID', limiting_mag_i_lsst, workers=nworkers) # fast, takes ~ < 2 min
		# I thought I could simply use dask instead of filter_by_multiprocessing but had some problems bc dask data frame did not have the attr filter
		# df_groups = dask_df_groups.groupby(df_groups.GroupID).filter(lambda q: -2.5*np.log10( np.sum(10**(-0.4*q['i']) ) ) <= limiting_mag_i_lsst).compute()
		print(time.strftime('%a %H:%M:%S')+" :: end grouping/filtering") 
		print('df_groups_filtered.shape = ',df_groups.shape)
		# Note: some numbers in brange are useless then (after magcut) - but I assume it is random

		if parallel and njob_per_cell>1: print("Blend chunk %d/%d [%d...%d] in cell #%d has been identified by core %d." %(nth+1,njob_per_cell,brange[0],brange[-1],cell,rank) )

		print(time.strftime('%a %H:%M:%S')+' :: start set index, ncpu = ',ncpu)
		df_groups = dd.from_pandas(df_groups,npartitions=nworkers//2) # use optimized npartitions #does partitioning conflicts with grouping? e.g two galaxies that belongs to the same group appear in two different chunks? I'd say no, I guess dask is smarter than this
		df_groups = df_groups.set_index(df_groups.GroupID) #, method='disk') # set_index is expensive for very huge datasets; we are fine; it makes dask's apply fast

		print(time.strftime('%a %H:%M:%S')+' :: end set index, rank = ', rank)
		gb = df_groups.groupby(df_groups.GroupID)
		del df_groups
		print(time.strftime('%a %H:%M:%S')+' :: gb done ', rank)

		# https://gist.github.com/edraizen/92391407f5301b15f179865cf74f07a2
		df_blends = gb.apply(mix_blends, meta=meta).compute(get=dget,num_workers=nworkers//2) # Here, compute() calls Dask to map the apply to each partition and (get=dget) tells Dask to run this in parallel.

		del gb

		# in the case where the metadata doesn’t change for the output, I thought I can also pass in the pd dataframe object [df_groups.compute()] itself directly but headers did not match the column and I guess it was because additional index we have here
		if parallel and njob_per_cell>1:
			print(time.strftime('%a %H:%M:%S')+' :: ++ weeding out faint blended systems done!, cell, chunk, nchunk, rank = ',cell, nth+1,njob_per_cell, rank)
		else:
			print(time.strftime('%a %H:%M:%S')+' :: -- weeding out faint blended systems done!, cell, rank = ',cell, rank)



		if parallel and njob_per_cell>1:
			
			if write_with_last_core:
				destrank = size-1 #(irank//njob_per_cell)*njob_per_cell # 
				# very important for large files: 
				# https://groups.google.com/forum/#!topic/mpi4py/oRs_D2FHH2Q
				# "comm.send()" usually buffers the message (typically when the 
				# message is small), but this buffering has a limit, so is you 
				# continuously send() but never recv(), your code end-up hanging. 
				# "comm.ssend()" is guaranteed to never buffer and halt execution until 
				# a matching recv() is posted at the destination.
				# >> so receive it somewhere soon like in loading
				COMM.ssend(df_blends, dest=destrank, tag=int(str(cell)+str(nth)))
				del df_blends
				print("Chunk %d/%d [%d...%d] in cell #%d got blended and sent to the master." %(nth+1,njob_per_cell,brange[0],brange[-1],cell) )
			else:
				if nth==0:
					df_blends_all = df_blends
				else:
					df_blends_all = COMM.recv(source=MPI.ANY_SOURCE, tag=int(str(cell)+str(nth-1))) # previous tag
					df_blends_all = pd.concat([df_blends_all,df_blends])

				print("Blend chunk %d/%d [%d...%d] in cell #%d is pasted to data." %(nth+1,njob_per_cell,brange[0],brange[-1],cell) )

				if nth<njob_per_cell-1:
					# pass the last updated blend data to the next core
					COMM.ssend(df_blends_all, dest=rank+1, tag=int(str(cell)+str(nth)))
					del df_blends_all 
				else:
					# last core (for this cell) will wrap things up
					# df_singles = df[singles] # only make it for the core that writes (already did it) - it saves memory
					df = pd.concat([df_singles,df_blends_all])
					# reorder the columns
					df = df[header]
					# write to a fits file
					fio.write(output_file_name,df.to_records(index=False),clobber=True)
					print(time.strftime('%a %H:%M:%S')+" The full blended catalog for cell #%d is now saved in %s" % (cell, output_file_name) )

		else: # not that time consuming ~ 3-4 min
			print("Blended cell #%d is now BEING saved in %s" % (cell, output_file_name) )
			df = pd.concat([df_singles,df_blends]) #, ignore_index=True)
			# reorder the columns the way you want
			df = df[header]
			# write to a fits file - no appending etc.
			fio.write(output_file_name,df.to_records(index=False),clobber=True)
			print("Blended cell #%d is now saved in %s" % (cell, output_file_name) )
			progress.refresh(m)


# when we use MPI_barrier() for synchronization, we are guaranteed that no process 
# will leave the barrier until all processes have entered it
#if parallel: COMM.Barrier(); time.sleep(5)
if rank==size-1 and parallel and write_with_last_core: print("\nSaving output catalogs started...\n")

if parallel and rank==size-1:
#for cell in gen:
	if write_with_last_core: # last core is doing all the writings (slow)
		for cell in cell_ids:
			input_file_name = input_dir+zregime+'nb/'+zregime+'nb.'+str(cell)+'.grouped_r'+str(realization_number)+'.fit'
			output_file_name = output_dir+zregime+'b/'+zregime+'b.'+str(cell)+'_r'+str(realization_number)+'.fit'

			# grabbbing data for single galaxies first
			npcat = fio.read(input_file_name, extname='catalog')#, rows=range(1,100050), extname='catalog')
			npcat = npcat.byteswap().newbyteorder()
			df = pd.DataFrame.from_records(npcat)
			del npcat
			df = df[df['GroupID'].astype(int)==0] # only singles at this point

			# ---- adding the rest (blend ones) at the end ----
			for nth in range(njob_per_cell):
				tag=int(str(cell)+str(nth))
				df_blends = COMM.recv(source=MPI.ANY_SOURCE, tag=tag) # tags are unique
				if df_blends.shape[0]<10: print("* Warning: the master is not receiving the dataframe with the tag %d."%tag)
				df = pd.concat([df,df_blends]) #, ignore_index=True)

			# reorder the columns the way you want
			df = df[header]

			# write to a fits file - time-consuming
			# fio.write(output_file_name,df.to_records(index=False),clobber=True)
			write_df2fits(output_file_name, df, mode='w', comment='Blended Catalog', clobber=True) # this method has more options
			print("Blended cell #%d is now saved in %s" % (cell, output_file_name) )

if parallel: COMM.Barrier(); time.sleep(1)
#END: This needs to print after all MPI_Send/MPI_Recv has been completed - automatic pass in serial mode
if rank==size-1:
	t1 = datetime.datetime.now()
	print( "\n"+time.strftime('%a %H:%M:%S')+" :: All done! Elapsed time: "+str(datetime.timedelta(seconds=round((t1-t0).seconds))) )


