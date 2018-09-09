from __future__ import print_function

# In order to write this code I took advantage of:
# https://github.com/LSSTDESC/WLPipe/tree/master/wlpipe/bin
# technically I could do the whole thing using the twopoint package too (maybe later)

# ValueError: unsupported pickle protocol: 3 
# this error happens if you run with py2
# please use py3 to run this script

import sys

if (sys.version_info > (3, 0)):
	# Python 3 code in this block
	pyver = 3
	## adding a python env with packages like DESCQA treecorr etc...
	sys.path.insert(0, '/global/common/software/lsst/common/miniconda/py3-4.2.12/lib/python3.6/site-packages') # this one had fitsio, etc.
	sys.path.insert(0, '/global/common/software/lsst/common/miniconda/py3-4.3.21/lib/python3.6/site-packages')
	import pickle
else:
	# Python 2 code in this block
	pyver = 2
	sys.path.insert(0, '/global/common/cori/contrib/lsst/apps/anaconda/py2-envs/DESCQA/lib/python2.7/site-packages')
	sys.path.insert(0, '/global/homes/e/erfanxyz/.local/lib/python2.7/site-packages')
	import cPickle as pickle


import os
import time
import datetime
import numpy as np
import collections
from numpy import linalg as la
from astropy.io import fits as pf # aka pyfits
import fitsio as fio # more efficient for dNdz part
import multiprocessing # it can spawn multiple processes but they will still be bound within a single node
# from scoop import futures as scoop # for multiple nodes over network
import tqdm
import subprocess
# import pipes 

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/pyRMT') # Python for Random Matrix Theory: needed for shrinkage estimation
import pyRMT
# *** important: it is tweaked by me so that we have an option to NOT transpose
# the matrix to if T < N (we have a matrix of shape (T, N), where T denotes
# the number of samples and N labels the number of features)

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/Jupytermeter') # loading bar
from jupytermeter import *

###################################
# functions that we need
###################################

# from: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
def flatten(x): # look at the link for py2 compatibility
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def is_symmetric(a, tol=0): # 1e-20
	return np.allclose(a, a.T, atol=tol)

def check_sym_posdef(mat):
	ev = np.sort(np.linalg.eigvals(mat))
	print("min and max eigenvalues:",np.min(ev), np.max(ev))
	if not is_symmetric(mat):
		print("WARNING: covariance matrix is not symmetric!")
	neg = np.where(ev < 0)
	nnegev = len(neg[0])
	negdef = (nnegev != 0)
	if not isPD(mat): #if negdef:
		print("WARNING: covariance matrix is not positive definite!\n%d negative eigenvalues:\n" % nnegev ,ev[neg])
	return isPD(mat) #return not negdef # returns True if the matrix is positive definite

# nearestPD() from: https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
# there is also a very nice comprehensive one implemented in R:
# http://stat.ethz.ch/R-manual/R-devel/library/Matrix/html/nearPD.html
def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

	#from numpy import linalg as la

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrices with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrices of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B): # is a better way to check for posdef
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False



# from: http://www.statsmodels.org/0.6.1/_modules/statsmodels/stats/moment_helpers.html
def cov2corr(cov, return_std=False):
	'''convert covariance matrix to correlation matrix

	Parameters
	----------
	cov : array_like, 2d
	    covariance matrix, see Notes

	Returns
	-------
	corr : ndarray (subclass)
	    correlation matrix
	return_std : bool
	    If this is true then the standard deviation is also returned.
	    By default only the correlation matrix is returned.

	Notes
	-----
	This function does not convert subclasses of ndarrays. This requires
	that division is defined elementwise. np.ma.array and np.matrix are allowed.

	'''
	cov = np.asanyarray(cov)
	std_ = np.sqrt(np.diag(cov))
	corr = cov / np.outer(std_, std_)
	if return_std:
		return corr, std_
	else:
		return corr

def get_zdata(args):
	
	cell_ids_chunk, kwargs = args

	realization_numbers = kwargs.pop('realization_numbers')
	input_dir = kwargs.pop('input_dir')
	regime = kwargs.pop('regime')
	zcuts = kwargs.pop('zcuts')
	limiting_imag = kwargs.pop('limiting_imag')
	nest = kwargs.pop('nest')
	nside = kwargs.pop('nside')

	zdata_all_chunk, zdata_lens_chunk, zdata_source_chunk = [], [], []
	
	for realization_number in realization_numbers:
		for counter,cell in enumerate(cell_ids_chunk):
			input_file_name  = input_dir+regime+bperc+'/'+regime+bperc+'.'+str(cell)+'_r'+str(realization_number)+'.fit'
			
			# - using pyfits
			# input_data = pf.open(input_file_name)[1].data
			# input_data = input_data[input_data['i']<=limiting_imag] # do not forget to filter imag
			
			# - using fitsio [it loads selected columns w/o loading the entire data] # great speed-up! 60X with 10 cells
			input_data = fio.read(input_file_name,columns=['i',regime[:2],'badshape']) # NOTE: maybe for photo-z you want to use the p(z) itself!!
			input_data = input_data[input_data['i']<=limiting_imag] # do not forget to filter imag
			imag = input_data['i']
			badshape = input_data['badshape']
			z = input_data['zs'] if regime[:2]=='zs' else input_data['zp'] # spec-z or photo-z
			del input_data

			# very important: make sure you did the same cuts while computing the corresponding correlation functions
			idx_lens = (imag<21)&(z>zcuts[0])&(z<=zcuts[-2]) # &(badshape==0)
			idx_source = (imag>=21)&(z>zcuts[1])&(z<=zcuts[-1])&(badshape==0)
			# no magcut or bad shape filter for wtheta:
			zdata_all_chunk = np.append(zdata_all_chunk,z) # since axis is not given, both arr and values are flattened before use
			zdata_lens_chunk = np.append(zdata_lens_chunk,z[idx_lens]) 
			zdata_source_chunk = np.append(zdata_source_chunk,z[idx_source])
			del z, imag, badshape

	return zdata_all_chunk, zdata_lens_chunk, zdata_source_chunk

def get_zdata_by_multiprocessing(cell_ids,nworkers,**kwargs):
	nwork = len(cell_ids)
	if nwork<nworkers:
		print('Number of workers (%d) is more than number of work units (%d). `nworkers` is set to %d.'%(nworkers,nwork,nwork))
		nworkers = nwork
	cell_ids_chunks = np.array_split(cell_ids, nworkers)
	pool = multiprocessing.Pool(processes=nworkers)
	results = []
	for y in list(tqdm.tqdm(pool.imap_unordered(get_zdata, [(cid, kwargs) for cid in cell_ids_chunks]), total=nworkers)):
		results.append(y)
	results = zip(*results)
	pool.close()
	return [np.concatenate(res,axis=0) for res in results]


###################################
# main program starts here
###################################

# runtime ~ zsnb 2 min with 63 cores on cori | 143 healpix pixels and imag cut 24

# - determine start time
t0 = datetime.datetime.now()

# - important configs
regime = 'zsnb' # 1min 44sec for zsnb and 33sec for zsb (already cut data in imag 23)
blend_percent = 5
limiting_imag = 23.0
realization_numbers = [0]
ctypes = ['pp','ps','ss'] 
zcuts = [0.0,0.25,0.5,0.75,1.0] 
ntomo = len(zcuts)-1
ntheta = 6
shrink_covmat = False # True
covboost = 1.0 
M_nercome = 1000 # number of realizations for nercome

explained = {'zsnb':['spectroscopic','unblended'], 'zsb':['spectroscopic','blended'], 'zpnb':['photometric','unblended'], 'zpb':['photometric','blended']}
print("\n"+time.strftime('%a %H:%M:%S')+" :: This script will create a twopoint file for the %s sample with %s redshifts:\n" % (explained[regime][1],explained[regime][0]))

# - set input/output directory
input_dir  = '/global/cscratch1/sd/erfanxyz/projects/blending/buzzard_v1.6_lsst_y1/'
output_dir = '/global/cscratch1/sd/erfanxyz/projects/blending/corrs/'

# - output twopoint file would be stored in:
bperc = '' if regime=='zsnb' else '_'+str(blend_percent)+'percent'
foutput = output_dir+regime+bperc+'/all_realizations_combined/'+'cosmosis.twopoint.'+regime+bperc+'.fit'

# - the averaged values of correlations is already stored in:
corr_file_name = output_dir+regime+bperc+'/all_realizations_combined/'+'corrs.'+regime+bperc+'.grand_'+str(len(realization_numbers))+'_realizations.fit'

# - the covariance matrix is already stored in:
covmat_file_name = output_dir+regime+bperc+'/all_realizations_combined/'+'covmat.'+regime+bperc+'_'+str(len(realization_numbers))+'_realizations.npz'

# - open the fits file with calculated correlations
corr_table = pf.open(corr_file_name)

# ------- extracting info for position-position correlations -------
if 'pp' in ctypes:
	ctype='pp'
	angbin_wtheta_vec, ang_wtheta_vec = [], []
	bin1_wtheta_vec, bin2_wtheta_vec = [], []
	wtheta_vec = []

	# - make sure the following loop conforms with the format of our data vector
	for i in range(ntomo):
		for j in range(ntomo):
			if(j==i):
				corrdata = corr_table[ctype+'_'+str(i)+'_'+str(j)].data
				bin1_wtheta_vec.append(corrdata['zbin1']+1) # +1 because it was zero-indexed :: (nparray).astype(int)
				bin2_wtheta_vec.append(corrdata['zbin2']+1)
				angbin_wtheta_vec.append(list(range(1,len(corrdata['theta'])+1)))
				wtheta_vec.append(corrdata['wtheta'])
				ang_wtheta_vec.append(corrdata['theta'])

	# - flatten the lists to feed to the twopoint fits file
	bin1_wtheta_vec = flatten(bin1_wtheta_vec)
	bin2_wtheta_vec = flatten(bin2_wtheta_vec)
	angbin_wtheta_vec = flatten(angbin_wtheta_vec)
	wtheta_vec = flatten(wtheta_vec)
	ang_wtheta_vec = flatten(np.array(ang_wtheta_vec)*60.0) # degrees to arcmin

	# - length of wtheta data vector considering tomo bins and theta bins		
	npt_pp = len(wtheta_vec) # important to do it after flattening
else:
	npt_pp = 0

# ------- extracting info for position-shear correlations -------
if 'ps' in ctypes:
	ctype='ps'
	angbin_gammat_vec, ang_gammat_vec = [], []
	bin1_gammat_vec, bin2_gammat_vec = [], []
	gammat_vec = []

	# - make sure the following loop conforms with the format of our data vector
	for i in range(1,ntomo): # [source]
		for j in range(ntomo-1): # [lens]
			if(j<i):
				corrdata = corr_table[ctype+'_'+str(i)+'_'+str(j)].data
				bin1_gammat_vec.append(corrdata['zbin1']) # no +1 because the first zbin for source is already numbered as 1 because of our design
				bin2_gammat_vec.append(corrdata['zbin2']+1) # +1 because it was zero-indexed starting from lenses
				angbin_gammat_vec.append(list(range(1,len(corrdata['theta'])+1)))
				gammat_vec.append(corrdata['gammat'])
				ang_gammat_vec.append(corrdata['theta'])

	# - flatten the lists to feed to the twopoint fits file
	bin1_gammat_vec = flatten(bin1_gammat_vec)
	bin2_gammat_vec = flatten(bin2_gammat_vec)
	angbin_gammat_vec = flatten(angbin_gammat_vec)
	gammat_vec = flatten(gammat_vec)
	ang_gammat_vec = flatten(np.array(ang_gammat_vec)*60.0) # degrees to arcmin

	# - length of wtheta data vector considering tomo bins and theta bins		
	npt_ps = len(gammat_vec) # important to do it after flattening
else:
	npt_ps = 0

# ------- extracting info for shear-shear correlations -------
if 'ss' in ctypes:
	ctype='ss'
	angbin_xipm_vec, ang_xipm_vec = [], []
	bin1_xipm_vec, bin2_xipm_vec = [], []
	xip_vec, xim_vec = [], []

	# - make sure the following loop conforms with the format of our data vector
	for i in range(1,ntomo):
		for j in range(1,ntomo):
			if(j<=i):
				corrdata = corr_table[ctype+'_'+str(i)+'_'+str(j)].data
				# NOTE: I am switching zbin1 and zbin2. It doesn't change the physics. I am just being conservative
				# since I noticed the example 2pt files for shear-shear correlations follow a pattern with i<=j
				bin1_xipm_vec.append(corrdata['zbin2']) # no +1 because first bin is numbered 1 already
				bin2_xipm_vec.append(corrdata['zbin1']) # no +1 because first bin is numbered 1 already
				angbin_xipm_vec.append(list(range(1,len(corrdata['theta'])+1)))
				xip_vec.append(corrdata['xip'])
				xim_vec.append(corrdata['xim'])
				ang_xipm_vec.append(corrdata['theta'])

	# - flatten the lists to feed to the twopoint fits file
	bin1_xipm_vec = flatten(bin1_xipm_vec)
	bin2_xipm_vec = flatten(bin2_xipm_vec)
	angbin_xipm_vec = flatten(angbin_xipm_vec)
	xip_vec = flatten(xip_vec)
	xim_vec = flatten(xim_vec)
	ang_xipm_vec = flatten(np.array(ang_xipm_vec)*60.0) # degrees to arcmin

	# - length of one of the two shear data vectors considering tomo bins and theta bins		
	npt_ss = len(xip_vec) # important to do it after flattening
else:
	npt_ss = 0

###################################
# Do the COVMAT extension
###################################

print("Doing COVMAT extension...")

dv4m_fname = output_dir+regime+bperc+'/all_realizations_combined/'+'datavec_for_matlab.npy'

# - loading covmat file
covdata = np.load(covmat_file_name)

# https://stats.stackexchange.com/questions/52976/is-a-sample-covariance-matrix-always-symmetric-and-positive-definite
# "A correct covariance matrix is always symmetric and positive *semi*definite.""
# https://math.stackexchange.com/questions/2167347/can-a-positive-definite-matrix-have-complex-eigenvalues
# if your matrix is a covariance matrix. It should therefore have real, non-negative eigenvalues.
# Any imaginary component to your eigenvalues is likely due to numerical error. If this is the case,
# try just taking the real part of the eigenvalues (you should find that the imaginary parts are close to 0 anyway).

# - estimate covariance matrix from data vectors with non-linear shrinkage
#covmat = covdata['covmat'] - old covmat without shrinkage
#datavector = np.tile(covdata['corrsupervec'], 8) # tiling just to make T>N for testing
covmat = covboost*covdata['covmat']
datavector = covdata['corrsupervec'] #[0:40] # filtering just to make T>N for testing

if shrink_covmat:
	print('Shrinking the covariance matrix using MATLAB...')
	np.save(dv4m_fname, datavector)
	# pro tip: in the following "" should be inside ''
	os.system('module load matlab && matlab -nodisplay -nojvm -logfile matlab.log -r "cd matlab_codes; try shrinkit(\'%s\',%s); catch; end; quit"' % (dv4m_fname, M_nercome))
	fpath = dv4m_fname.rsplit('/',1)[0]+'/'
	pklfname = fpath+'shrunk_covmat_nercome.pkl'
	with open(pklfname, "rb") as f: covmat = covboost*pickle.load(f)
	print('Successfully loaded the shrunk covariance matrix!')


X = datavector.T # transpose here to conform with the input for the shrinkage function
# X: a matrix of shape (T, N), where T denotes the number of samples and N labels the number of features
print("Based on the provided data vectors the following is assumed:")
print("X.shape (T, N)     =", X.shape)
print("Number of samples  = %d" % X.shape[0])
print("Number of feauters = %d" % X.shape[1])
# print("Shrinking the noisy covariace matrix using Ledoit-Wolf non-linear shrinkage estimator...")
# - the following LW shrinkage results were not good, forget it for now
# - calculate the shrunk covariance matrix using Ledoit-Wolf non-linear
#   shrinkage estimator 2017 (http://www.econ.uzh.ch/static/wp/econwp264.pdf)
# covmat = pyRMT.optimalShrinkage(X, return_covariance=True, method='iw') #method='kernel', allow_transpose=False)
# print("The original %dx%d covariance matrix has been shrunk successfully to a smoother %dx%d covariance matrix!" % (X.shape[1],X.shape[1],covmat.shape[0],covmat.shape[1]) )
# shrunk_outfile = output_dir+regime+'/all_realizations_combined/'+'covmat_shrunk_LW.'+regime+'_'+str(len(realization_numbers))+'_realizations.npz' 
# np.savez(shrunk_outfile, covmat=covmat, corrmat=cov2corr(covmat), corrvec=covdata['corrvec'], corrsupervec=covdata['corrsupervec'], labels=covdata['labels'])
# print('Saved '+shrunk_outfile+'\n')

# - check if covmat is positive definite
posdef = check_sym_posdef(covmat) # expect a pos-def matrix after Ledoit-Wolf non-linear shrinkage
if posdef:
	print("GOOD NEWS: the covariance matrix is already positive definte!")
else:
	print("\nComputing the nearest *semi* positive definte covariance matrix...")
	covmat_posdef = nearestPD(covmat) # by pos-def we actually mean semi pos-def (or in fact: not neg-def)
	delta = np.abs(covmat_posdef-covmat)
	delta_percent = 100.*delta/covmat
	print("Min, mean and max difference b/w old and new elements [absolute] =",np.min(delta),np.mean(delta),np.max(delta))
	print("Min, mean and max difference [percentage] =",np.min(delta_percent),np.mean(delta_percent),np.max(delta_percent))
	ispd = check_sym_posdef(covmat_posdef) # check after fixing to make sure
	posdef_outfile = output_dir+regime+bperc+'/'+'covmat_shrunk_posdef.'+regime+bperc+'.npz' # should we also recalculate corrvec and errors for the posdef matrix?
	np.savez(posdef_outfile, covmat=covmat_pd, corrmat=cov2corr(covmat_pd), corrvec=covdata['corrvec'], corrsupervec=covdata['corrsupervec'], labels=covdata['labels'])
	print('Saved '+posdef_outfile+'\n')
	covmat = covmat_posdef

# print("Shrinking the covariance matrix using NERCOME method...")
# subprocess.call(["module","load","matlab"]) 
# subprocess.call(["matlab","nercome.m","---"]) 

# - create a new image file using the 2d numpy array representing the covariance matrix
imhdu=pf.ImageHDU(covmat)

# TODO: do not make NAME_1,2,3 if we only have one probe in covmat - it made a problem and cosmosis did not output any chain

# - update the table's header
imhdr=imhdu.header
imhdr.set('COVDATA',True,after='GCOUNT')
imhdr.set('EXTNAME','COVMAT',after='COVDATA')
# - chunk 1
imhdr.set('STRT_0',0,after='EXTNAME') # 0: starting position of the first dataset stored in covmat
imhdr.set('NAME_0','WTHETA',after='STRT_0')
# - chunk 2
imhdr.set('STRT_1',npt_pp,after='NAME_0') # npt_pp: starting position of the second dataset stored in covmat
imhdr.set('NAME_1','GAMMAT',after='STRT_1')
# - chunk 3
imhdr.set('STRT_2',npt_pp+npt_ps,after='NAME_1') # npt_pp+npt_ps: starting position of the third dataset stored in covmat
imhdr.set('NAME_2','XIP',after='STRT_2')
# - chunk 4
imhdr.set('STRT_3',npt_pp+npt_ps+npt_ss,after='NAME_2') # npt_pp+npt_ps+npt_ss: starting position of the fourth dataset stored in covmat
imhdr.set('NAME_3','XIM',after='STRT_3')

imhdu.header = imhdr

# - write the COVMAT extension to a new output file to start
imhdu.writeto(foutput,overwrite=True)

###################################
# Do the WTHETA 2PTDATA extensions
###################################
if 'pp' in ctypes:

	print("Doing WTHETA 2PTDATA extension...")

	colnames=['BIN1','BIN2','ANGBIN','VALUE','ANG']
	c1 =pf.Column(name=colnames[0],format='K',array=bin1_wtheta_vec)
	c2 =pf.Column(name=colnames[1],format='K',array=bin2_wtheta_vec)
	c3 =pf.Column(name=colnames[2],format='K',array=angbin_wtheta_vec)
	c4 =pf.Column(name=colnames[3],format='D',array=wtheta_vec)
	c5 =pf.Column(name=colnames[4],format='D',unit='arcmin',array=ang_wtheta_vec) # make sure it's in arcmin!

	# - first do XIP
	tbhdu=pf.BinTableHDU.from_columns([c1,c2,c3,c4,c5])

	# - update the table's header
	tbhdr=tbhdu.header
	tbhdr.set('2PTDATA',True,before='TTYPE1')
	tbhdr.set('EXTNAME','WTHETA','name of this 2ptcorr table',after='2PTDATA') # Note: extname should be consistent with covmat
	tbhdr.set('QUANT1','GPR','1st 2ptcorr quantity code',after='EXTNAME')
	tbhdr.set('QUANT2','GPR','2nd 2ptcorr quantity code',after='QUANT1')
	tbhdr.set('KERNEL_1','NZ_LENS',after='QUANT2')
	tbhdr.set('KERNEL_2','NZ_LENS',after='KERNEL_1')
	tbhdr.set('WINDOWS','SAMPLE',after='KERNEL_2')
	tbhdr.set('N_ZBIN_1',ntomo-1,'number of redshift/distance bins',after='WINDOWS')
	tbhdr.set('N_ZBIN_2',ntomo-1,'number of redshift/distance bins',after='N_ZBIN_1')
	tbhdr.set('N_ANG',ntheta,'number of angular bins',after='N_ZBIN_2')

	# - append the XIP extension to the output file created above
	pf.append(foutput,tbhdu.data,header=tbhdr)

###################################
# Do the GAMMAT 2PTDATA extensions
###################################
if 'ps' in ctypes:

	print("Doing GAMMAT 2PTDATA extension...")

	colnames=['BIN1','BIN2','ANGBIN','VALUE','ANG']
	c1 =pf.Column(name=colnames[0],format='K',array=bin1_gammat_vec)
	c2 =pf.Column(name=colnames[1],format='K',array=bin2_gammat_vec)
	c3 =pf.Column(name=colnames[2],format='K',array=angbin_gammat_vec)
	c4 =pf.Column(name=colnames[3],format='D',array=gammat_vec)
	c5 =pf.Column(name=colnames[4],format='D',unit='arcmin',array=ang_gammat_vec) # make sure it's in arcmin!

	# - first do XIP
	tbhdu=pf.BinTableHDU.from_columns([c1,c2,c3,c4,c5])

	# - update the table's header
	tbhdr=tbhdu.header
	tbhdr.set('2PTDATA',True,before='TTYPE1')
	tbhdr.set('EXTNAME','GAMMAT','name of this 2ptcorr table',after='2PTDATA') # Note: extname should be consistent with covmat
	tbhdr.set('QUANT1','GPR','1st 2ptcorr quantity code',after='EXTNAME')
	tbhdr.set('QUANT2','G+R','2nd 2ptcorr quantity code',after='QUANT1')
	tbhdr.set('KERNEL_1','NZ_LENS',after='QUANT2')
	tbhdr.set('KERNEL_2','NZ_SOURCE',after='KERNEL_1')
	tbhdr.set('WINDOWS','SAMPLE',after='KERNEL_2')
	tbhdr.set('N_ZBIN_1',ntomo-1,'number of redshift/distance bins',after='WINDOWS')
	tbhdr.set('N_ZBIN_2',ntomo-1,'number of redshift/distance bins',after='N_ZBIN_1')
	tbhdr.set('N_ANG',ntheta,'number of angular bins',after='N_ZBIN_2')

	# - append the XIP extension to the output file created above
	pf.append(foutput,tbhdu.data,header=tbhdr)

###################################
# Do the XIP/XIM 2PTDATA extensions
###################################
if 'ss' in ctypes:

	print("Doing XIP 2PTDATA extension...")

	colnames=['BIN1','BIN2','ANGBIN','VALUE','ANG']
	c1 =pf.Column(name=colnames[0],format='K',array=bin1_xipm_vec)
	c2 =pf.Column(name=colnames[1],format='K',array=bin2_xipm_vec)
	c3 =pf.Column(name=colnames[2],format='K',array=angbin_xipm_vec)
	c4p=pf.Column(name=colnames[3],format='D',array=xip_vec)
	c4m=pf.Column(name=colnames[3],format='D',array=xim_vec)
	c5 =pf.Column(name=colnames[4],format='D',unit='arcmin',array=ang_xipm_vec) # make sure it's in arcmin!

	# -------- first do XIP --------
	tbhdu=pf.BinTableHDU.from_columns([c1,c2,c3,c4p,c5])

	# - update the table's header
	tbhdr=tbhdu.header
	tbhdr.set('2PTDATA',True,before='TTYPE1')
	tbhdr.set('EXTNAME','XIP','name of this 2ptcorr table',after='2PTDATA') # Note: extname should be consistent with covmat
	tbhdr.set('QUANT1','G+R','1st 2ptcorr quantity code',after='EXTNAME')
	tbhdr.set('QUANT2','G+R','2nd 2ptcorr quantity code',after='QUANT1')
	tbhdr.set('KERNEL_1','NZ_SOURCE',after='QUANT2')
	tbhdr.set('KERNEL_2','NZ_SOURCE',after='KERNEL_1')
	tbhdr.set('WINDOWS','SAMPLE',after='KERNEL_2')
	tbhdr.set('N_ZBIN_1',ntomo-1,'number of redshift/distance bins',after='WINDOWS')
	tbhdr.set('N_ZBIN_2',ntomo-1,'number of redshift/distance bins',after='N_ZBIN_1')
	tbhdr.set('N_ANG',ntheta,'number of angular bins',after='N_ZBIN_2')

	# - append the XIP extension to the output file created above
	pf.append(foutput,tbhdu.data,header=tbhdr)

	print("Doing XIM 2PTDATA extension...")

	# -------- now do XIM --------
	tbhdu=pf.BinTableHDU.from_columns([c1,c2,c3,c4m,c5])

	# - update the table's header
	tbhdr=tbhdu.header
	tbhdr.set('2PTDATA',True,before='TTYPE1')
	tbhdr.set('EXTNAME','XIM','name of this 2ptcorr table',after='2PTDATA')
	tbhdr.set('QUANT1','G-R','1st 2ptcorr quantity code',after='EXTNAME')
	tbhdr.set('QUANT2','G-R','2nd 2ptcorr quantity code',after='QUANT1')
	tbhdr.set('KERNEL_1','NZ_SAMPLE',after='QUANT2')
	tbhdr.set('KERNEL_2','NZ_SAMPLE',after='KERNEL_1')
	tbhdr.set('WINDOWS','SAMPLE',after='KERNEL_2')
	tbhdr.set('N_ZBIN_1',ntomo-1,'number of redshift/distance bins',after='WINDOWS')
	tbhdr.set('N_ZBIN_2',ntomo-1,'number of redshift/distance bins',after='N_ZBIN_1')
	tbhdr.set('N_ANG',ntheta,'number of angular bins',after='N_ZBIN_2')

	# - append the XIM extension to the output file created above
	pf.append(foutput,tbhdu.data,header=tbhdr)

###################################
# Do the NZDATA extension
###################################

# TODO: for now it calculates dNdz's for the lenses and the sources and everithing,
# I have to only save dNdz for sources if I only deal with ss in ctypes

nworkers = multiprocessing.cpu_count()-1
print("Doing NZDATA 2PTDATA extension...\nCalculating dNdz for the whole sample with %d workers..."%nworkers)

# NOTE: for test I think 10 cells can give us a very good dNdz for the sample
# NEST ordering
cell_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 120, 121, 122, 123, 124, 126, 285, 286, 287, 308, 309, 311, 349, 350, 351, 365, 366, 367, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 429, 430, 431, 440, 442, 443]
ncell = len(cell_ids)

nest  = True
nside = 8 # for healpix

kwargs = {'realization_numbers':realization_numbers, 'input_dir':input_dir, 'regime':regime, 'limiting_imag':limiting_imag, 'nest':nest, 'nside':nside, 'zcuts':zcuts}
zdata_all, zdata_lens, zdata_source = get_zdata_by_multiprocessing(cell_ids,nworkers,**kwargs)

# - computing the histograms for lenses and sources
lower, upper = 0.0, 2.0 # the lower and upper range of the bins 
nzbin = 400  # number of redshift bins for the whole distribution
dNdz_all, bin_edges_all = np.histogram(zdata_all,    range=(lower,upper), bins=nzbin, density=True) 
dNdz_l,   bin_edges_l   = np.histogram(zdata_lens,   range=(lower,upper), bins=nzbin, density=True) # make sure the cut is consistent
dNdz_s,   bin_edges_s   = np.histogram(zdata_source, range=(lower,upper), bins=nzbin, density=True) # make sure the cut is consistent

# -------- do NZ_LENS first --------

print("Doing NZ_ALL extension...")

# - defining column names (header)
colnames = ['Z_LOW','Z_MID','Z_HIGH']+['BIN'+str(num) for num in range(1,ntomo+1)] # ntomo bins :: make it 1-indexed

zlow  = bin_edges_all[:-1]  # exclude the last edge
zhigh = bin_edges_all[1:]   # exclude the first edge
zmid  = 0.5*(zlow+zhigh)    # bin centers

ncol, nrow = len(colnames), nzbin
coldata = np.zeros((ncol,nrow),dtype=float)
coldata[0],coldata[1],coldata[2] = zlow, zmid, zhigh

# - assigning counts to each tomographic bin
for k in range(3,ncol):
	kk = k-3 # start from index 0 for clustering
	zlim = [zcuts[kk], zcuts[kk+1]]
	idx_ztomo = (zmid>zlim[0])&(zmid<=zlim[1])
	coldata[k][idx_ztomo] = dNdz_all[idx_ztomo]

# - filling the fits columns with the data we just created
c=[]
for i in range(ncol):
	c.append(pf.Column(name=colnames[i],format='D',array=coldata[i]))

cols=pf.ColDefs(c)

# - create a binary table HDU object
tbhdu=pf.BinTableHDU.from_columns(cols)

# - update the table's header
tbhdr=tbhdu.header
tbhdr.set('NZDATA',True,before='TTYPE1')
tbhdr.set('EXTNAME','NZ_ALL','name of this n(z) table',after='NZDATA')
#tbhdr.set('NBIN',ntomo-1,'number of tomographic bins',after='EXTNAME')
#tbhdr.set('NZ',nrow,'number of n(z) bins',after='NBIN')
#'NGAL_1' etc.

# - append the NZDATA NZ_LENS extension to the output file created above
pf.append(foutput,tbhdu.data,header=tbhdr)


# -------- do NZ_LENS after --------

print("Doing NZ_LENS extension...")

# - defining column names (header)
colnames = ['Z_LOW','Z_MID','Z_HIGH']+['BIN'+str(num) for num in range(1,ntomo)] # ntomo-1 bins :: make it 1-indexed

zlow  = bin_edges_l[:-1]  # exclude the last edge
zhigh = bin_edges_l[1:]   # exclude the first edge
zmid  = 0.5*(zlow+zhigh)  # bin centers

ncol, nrow = len(colnames), nzbin
coldata = np.zeros((ncol,nrow),dtype=float)
coldata[0],coldata[1],coldata[2] = zlow, zmid, zhigh

# - assigning counts to each tomographic bin
for k in range(3,ncol):
	kk = k-3 # start from index 0 for lenses
	zlim = [zcuts[kk], zcuts[kk+1]]
	idx_ztomo = (zmid>zlim[0])&(zmid<=zlim[1])
	coldata[k][idx_ztomo] = dNdz_l[idx_ztomo]

# - filling the fits columns with the data we just created
c=[]
for i in range(ncol):
	c.append(pf.Column(name=colnames[i],format='D',array=coldata[i]))

cols=pf.ColDefs(c)

# - create a binary table HDU object
tbhdu=pf.BinTableHDU.from_columns(cols)

# - update the table's header
tbhdr=tbhdu.header
tbhdr.set('NZDATA',True,before='TTYPE1')
tbhdr.set('EXTNAME','NZ_LENS','name of this n(z) table',after='NZDATA')
#tbhdr.set('NBIN',ntomo-1,'number of tomographic bins',after='EXTNAME')
#tbhdr.set('NZ',nrow,'number of n(z) bins',after='NBIN')
#'NGAL_1' etc.

# - append the NZDATA NZ_LENS extension to the output file created above
pf.append(foutput,tbhdu.data,header=tbhdr)

# -------- now do NZ_SOURCE --------

print("Doing NZ_SOURCE extension...")

# - defining column names (header)
colnames = ['Z_LOW','Z_MID','Z_HIGH']+['BIN'+str(num) for num in range(1,ntomo)] # ntomo-1 bins :: make it 1-indexed

zlow  = bin_edges_s[:-1]   # exclude the last edge
zhigh = bin_edges_s[1:]    # exclude the first edge
zmid  = 0.5*(zlow+zhigh)   # bin centers

ncol, nrow = len(colnames), nzbin
#print(colnames,'ncol',ncol)
coldata = np.zeros((ncol,nrow),dtype=float)
coldata[0],coldata[1],coldata[2] = zlow, zmid, zhigh

# - assigning counts to each tomographic bin
for k in range(3,ncol):
	kk = k-2 # start from index 1 for sources
	zlim = [zcuts[kk], zcuts[kk+1]]
	idx_ztomo = (zmid>zlim[0])&(zmid<=zlim[1])
	coldata[k][idx_ztomo] = dNdz_s[idx_ztomo]

# - filling the fits columns with the data we just created
c=[]
for i in range(ncol):
	c.append(pf.Column(name=colnames[i],format='D',array=coldata[i]))

cols=pf.ColDefs(c)

# - create a binary table HDU object
tbhdu=pf.BinTableHDU.from_columns(cols)

# - update the table's header
tbhdr=tbhdu.header
tbhdr.set('NZDATA',True,before='TTYPE1')
tbhdr.set('EXTNAME','NZ_SOURCE','name of this n(z) table',after='NZDATA')

# - append the NZDATA NZ_SOURSE extension to the output file created above
pf.append(foutput,tbhdu.data,header=tbhdr)

t1 = datetime.datetime.now()
print( "\n"+time.strftime('%a %H:%M:%S')+" :: All done! Elapsed time: "+str(datetime.timedelta(seconds=round((t1-t0).seconds))) )