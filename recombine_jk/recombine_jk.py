from __future__ import print_function

## imports
import sys
import os
import time
import datetime
import multiprocessing

if (sys.version_info > (3, 0)):
	# Python 3 code in this block
	pyver = 3
	## adding a python env with packages like DESCQA treecorr etc...
	sys.path.insert(0, '/global/common/software/lsst/common/miniconda/py3-4.3.21/lib/python3.6/site-packages')
else:
	# Python 2 code in this block
	pyver = 2
	sys.path.insert(0, '/global/common/cori/contrib/lsst/apps/anaconda/py2-envs/DESCQA/lib/python2.7/site-packages')
	sys.path.insert(0, '/global/homes/e/erfanxyz/.local/lib/python2.7/site-packages')

# !!! since we pickled in py2 with treecorr imported, we should unpickle in py2 with treecorr imported

import os
import glob
import numpy as np
import cPickle as pickle
from collections import OrderedDict # to preserve the order of columns while writing to fits files
from itertools import combinations
from scipy.optimize import curve_fit

try:
	import treecorr
except(ImportError):
	import TreeCorr as treecorr # important in case you add treecorr through your PYTHONPATH for py3

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/util')
from util import *


# ------------------
# functions 
# ------------------

def bite_vector(input_vector, length):
	part = input_vector[0:length]
	trimmed_vector = input_vector[length:]
	return part, trimmed_vector

def usedir(mydir):
# https://stackoverflow.com/questions/12468022/python-fileexists-error-when-making-directory	
	if not mydir.endswith('/'): mydir += '/' # important
	try:
		if not os.path.exists(os.path.dirname(mydir)):
			os.makedirs(os.path.dirname(mydir))
	except OSError as err:
		print('[handled error]',err)

def choose(n,r):
	"""Computes n! / (r! (n-r)!) exactly. Returns a python long int."""
	# https://stackoverflow.com/questions/3025162/statistics-combinations-in-python/3027128
	assert n >= 0
	assert 0 <= r <= n

	# all integers in Python 3 are long. What was long in Python 2 is now the standard int type in Python 3.
	c = 1L # py2
	#c = 1 # py3

	denom = 1

	for (num,denom) in zip(range(n,n-r,-1), range(1,r+1,1)):
		c = (c * num) // denom
	return c

def append_vector(big_vector, to_be_appended, flatten=False):
	for h in range(len(to_be_appended)):
		if flatten:
			big_vector.append(to_be_appended[h])
		else:
			big_vector.append([to_be_appended[h]])
	return big_vector

def cumlen(array):
	clen = []
	lc = 0
	for chunk in array:
		lc += len(chunk)
		clen.append(lc)
	return clen

# wtheta_obs() will be used in add_IC
def wtheta_obs(theta,Aw,delta,C):
    wtheta_obs = Aw * ( theta**(-delta) - C )
    return wtheta_obs

def add_IC(theta,wtheta,werr,RR,i,j,verbose=True,method=2):

	# using the method similar to the one suggested in https://arxiv.org/abs/astro-ph/0205259 but with my own modifications
	if method==1:
		npt = len(theta)
		X = np.log10(theta[:npt//2])
		Y = np.log10(wtheta[:npt//2])
		dY = np.abs(werr[:npt//2]/(wtheta[:npt//2]*np.log(10))) # https://en.wikipedia.org/wiki/Propagation_of_uncertainty
		w0,log10_Aw = np.polyfit(X, Y, 1, w=1./dY)
		delta = -w0
		Aw = 10**log10_Aw
		C = np.sum( RR.npairs*theta**(-delta) ) / np.sum(RR.npairs)
		IC = Aw*C
		wtheta_corrected = wtheta+IC
		# if verbose: print('Corrected for the Integral Constraint:\n(i,j) = (%d,%d)\n(Aw, delta) = (%0.3f, %0.3f)\n(C,IC) = (%0.3f,%0.4f)'%(i,j,Aw,delta,C,IC))
	elif method==2:	# using the method suggested in https://arxiv.org/abs/astro-ph/0205259
		delta = 0.8
		C = np.sum( RR.npairs*theta**(-delta) ) / np.sum(RR.npairs)
		popt, pcov = curve_fit(lambda theta, Aw: wtheta_obs(theta,Aw,delta,C), theta, wtheta, sigma=werr) # only find Aw, other parameters fixed
		Aw = popt[0]
		IC = Aw*C
		wtheta_corrected = wtheta+IC
	elif method==3: # my thoughts
		popt, pcov = curve_fit(wtheta_obs, theta, wtheta, sigma=werr) # find all prameters
		Aw,delta,C = popt[0],popt[1],popt[2]
		IC = Aw*C
		wtheta_corrected = wtheta+IC

	return wtheta_corrected


def write_covmat(data_vector=None,regime=None,output_dir=None,npiece=None,k=None,nthetabin=None,vec_labels=None,rlist=None,full=None):
	# jkfactor = (npiece-k)*(npiece*num_realization-1)/(npiece*num_realization) # not sure about the appropriate factor for all realizations combined!!!
	jkfactor = (npiece-k)/k # at first was not sure about the appropriate factor for all realizations combined!!! but figured it out thanks to eq (9) in https://arxiv.org/pdf/1606.00233.pdf
	# see https://arxiv.org/pdf/1606.00233.pdf for hybrid jackknife where they average covmats (not good if you want to shrink later)
	# covariance matrix and errors
	covmat = jkfactor * np.cov(data_vector, ddof=0)#/npiece # np.cov: denominator is (N - 1) by default -> unbised estimatorbut for my application I am considering ddof=0 here for now, will take care of it in factors applied	
	if covmat.shape[0] != nthetabin*len(vec_labels): print('WARNING: covmat dimension %d and the number of labels %d times the number of theta bins %d do not match! (all realizations)'%(covmat.shape[0],len(vec_labels),nthetabin))
	err_vec = np.sqrt(covmat.diagonal()) #*np.sqrt(jkfactor)# 2002 MNRAS 337, 1282-1298, clustering, count and morphology of red galaxies
	# corrcoef returns the normalized covariance matrix, i.e. [Pearson] correlation matrix
	corrmat = np.corrcoef(data_vector)
	data_vector_avg = np.mean(data_vector, axis=-1)

	isfull = "full" if full else "selected for maximum likelihood analysis"
	print("\n"+time.strftime('%a %H:%M:%S')+" :: Saving correlation vectors ("+isfull+") and the corresponding covariance matrix for all the realizations combined:")
	bperc = '' if regime=='zsnb' else '_'+str(blend_percent)+'percent'
	tail = '_'+str(len(rlist))+'_realizations.full.npz' if full else '_'+str(len(rlist))+'_realizations.npz'
	output_fname = output_dir+regime+bperc+'/all_realizations_combined/'+'covmat.'+regime+bperc+tail
	usedir(output_fname.rsplit('/',1)[0])
	np.savez(output_fname, covmat=covmat, corrmat=corrmat, corrvec=data_vector_avg, corrsupervec=data_vector, labels=vec_labels, err_vec=err_vec, realizations=rlist)
	print('Saved ./'+output_fname.rsplit('/',1)[1])


def recombine_regions_by_multiprocessing(k=1, nworkers=None, cell_ids=None, **kwargs):
	# cell_ids = kwargs.pop('cell_ids') # cell_ids to delete from
	calc_total = kwargs['calc_total'] # do not use .pop() it uses it up, we need it later
	npiece = len(cell_ids)

	if not calc_total:
		pool = multiprocessing.Pool(processes=nworkers)
		n_choose_k = list(combinations(cell_ids, k))
		nwork = len(n_choose_k)
		if nwork<nworkers:
			print('Number of workers (%d) is more than number of works (%d) to do. nworkers is set to %d.'%(nworkers,nwork,nwork))
			nworkers = nwork
		ncks = np.array_split(n_choose_k, nworkers)
		njksamples = choose(npiece,k) # total for each realization - each worker is computing a part of it in the parallel mode
		jknums = [list(range(cumlen(ncks)[0]))] + [list(range(cumlen(ncks)[nw-1],cumlen(ncks)[nw])) for nw in range(1,nworkers)]
		# result is for all realizations passed to the function, I don't think it is necessary to know that for each realization individually for now
		# for now I did not split the realizations b/w cpus, might do it some day
		result, result_cosmo = zip(*pool.map(recombine_regions, [(nw, nworkers, nck, jknums[nw], njksamples, npiece, kwargs) for nw,nck in enumerate(ncks)])) 
		# print('-- len res cos --',len(result_cosmo),result_cosmo)
		pool.close()
		# concat_result, concat_result_cosmo = np.concatenate(result), np.concatenate(result_cosmo)
		dv, dv_cosmo = np.concatenate(result,axis=1), np.concatenate(result_cosmo,axis=1)

		# make vector labels once :: should look at zbindict for consistency
		vec_labels, vec_labels_cosmo = [], []
		if 'pp' in ctypes: _ = [vec_labels.append(r'$w('+str(i+1)+','+str(j+1)+')$') for i in range(ntomo) for j in range(ntomo) if j<=i]
		if 'ps' in ctypes: _ = [vec_labels.append(r'$\gamma_t('+str(i+1)+','+str(j+1)+')$') for i in range(ntomo) for j in range(ntomo) if j!=i]
		if 'ps' in ctypes: _ = [vec_labels.append(r'$\gamma_{\times}('+str(i+1)+','+str(j+1)+')$') for i in range(ntomo) for j in range(ntomo) if j!=i]
		if 'ss' in ctypes: _ = [vec_labels.append(r'$\xi_{+}('+str(i+1)+','+str(j+1)+')$') for i in range(ntomo) for j in range(ntomo) if j<=i]
		if 'ss' in ctypes: _ = [vec_labels.append(r'$\xi_{-}('+str(i+1)+','+str(j+1)+')$') for i in range(ntomo) for j in range(ntomo) if j<=i]
		# -----------------------
		if 'pp' in ctypes: _ = [vec_labels_cosmo.append(r'$w('+str(i+1)+','+str(j+1)+')$') for i in range(ntomo) for j in range(ntomo) if j==i]
		if 'ps' in ctypes: _ = [vec_labels_cosmo.append(r'$\gamma_t('+str(i+1)+','+str(j+1)+')$') for i in range(ntomo) for j in range(ntomo) if j<i]
		if 'ss' in ctypes: _ = [vec_labels_cosmo.append(r'$\xi_{+}('+str(i+1)+','+str(j+1)+')$') for i in range(1,ntomo) for j in range(1,ntomo) if j<=i]
		if 'ss' in ctypes: _ = [vec_labels_cosmo.append(r'$\xi_{-}('+str(i+1)+','+str(j+1)+')$') for i in range(1,ntomo) for j in range(1,ntomo) if j<=i]
		# -----------------------
		write_covmat(data_vector=dv,regime=regime,output_dir=output_dir,npiece=npiece,k=k,nthetabin=nthetabin,vec_labels=vec_labels,rlist=realization_numbers,full=True)
		write_covmat(data_vector=dv_cosmo,regime=regime,output_dir=output_dir,npiece=npiece,k=k,nthetabin=nthetabin,vec_labels=vec_labels_cosmo,rlist=realization_numbers,full=False)

	else:
		print("Note: %d worker(s) doing the work to calculate total correlation functions"%nworkers)
		recombine_regions([0, 1, [9999], 99, 99, npiece, kwargs]) # it writes a fits file

def recombine_regions(args):

	worker_id, nworkers, n_choose_k, jknums, njksamples, npiece, kwargs = args

	realization_numbers = kwargs.pop('realization_numbers')
	output_dir = kwargs.pop('output_dir')
	regime = kwargs.pop('regime')
	ctypes = kwargs.pop('ctypes')
	zcuts = kwargs.pop('zcuts')
	calc_total = kwargs.pop('calc_total')
	verbose = kwargs.pop('verbose')
	savefits = kwargs.pop('savefits')
	nthetabin = kwargs.pop('nthetabin')

	num_realization = len(realization_numbers)
	ntomo = len(zcuts)-1
	zbindict = {'pp': [[i,j] for i in range(ntomo) for j in range(ntomo) if j<=i],
			    'ps': [[i,j] for i in range(ntomo) for j in range(ntomo) if j!=i],
			    'ss': [[i,j] for i in range(ntomo) for j in range(ntomo) if j<=i]}

	wt_vec_ij, wt_noIC_vec_ij, wt_perrvec_ij, gt_vec_ij, gx_vec_ij, gtx_perrvec_ij, xip_vec_ij, xim_vec_ij, xipm_perrvec_ij, npairs_dd_ij, npairs_dr_ij, npairs_rd_ij, npairs_rr_ij, npairs_ng_ij, npairs_gg_ij = [np.empty((ntomo,ntomo), dtype=object) for _ in range(15)]

	# --- write the grand values of correlations (using all jk regions) in a separate fits file ---
	if not calc_total:
		correlation_supervector_all_realizations, correlation_supervector_all_realizations_cosmo = [], []
		# if not parallel: # n_choose_k = list(combinations(cell_ids, k))
		# npiece = len(cell_ids)
		# njksamples = choose(npiece,k) # total for each realization- we are computing a part of it for each core in parallel modes
		# print('Number of jackknife samples is C(%d,%d) = %d for each realization'%(npiece,k,njksamples)) 
		#NOTE: doing list(iterator) or putting it in a loop consumes the iterator and we have to reset it
	else: # calculate correlation function for full sample
		n_choose_k = [9999] # this way we only have an iteration with a dummy number 99
		print('Recombining all %d pieces to calculate the overall correlation function for %d realization(s)...'%(npiece,len(realization_numbers)))

	for rn,realization_number in enumerate(realization_numbers):

		print('Realization # %d, chunk %d out of %d is being analyzed with worker # %d...'%(realization_number,worker_id+1,nworkers,worker_id))
		# --- write the grand values of correlations (using all jk regions) in a separate fits file ---
		if not calc_total:
			correlation_supervector, correlation_supervector_cosmo = [], []
			wt_supervec, wt_supervec_cosmo, gt_supervec, gt_supervec_cosmo, gx_supervec, xip_supervec, xim_supervec = [], [], [], [], [], [], []
		
		bperc = '' if regime=='zsnb' else '_'+str(blend_percent)+'percent'

		for jknum_idx,leftouts in enumerate(n_choose_k):
			if not calc_total:
				jknum = jknums[jknum_idx] 
				if verbose and not calc_total: print('Sample',jknum+1,'out of total',njksamples,'----> leaving out',leftouts)
				bperc = '' if regime=='zsnb' else '_'+str(blend_percent)+'percent'
				tag = '' if calc_total else '_jk'+str(jknum)
				output_fname = output_dir+regime+bperc+'/r'+str(realization_number)+'/jkcomb/corrs.'+regime+bperc+'.grand'+'_r'+str(realization_number)+tag+'.fit'
				usedir(output_fname.rsplit('/',1)[0])
				clobber=True # create a new file for each realization

			# _vec: individual, _supervec: for all samples, _cosmo: selected ones for maximum likelihood analysis
			if not calc_total: wt_vec, wt_vec_cosmo, gt_vec, gt_vec_cosmo, gx_vec, xip_vec, xim_vec, xip_vec_cosmo, xim_vec_cosmo = [], [], [], [], [], [], [], [], []

			for cti, ctype in enumerate(ctypes):
				for ij,[i,j] in enumerate(zbindict[ctype]):
					# ------------------------------------------------------------
					BASE_PATH = output_dir+regime+bperc+'/r'+str(realization_number)+'/'+ctype+'_'+str(i)+'_'+str(j)+'/' #+regime+'.r'+str(realization_number)+'.pp_'+str(i)+'_'+str(j)+'.hp_'+str(c1)+'_'+str(c2)+'.pkl' 
					if not calc_total:
						files_excluded = []
						for leftout in leftouts:
							patterns = ('*hp_'+str(leftout)+'_*.pkl', '*_'+str(leftout)+'.pkl') # the tuple of pattterns of interest
							for name in patterns: files_excluded.extend(glob.glob(BASE_PATH+name))
					files_all = [BASE_PATH+fn for fn in os.listdir(BASE_PATH)]
					files_included = files_all if calc_total else [f for f in files_all if f not in files_excluded]
					# ---------------------------- [pp] --------------------------
					if ctype == 'pp':
						if verbose:
							# if ij==0: sys.stdout.write('recombining ')
							sys.stdout.write('.'); sys.stdout.flush()
						for fi, file in enumerate(files_included):
							with open(file, "rb") as f: DD_,RR_,DR_,RD_ = pickle.load(f) # pickle doesn't actually store information about how a class/object is constructed, and needs access to the class when unpickling (so import treecorr)
							if fi==0:
								DD, RR, DR, RD = DD_.copy(), RR_.copy(), DR_.copy(), RD_.copy()
							else:
								DD += DD_
								RR += RR_
								DR += DR_
								RD += RD_
						DD.finalize()
						RR.finalize()
						DR.finalize()
						RD.finalize()
						theta = np.exp(DD.meanlogr)
						wtheta, var = DD.calculateXi(RR,DR,RD) # var gives the Poisson error
						err_poisson = np.sqrt(var)
						# Correcting for the Integral Constraint based on http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?2002MNRAS.337.1282R&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf
						# wtheta_true = wtheta_obs + IC = Aw * theta^(-delta), IC = Aw*C

						if calc_total:
							# err_jk,err_vec = bite_vector(err_vec, len(wtheta)) # could use err_jk instead of err_poisson but I thought the deviation from a point at large angle at 10^-6 (not corrected) is different from the same at 10^-4 (corrected)
							wtheta_noIC = wtheta.copy()
							if i==j: wtheta = add_IC(theta,wtheta,err_poisson,RR,i,j,verbose=verbose,method=2) # otherwise will encounter log of zero or negative
							theta_nn = theta_nn if 'theta_nn' in vars() else theta.copy() # one time assignment is enough
							npairs_dd_ij[i][j] = npairs_dd_ij[i][j]+[DD.npairs] if npairs_dd_ij[i][j] is not None else [DD.npairs]
							npairs_dr_ij[i][j] = npairs_dr_ij[i][j]+[DR.npairs] if npairs_dr_ij[i][j] is not None else [DR.npairs]
							npairs_rd_ij[i][j] = npairs_rd_ij[i][j]+[RD.npairs] if npairs_rd_ij[i][j] is not None else [RD.npairs]
							npairs_rr_ij[i][j] = npairs_rr_ij[i][j]+[RR.npairs] if npairs_rr_ij[i][j] is not None else [RR.npairs]
							wt_vec_ij[i][j] = [wtheta] if rn==0 else append_vector(wt_vec_ij[i][j], [wtheta], flatten=True)
							wt_noIC_vec_ij[i][j] = [wtheta_noIC] if rn==0 else append_vector(wt_noIC_vec_ij[i][j], [wtheta_noIC], flatten=True)
							wt_perrvec_ij[i][j] = [err_poisson.copy()] if rn==0 else append_vector(wt_perrvec_ij[i][j], [err_poisson], flatten=True)
						else:
							wtheta_noIC = wtheta.copy()
							if i==j: wtheta = add_IC(theta,wtheta,err_poisson,RR,i,j,verbose=verbose,method=2)
							pp_dict = {'theta':theta, 'wtheta':wtheta, 'wtheta_noIC':wtheta_noIC, 'err':err_poisson, 'zbin1':np.repeat(i,len(wtheta)), 'zbin2':np.repeat(j,len(wtheta)), 'npairs_dd':DD.npairs, 'npairs_dr':DR.npairs, 'npairs_rd':RD.npairs, 'npairs_rr':RR.npairs}
							order_of_keys = ['theta','wtheta','wtheta_noIC','err','zbin1','zbin2','npairs_dd','npairs_dr','npairs_rd','npairs_rr'] 
							pp_dict = OrderedDict([(key, pp_dict[key]) for key in order_of_keys])
							wt_vec = append_vector(wt_vec, wtheta_noIC) # important!!! do you want your covmat with IC correction or not? I guess IC is not very precise so far so no
							if i==j: wt_vec_cosmo = append_vector(wt_vec_cosmo, wtheta_noIC) # important!!! do you want

						if not calc_total and savefits: gen_write(output_fname, list(pp_dict.keys()), list(pp_dict.values()), extname='pp_'+str(i)+'_'+str(j), clobber=clobber) # create/append
						clobber = False # to allow overwriting later
					if not calc_total: clobber = False if os.path.isfile(output_fname) else True 
					# ---------------------------- [ps] --------------------------
					if ctype == 'ps':
						if verbose:
							# if ij==0: sys.stdout.write('recombining ps ')
							sys.stdout.write("."); sys.stdout.flush()#print('ps',jknum)
						
						nvar = 0
						for fi, file in enumerate(files_included):
							with open(file, "rb") as f: ng_, varg_ = pickle.load(f) # pickle doesn't actually store information about how a class/object is constructed, and needs access to the class when unpickling (so import treecorr)
							if fi==0:
								ng = ng_.copy()
								if not np.isnan(varg_):
									varg = varg_.copy()
									nvar+=1
								else:
									varg = 0
							else:
								ng += ng_
								if not np.isnan(varg_):
									varg += varg_
									nvar += 1

						# print(varg,nvar)
						varg /= nvar # the right thing to do was to get the variance of shear from the whole catalog but this one must be pretty close to that since the Universe is homogenous in large scales (50 sq. deg. cells) - they're e.g. 0.0835257899988, 0.0835257899988, 0.083336168123, 0.083336168123
						ng.finalize(varg)
						gammat, gammax = ng.xi, ng.xi_im
						err_poisson = np.sqrt(ng.varxi)

						if calc_total:
							theta_ng = theta_ng if 'theta_ng' in vars() else np.exp(ng.meanlogr) # one time assignment is enough
							npairs_ng_ij[i][j] = npairs_ng_ij[i][j]+[ng.npairs] if npairs_ng_ij[i][j] is not None else [ng.npairs]
							gt_vec_ij[i][j] = [gammat] if rn==0 else append_vector(gt_vec_ij[i][j], [gammat], flatten=True)
							gx_vec_ij[i][j] = [gammax] if rn==0 else append_vector(gx_vec_ij[i][j], [gammax], flatten=True)
							gtx_perrvec_ij[i][j] = [err_poisson.copy()] if rn==0 else append_vector(gtx_perrvec_ij[i][j], [err_poisson], flatten=True)
						else:
							ps_dict = {'theta':np.exp(ng.meanlogr), 'gammat':gammat, 'gammax':gammax, 'err':err_poisson, 'zbin1':np.repeat(i,len(ng.xi)), 'zbin2':np.repeat(j,len(ng.xi)),'npairs':ng.npairs}
							order_of_keys = ['theta','gammat','gammax','err','zbin1','zbin2','npairs'] # ,'err_jk'
							ps_dict = OrderedDict([(key, ps_dict[key]) for key in order_of_keys])
							gt_vec = append_vector(gt_vec, gammat)
							gx_vec = append_vector(gx_vec, gammax)
							if j<i: gt_vec_cosmo = append_vector(gt_vec_cosmo, gammat)

						if not calc_total and savefits: gen_write(output_fname, list(ps_dict.keys()), list(ps_dict.values()), extname='ps_'+str(i)+'_'+str(j), clobber=clobber) # append
						clobber = False # to allow overwriting later
					if not calc_total: clobber = False if os.path.isfile(output_fname) else True 
					# ---------------------------- [ss] --------------------------
					if ctype == 'ss':
						if verbose:
							# if ij==0: sys.stdout.write('recombining ss ')
							sys.stdout.write("."); sys.stdout.flush() #print('ss',jknum)
						
						nvar = 0
						for fi, file in enumerate(files_included):
							with open(file, "rb") as f: gg_, varg1_, varg2_ = pickle.load(f) # pickle doesn't actually store information about how a class/object is constructed, and needs access to the class when unpickling (so import treecorr)
							if fi==0:
								gg = gg_.copy()
								if not np.isnan(varg1_):
									varg1, varg2 = varg1_.copy(), varg2_.copy()
									nvar+=1
								else:
									varg1, varg2 = 0, 0
							else:
								gg += gg_
								if not np.isnan(varg1_):
									varg1 += varg1_
									varg2 += varg2_
									nvar  += 1

						varg1 /= nvar # the right thing to do was to get the variance of shear from the whole catalog but this one must be pretty close to that
						varg2 /= nvar

						gg.finalize(varg1,varg2) # pass (1,1) if you don't use them
						xip, xim = gg.xip, gg.xim
						err_poisson = np.sqrt(gg.varxi)

						if calc_total:
							theta_gg = theta_gg if 'theta_gg' in vars() else np.exp(gg.meanlogr) # one time assignment is enough
							npairs_gg_ij[i][j] = npairs_gg_ij[i][j]+[gg.npairs] if npairs_gg_ij[i][j] is not None else [gg.npairs]
							xip_vec_ij[i][j] = [xip] if rn==0 else append_vector(xip_vec_ij[i][j], [xip], flatten=True)
							xim_vec_ij[i][j] = [xim] if rn==0 else append_vector(xim_vec_ij[i][j], [xim], flatten=True)
							xipm_perrvec_ij[i][j] = [err_poisson.copy()] if rn==0 else append_vector(xipm_perrvec_ij[i][j], [err_poisson], flatten=True)
						else:
							ss_dict = {'theta':np.exp(gg.meanlogr), 'xip':xip, 'xim':xim, 'err':err_poisson, 'zbin1':np.repeat(i,len(gg.xip)), 'zbin2':np.repeat(j,len(gg.xip)),'npairs':gg.npairs}
							order_of_keys = ['theta','xip','xim','err','zbin1','zbin2','npairs'] # ,'err_jk'
							ss_dict = OrderedDict([(key, ss_dict[key]) for key in order_of_keys])
							xip_vec = append_vector(xip_vec, xip)
							xim_vec = append_vector(xim_vec, xim)
							if i!=0 and j!=0: # first bin is not sheared that much
								xip_vec_cosmo = append_vector(xip_vec_cosmo, xip)
								xim_vec_cosmo = append_vector(xim_vec_cosmo, xim)

							if not calc_total and savefits: gen_write(output_fname, list(ss_dict.keys()), list(ss_dict.values()), extname='ss_'+str(i)+'_'+str(j), clobber=clobber) # append
						clobber = False # to allow overwriting later
					if not calc_total: clobber = False if os.path.isfile(output_fname) else True 

			if not calc_total and verbose and savefits: print('\nSaved ./'+output_fname.rsplit('/',1)[1])

			# -------------------------------------------------------------
			if not calc_total:
				# for this jk region
				pp_ps_ss_vec = wt_vec+gt_vec+gx_vec+xip_vec+xim_vec # it strings them together with the desirable order (should be consistent with vec_labels)
				pp_ps_ss_vec_cosmo = wt_vec_cosmo+gt_vec_cosmo+xip_vec_cosmo+xim_vec_cosmo

				if jknum_idx==0:
					correlation_supervector = pp_ps_ss_vec[:] # shallow copy
					correlation_supervector_cosmo = pp_ps_ss_vec_cosmo[:]
					# ---------------------
					# super vectors contain the correlation info for all samples
					# wt_supervec, wt_supervec_cosmo, gt_supervec, gt_supervec_cosmo, gx_supervec, xip_supervec, xip_supervec_cosmo, xim_supervec, xim_supervec_cosmo = wt_vec, wt_vec_cosmo, gt_vec, gt_vec_cosmo, gx_vec, xip_vec, xip_vec_cosmo, xim_vec, xim_vec_cosmo
					# ---------------------
				else:
					correlation_supervector = np.concatenate((correlation_supervector, pp_ps_ss_vec), axis=-1) # -1 implies the last dimension
					correlation_supervector_cosmo = np.concatenate((correlation_supervector_cosmo, pp_ps_ss_vec_cosmo), axis=-1) # -1 implies the last dimension

			if not calc_total: # combining all realizations
				if rn==0:
					correlation_supervector_all_realizations = correlation_supervector[:]
					correlation_supervector_all_realizations_cosmo = correlation_supervector_cosmo[:]
				else:
					correlation_supervector_all_realizations = np.concatenate((correlation_supervector_all_realizations, correlation_supervector), axis=-1) # -1 implies the last dimension
					correlation_supervector_all_realizations_cosmo = np.concatenate((correlation_supervector_all_realizations_cosmo, correlation_supervector_cosmo), axis=-1) # -1 implies the last dimension

	if calc_total:

		bperc = '' if regime=='zsnb' else '_'+str(blend_percent)+'percent'
		output_fname = output_dir+regime+bperc+'/all_realizations_combined/'+'corrs.'+regime+bperc+'.grand_'+str(len(realization_numbers))+'_realizations.fit'
		usedir(output_fname.rsplit('/',1)[0])

		# covmat_fname = output_dir+regime+'/r'+str(realization_number)+'/jkcomb/'+'covmat.'+regime+'_r'+str(realization_number)+'.npz'
		covmat_fname = output_dir+regime+bperc+'/all_realizations_combined/'+'covmat.'+regime+bperc+'_'+str(len(realization_numbers))+'_realizations.full.npz'
		covmat_data = np.load(covmat_fname)
		err_vec = covmat_data['err_vec']
		# MAYBE I SHOULD MAKE MORE IN CASE I AM PRODUCING THING FOR ONLY ONE PROBE
		trash,err_vec_1 = bite_vector(err_vec,   len(zbindict['pp'])*nthetabin) if 'pp' in ctype else None, err_vec.copy() # WTHETA DROPPED if it is in covmat
		trash,err_vec_2 = bite_vector(err_vec_1, len(zbindict['ps'])*nthetabin) if 'ps' in ctype else None, err_vec_1.copy()  # ALSO GAMMAT DROPPED if it is in covmat
		trash,err_vec_3 = bite_vector(err_vec_2, len(zbindict['ps'])*nthetabin + len(zbindict['ss'])*nthetabin) if 'ps' in ctype else None, err_vec_2.copy() # WTHETA, GAMMAT, GAMMAX AND XIP DROPPED depending on their existance in covmat

		if savefits: gen_write(output_fname, ['realization_numbers'], [np.array(realization_numbers)], dtypes=['i8'], extname='realization_numbers', clobber=True) # append
		clobber = False

		for ctype in ctypes:
			for [i,j] in zbindict[ctype]:

				if ctype=='pp':
					npairs_dd = np.sum(npairs_dd_ij[i][j], axis=0) # you can also use mean to show average npairs per realization
					npairs_dr = np.sum(npairs_dr_ij[i][j], axis=0)
					npairs_rd = np.sum(npairs_rd_ij[i][j], axis=0)
					npairs_rr = np.sum(npairs_rr_ij[i][j], axis=0)
					wtheta = np.mean(wt_vec_ij[i][j], axis=0)
					wtheta_noIC = np.mean(wt_noIC_vec_ij[i][j], axis=0)
					err_poisson = np.mean(wt_perrvec_ij[i][j], axis=0)
					err_wt_jk,err_vec = bite_vector(err_vec, len(wtheta))
					pp_dict = {'theta':theta_nn, 'wtheta':wtheta, 'wtheta_noIC':wtheta_noIC, 'err':err_poisson, 'err_jk':err_wt_jk, 'zbin1':np.repeat(i,len(wtheta)), 'zbin2':np.repeat(j,len(wtheta)), 'npairs_dd':npairs_dd, 'npairs_dr':npairs_dr, 'npairs_rd':npairs_rd, 'npairs_rr':npairs_rr}
					order_of_keys = ['theta','wtheta','wtheta_noIC','err','err_jk','zbin1','zbin2','npairs_dd','npairs_dr','npairs_rd','npairs_rr']
					pp_dict = OrderedDict([(key, pp_dict[key]) for key in order_of_keys])
					if savefits: gen_write(output_fname, list(pp_dict.keys()), list(pp_dict.values()), extname='pp_'+str(i)+'_'+str(j), clobber=clobber) # append
				elif ctype=='ps':
					npairs_ng = np.sum(npairs_ng_ij[i][j], axis=0)
					gammat = np.mean(gt_vec_ij[i][j], axis=0)
					gammax = np.mean(gx_vec_ij[i][j], axis=0)
					err_poisson = np.mean(gtx_perrvec_ij[i][j], axis=0)
					err_gt_jk,err_vec_1 = bite_vector(err_vec_1, len(gammat))
					err_gx_jk,err_vec_2 = bite_vector(err_vec_2, len(gammax))
					ps_dict = {'theta':theta_ng, 'gammat':gammat, 'gammax':gammax, 'err':err_poisson, 'err_gt_jk':err_gt_jk, 'err_gx_jk':err_gx_jk, 'zbin1':np.repeat(i,len(ng.xi)), 'zbin2':np.repeat(j,len(ng.xi)),'npairs':npairs_ng}
					order_of_keys = ['theta','gammat','gammax','err','err_gt_jk','err_gx_jk','zbin1','zbin2','npairs'] 
					ps_dict = OrderedDict([(key, ps_dict[key]) for key in order_of_keys])
					if savefits: gen_write(output_fname, list(ps_dict.keys()), list(ps_dict.values()), extname='ps_'+str(i)+'_'+str(j), clobber=clobber) # append
				elif ctype=='ss':
					npairs_gg = np.sum(npairs_gg_ij[i][j], axis=0)
					xip = np.mean(xip_vec_ij[i][j], axis=0)
					xim = np.mean(xim_vec_ij[i][j], axis=0)
					err_poisson = np.mean(xipm_perrvec_ij[i][j], axis=0)
					err_xip_jk,err_vec_2 = bite_vector(err_vec_2, len(xip))
					err_xim_jk,err_vec_3 = bite_vector(err_vec_3, len(xim))
					ss_dict = {'theta':theta_gg, 'xip':xip, 'xim':xim, 'err':err_poisson, 'err_xip_jk':err_xip_jk, 'err_xim_jk':err_xim_jk, 'zbin1':np.repeat(i,len(gg.xip)), 'zbin2':np.repeat(j,len(gg.xip)),'npairs':npairs_gg}
					order_of_keys = ['theta','xip','xim','err','err_xip_jk','err_xim_jk','zbin1','zbin2','npairs'] 
					ss_dict = OrderedDict([(key, ss_dict[key]) for key in order_of_keys])
					if savefits: gen_write(output_fname, list(ss_dict.keys()), list(ss_dict.values()), extname='ss_'+str(i)+'_'+str(j), clobber=clobber) # append
		
		print('\nSaved ./'+output_fname.rsplit('/',1)[1])

	if not calc_total: return correlation_supervector_all_realizations, correlation_supervector_all_realizations_cosmo

# ----------------
# main program
# ----------------

resampling = True
total_corr = True

output_dir = '/global/cscratch1/sd/erfanxyz/projects/blending/corrs/'
regime = 'zsb'
blend_percent = 5
realization_numbers = [0]
ctypes = ['pp','ps','ss'] 
nthetabin = 6 # raises warning if inconsistent

# cell_ids to delete from
cell_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 120, 121, 122, 123, 124, 126, 285, 286, 287, 308, 309, 311, 349, 350, 351, 365, 366, 367, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 429, 430, 431, 440, 442, 443]

zcuts = [0.0,0.25,0.5,0.75,1.0] #[0.0,0.3,0.6,0.9,1.2]
ntomo = len(zcuts)-1
k = 1 # k in C(n,k) or "n choose k" for jackknife --- k=1 leads to a leave-one-out jackknife

n_choose_k = list(combinations(cell_ids, k))
npiece = len(cell_ids)
njksamples = choose(npiece,k)
nworkers = multiprocessing.cpu_count()-1

t0 = datetime.datetime.now()
print("\n"+time.strftime('%a %H:%M:%S')+" :: Starting with",nworkers,"workers ...\n\nRegime:",regime,"\nCorrelation types:",ctypes)
print('Redshift pinpoints:',zcuts)
print('Number of tomographic bins:',ntomo)
print('Number of angular bins:',nthetabin)
print('Number of jackknife samples is C(%d,%d) = %d for each realization'%(npiece,k,njksamples)) 

# leave one out - runs N times to create N samples where N is the number of cells
if resampling:
	kwargs = {'realization_numbers':realization_numbers, 'output_dir':output_dir, 'regime':regime, 'ctypes':ctypes, 'zcuts':zcuts, 'calc_total':False, 'verbose':True, 'savefits':False, 'nthetabin':nthetabin}
	recombine_regions_by_multiprocessing(k=1, nworkers=nworkers, cell_ids=cell_ids, **kwargs)

# total - runs one time and won't leave any cell out
if total_corr: # do resampling first to have the covmat
	kwargs_tot = {'realization_numbers':realization_numbers, 'output_dir':output_dir, 'regime':regime, 'ctypes':ctypes, 'zcuts':zcuts, 'calc_total':True, 'verbose':True, 'savefits':True, 'nthetabin':nthetabin}
	recombine_regions_by_multiprocessing(k=1, nworkers=1, cell_ids=cell_ids, **kwargs_tot) # did not make it parallel yet since it was fast enough; so nworkers=1

t1 = datetime.datetime.now()
print( "\n"+time.strftime('%a %H:%M:%S')+" :: All done! Elapsed time: "+str(datetime.timedelta(seconds=round((t1-t0).seconds))) )
