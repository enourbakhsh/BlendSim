from __future__ import print_function
## imports
import sys
import os

if (sys.version_info > (3, 0)):
	# Python 3 code in this block
	pyver = 3
	sys.path.insert(0, '/global/common/software/lsst/common/miniconda/py3-4.3.21/lib/python3.6/site-packages')
else:
	# Python 2 code in this block
	pyver = 2
	sys.path.insert(0, '/global/common/cori/contrib/lsst/apps/anaconda/py2-envs/DESCQA/lib/python2.7/site-packages') # activate it otherwise won't even start python
	sys.path.insert(0, '/global/homes/e/erfanxyz/.local/lib/python2.7/site-packages')

import numpy as np
import healpy as hp
import fitsio as fio
import time
import datetime

from itertools import compress
from collections import OrderedDict, Counter # OrderedDict to preserve the order of columns while writing to fits files
import cPickle as pickle
import random

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/Jupytermeter') # loading bar
from jupytermeter import *

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/util')
from util import *

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/footprinter')
import footprinter as fp

try:
	# import nonsense # to provoke python to run in serial
	from mpi4py import MPI
	# for parallel jobs
	COMM = MPI.COMM_WORLD
	size = COMM.Get_size()
	rank = COMM.Get_rank()
	parallel = True
except(ImportError):
	print("*** Error in importing/utilizing MPI, all jobs will be executing on a single core...")
	size = 1
	rank = 0
	parallel = False
	pass

import treecorr

##############################################################################
# main program to calculate correlation functions
##############################################################################

def calc_corrs(regime='zsnb', pp=True, ps=True, ss=True, slurm_jobid=0, num_threads=0, realization_number=0, blend_percent=10):

	corrfns_abbr = ''
	if pp: corrfns_abbr = corrfns_abbr.join('pp')
	if ps: corrfns_abbr = corrfns_abbr.join('ps')
	if ss: corrfns_abbr = corrfns_abbr.join('ss')

	realization_number = int(realization_number)
	fdir  = os.path.dirname(os.path.abspath(__file__)) # the directory of the script being run
	fdir += '/tmp/'+slurm_jobid+'_'+regime+'_'+corrfns_abbr+'_running/'
	tdir = fdir+'iranks_taken/'
	usedir(tdir) # to put progress info

	if not parallel and rank!=0: return # avoid doing the same thing by different ranks when mpi4py is not imported
	t0 = datetime.datetime.now()
	regimes = ['zsnb','zsb','zpnb','zpb'] # spec-z/photo-z :: blend/no-blend
	if regime not in regimes:
		valid = '{'+', '.join(regimes)+'}'
		raise ValueError("'%s' is not one of the valid regimes: "%regime + valid)

	nest = True
	nside = 8 # for healpix
	hpinfo = {'nside': nside, 'nest': nest}
	eps = 0.4 # degrees from the edges of the simulation

	# edges of the simulation
	rai, raf = [0,180] 
	deci, decf = [0,90] 

	# generate randoms such that at least one random position falls into each healpix cell for the iven resolution nside = 8 -> so 1000 is fine
	cell_ids_in_a_quarter = radeclim2cell([rai,raf], [deci,decf],num_points=256000,pixinfo=hpinfo) # Buzzard galaxies are all in a quarter of sky

	# find the edge cells to remove them later
	cell_ids_in_stripe = radeclim2cell([rai,rai+eps], [deci,decf],num_points=256000,pixinfo=hpinfo) 
	cell_ids_in_stripe.extend( radeclim2cell([rai,raf], [deci,deci+eps],num_points=256000,pixinfo=hpinfo) )
	cell_ids_in_stripe.extend( radeclim2cell([raf-eps,raf], [deci,decf],num_points=256000,pixinfo=hpinfo) )
	cell_ids_in_stripe.extend( radeclim2cell([rai,raf], [decf-eps,decf],num_points=256000,pixinfo=hpinfo) )

	blacklist = list(set(cell_ids_in_stripe)) # initial blacklist: cells close to the edges

	cell_ids = [aa for aa in cell_ids_in_a_quarter if aa not in blacklist]
	cell_ids.sort() # to have identical sorted lists for all cores

	# cell numbers in NEST ordering - all ids in a quarter of sky excluding the edge cells
	# cell_ids_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 120, 121, 122, 123, 124, 126, 285, 286, 287, 308, 309, 311, 349, 350, 351, 365, 366, 367, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 429, 430, 431, 440, 442, 443]

	if np.array_equal(cell_ids_, cell_ids):
		if rank==size-1: print('Cell ids are selected successfully!')
	else:
		print('Warning: bad cell_id selection in rank %d.'%rank) # in case other ranks lead to different numbers because of randomization

	if rank==size-1:
		print('Cell ids =',cell_ids)

	ncell = len(cell_ids)

	zcuts = [0.0,0.25,0.5,0.75,1.0]
	ntomo = len(zcuts)-1

	pp_i = [i_ for i_ in range(ntomo) for  _ in range(i_+1)] if pp else []
	pp_j = [i_ for j_ in range(ntomo) for i_ in range(j_+1)] if pp else []
	ps_i = [i_ for  _ in range(ntomo-1) for i_ in range(ntomo)] if ps else [] # no ggl for lens and source from the same zbin (also below)
	ps_j = sum([list(np.roll(range(ntomo), l)) for l in range(1,ntomo)], []) if ps else [] # sum is inefficient only for a very huge list (we are ok)
	ss_i = [i_ for i_ in range(ntomo) for  _ in range(i_+1)] if ss else [] #pp_i #[i_ for i_ in range(ntomo) for  _ in range(i_+1)]
	ss_j = [i_ for j_ in range(ntomo) for i_ in range(j_+1)] if ss else [] #pp_j #[i_ for j_ in range(ntomo) for i_ in range(j_+1)]

	njob_per_cxc = len(pp_i)+len(ps_i)+len(ss_i) # num of jobs per cross corr b/w healpix cells
	pp_iranks = range(len(pp_i)) if parallel else [0] # or (n*(n+1))/2
	ps_iranks = range(len(pp_i),len(pp_i)+len(ps_i)) if parallel else [0]
	ss_iranks = range(len(pp_i)+len(ps_i),njob_per_cxc) if parallel else [0]

	##############################################################################
	# setting inputs/outputs and preparing for the main run
	##############################################################################

	limiting_imag = 23.0
	sigma_SN = {'nb':0.243,'b':0.257}
	shape_noise = sigma_SN['nb'] if regime[2:]=='nb' else sigma_SN['b']

	input_dir  = '/global/cscratch1/sd/erfanxyz/projects/blending/buzzard_v1.6_lsst_y1/'
	output_dir = '/global/cscratch1/sd/erfanxyz/projects/blending/corrs/'

	# get the adjacent pairs (of healpix cells) to carry out the jackknife resampling
	pairs_of_cell_ids, nadjpairs = get_pairs(nside,cell_ids,blacklist,nest=True,unique=False) #get_unique_pairs(nside,cell_ids,blacklist,nest=True)
 	if rank==size-1: print('\nPairs of cell ids =',pairs_of_cell_ids,'\n')

	njob = nadjpairs*njob_per_cxc

	corrfns = ''
	if pp: corrfns = corrfns.join('position-position ')
	if ps: corrfns = corrfns.join('position-shear ')
	if ss: corrfns = corrfns.join('shear-shear ')
	if not os.path.isfile(fdir+'ctype-njob.pkl'):
		with open(fdir+'ctype-njob.pkl','w') as f: f.write(corrfns+'\n'+str(njob))

	jobs = list(range(njob))
	mpi_index_list = fair_division(size,njob) #if parallel else fair_division(1,njob) # last core reserved for the progress bar and saving the outputs?

	starting_index = mpi_index_list[rank] if parallel else 0 #and rank<size-1 else 0
	ending_index = mpi_index_list[rank+1] if parallel else 1 #and rank<size-1 else 1

	if rank>=njob and rank!=size-1: return None # idle cores should not waste their time going through the rest of code and doing some unexpected (possibly interruptive) tasks

	if rank==size-1: # last core in parallel or just the only core in serial
		nthreads = treecorr.config.set_omp_threads(num_threads)
		ncores_omp = ' ['+str(nthreads)+' cpus working together]' if nthreads !=0 else ''
		print("Total CPUs to use: %d\nNumber of MPI cores requested: %d\nNumber of jobs assigned per MPI core: %d" % (size*nthreads,size,njob//size))
		if size==njob+1:
			print("Warning: %d MPI core is idle and has no job assigned to it. Use maximum %d MPI cores.\n" % (size-njob,njob))
		elif size>njob:
			print("Warning: %d MPI cores are idle and have no jobs assigned to them. Use maximum %d MPI cores.\n" % (size-njob,njob))
		else:
			print("An optimum number of MPI cores x OpenMP threads is maximum %d for this configuration." % njob)
		print("Number of OpenMP threads per MPI task: %d" % nthreads )
		print("Total cells (HEALPix pixels): %d" % ncell)
		print('Number of adjacent pairs to find correlation for: %d' % nadjpairs) # analytical 5*NSIDE^2-6*NSiDE+2 gives 645 instead of 651 though
		print("Correlation functions: %s \n\njob #" % corrfns )
		range_rank = range(size) #if parallel else range(size)
		for tempRank in range_rank:
			indi = mpi_index_list[tempRank]
			indf = mpi_index_list[tempRank+1]-1
			if indi==indf:
				arrow = " \t"+str(indf)+"\t-------------> to core "+str(tempRank)+ncores_omp
			elif indi>indf:
				arrow = " \tNA"+"\t-------------> to core "+str(tempRank)+ncores_omp # no job is avalible for this core
			else:
				arrow = str(indi)+"\t"+str(indf)+"\t-------------> to core "+str(tempRank)+ncores_omp
			print( arrow.expandtabs(len(str(mpi_index_list[-1]))+1) ) # some formatting hacks

	##############################################################################
	# using TreeCorr
	##############################################################################

	if not parallel: return

	explained = {'zsnb':['spectroscopic','unblended'], 'zsb':['spectroscopic','blended'], 'zpnb':['photometric','unblended'], 'zpb':['photometric','blended']}
	if rank==size-1: print("\n"+time.strftime('%a %H:%M:%S')+" :: Calculating correlation functions for the %s sample with %s redshifts:\n" % (explained[regime][1],explained[regime][0]))

	loop_ended = False

	while sum( updated_checklist(tdir,njob) ) < njob:

		if loop_ended:
			iranks_left = [il for il, x_ in enumerate( updated_checklist(tdir,njob) ) if not x_] # use compress for huge lists: https://stackoverflow.com/questions/21448225/getting-indices-of-true-values-in-a-boolean-list
			if len(iranks_left)==0: break

		jobs_to_run = [random.choice(iranks_left)] if loop_ended else jobs[starting_index:ending_index]

		for irank in jobs_to_run: # turns into only one iteration in serial!
			if updated_checklist(tdir,njob)[irank]:
				continue
			else:
				open(tdir+str(irank), 'w').close()

			if not parallel: progress = progress_bar(njob,'corrs',per=5, disappear=False, html=False); jobid=0
			pairs_of_cell_ids_used = [pairs_of_cell_ids[irank//njob_per_cxc]] if parallel else pairs_of_cell_ids

			irank_done = [irank][0] # just making sure we have a copy # also save it before reset_mpi_irank

			for c1, c2 in pairs_of_cell_ids_used:

				# reset indices to start from zero for a new chunk/tile/cell in parallel mode
				irank = reset_mpi_irank(irank,njob_per_cxc) if parallel else 0

				input_fname_c1  = input_dir+regime+'_'+str(blend_percent)+'percent/'+regime+'.'+str(c1)+'_'+str(blend_percent)+'percent_r'+str(realization_number)+'.fit' if regime=='zsb' else input_dir+regime+'/'+regime+'.'+str(c1)+'_r'+str(realization_number)+'.fit' 
				input_fname_c2  = input_dir+regime+'_'+str(blend_percent)+'percent/'+regime+'.'+str(c2)+'_'+str(blend_percent)+'percent_r'+str(realization_number)+'.fit' if regime=='zsb' else input_dir+regime+'/'+regime+'.'+str(c2)+'_r'+str(realization_number)+'.fit' 
				random_fname_c1 = input_dir+'rand/rand.'+str(c1)+'.fit'
				random_fname_c2 = input_dir+'rand/rand.'+str(c2)+'.fit'
	  
				input_data_c1   = fio.read(input_fname_c1,columns=['ra','dec','i',regime[:2],'e1','e2','delta_e','shape_weight','badshape']) # read a subset without loading all the data to memory :: rows?
				input_data_c2   = input_data_c1 if (c1==c2) else fio.read(input_fname_c2,columns=['ra','dec','i',regime[:2],'e1','e2','delta_e','shape_weight','badshape'])
				input_data_c1   = input_data_c1[input_data_c1['i']<=limiting_imag]
				input_data_c2   = input_data_c1 if (c1==c2) else input_data_c2[input_data_c2['i']<=limiting_imag]

				ra_c1,ra_c2 = input_data_c1['ra'], input_data_c2['ra']
				dec_c1, dec_c2 = input_data_c1['dec'], input_data_c2['dec']
				shape_weight_c1,shape_weight_c2 = input_data_c1['shape_weight'], input_data_c2['shape_weight']
				delta_e_c1, delta_e_c2 = input_data_c1['delta_e'], input_data_c2['delta_e']
				badshape_c1,badshape_c2 = input_data_c1['badshape'], input_data_c2['badshape']
				imag_c1, imag_c2 = input_data_c1['i'], input_data_c2['i']
				z_c1 = input_data_c1['zs'] if regime[:2]=='zs' else input_data_c1['zp'] # spec-z or photo-z
				z_c2 = input_data_c2['zs'] if regime[:2]=='zs' else input_data_c2['zp'] # spec-z or photo-z

				if (irank in ss_iranks) or (not parallel):
					g1_c1, g2_c1 = input_data_c1['e1'], input_data_c1['e2']
					g1_c2, g2_c2 = [g1_c1, g2_c1] if (c1==c2) else [input_data_c2['e1'], input_data_c2['e2']]

				if (irank in ps_iranks) and parallel: g1_c2, g2_c2 = input_data_c2['e1'], input_data_c2['e2']

				# reading random galaxies
				if irank in pp_iranks or not parallel:
					input_data_R_c1 = fio.read(random_fname_c1)
					input_data_R_c2 = input_data_R_c1 if (c1==c2) else fio.read(random_fname_c2)
					posr_c1 = [ input_data_R_c1['ra'], input_data_R_c1['dec'] ]
					posr_c2 = [ input_data_R_c2['ra'], input_data_R_c2['dec'] ]

				del input_data_c1, input_data_c2
				if (irank in pp_iranks) or (not parallel): del input_data_R_c1, input_data_R_c2

				realization_dir  = output_dir+regime+'_'+str(blend_percent)+'percent/r' if regime=='zsb' else output_dir+regime+'/r'
				realization_dir += str(realization_number)

				# position-position correlation
				if irank in pp_iranks or not parallel:
					pp_irange = [pp_i[irank]] if parallel else range(ntomo)
					pp_jrange = [pp_j[irank]] if parallel else range(ntomo)
					for i in pp_irange:
						for j in pp_jrange: 
							if(j<=i): # is always true for our parallel mode  
								zlimi, zlimj = [zcuts[i], zcuts[i+1]], [zcuts[j], zcuts[j+1]]   
								indi, indj = (z_c1>zlimi[0])&(z_c1<=zlimi[1]), (z_c2>zlimj[0])&(z_c2<=zlimj[1])
								posi, posj = [ ra_c1[indi], dec_c1[indi] ], [ ra_c2[indj], dec_c2[indj] ]
								#wi, wj = w[indi], w[indj]
								DD, RR, DR, RD = pos_pos_corr(posi,posj,posr_c1,posr_c2,same_zshell=(i==j),same_cell=(c1==c2),unique_encounter=(c1>c2),num_threads=num_threads)

								output_fname = realization_dir+'/pp_'+str(i)+'_'+str(j)+'/'+regime+'.r'+str(realization_number)+'.pp_'+str(i)+'_'+str(j)+'.hp_'+str(c1)+'_'+str(c2)+'.pkl' 
								usedir(output_fname.rsplit('/',1)[0]+'/') #if not os.path.exists(output_fname.rsplit('/',1)[0]): os.makedirs(output_fname.rsplit('/',1)[0])
								with open(output_fname, "wb") as f: pickle.dump([DD,RR,DR,RD], f)

								if not parallel:
									progress.refresh(jobid); jobid+=1

				# position-shear correlation
				if irank in ps_iranks or not parallel:
					ps_irange = [ps_i[irank-len(pp_i)]] if parallel else range(ntomo)
					ps_jrange = [ps_j[irank-len(pp_i)]] if parallel else range(ntomo)
					for i in ps_irange: # omitting the first bin for sources
						for j in ps_jrange: # omitting the last bin for lenses
							zlim_lens = [zcuts[j], zcuts[j+1]]   # foreground (lower z for lensing to happen)           
							zlim_source = [zcuts[i], zcuts[i+1]] # background (higher z for lensing to happen)

							if(j<i): 
								ind_lens = (z_c1>zlim_lens[0])&(z_c1<=zlim_lens[1])&(imag_c1<21) 
								ind_source = (z_c2>zlim_source[0])&(z_c2<=zlim_source[1])&(imag_c2>=21)&(badshape_c2!=1)
							else: # the following is useful be repeated for i==j as well, maybe TODO
								ind_lens = (z_c1>zlim_lens[0])&(z_c1<=zlim_lens[1])&(imag_c1>=21) # important swap because i <-> j
								ind_source = (z_c2>zlim_source[0])&(z_c2<=zlim_source[1])&(imag_c2<21)&(badshape_c2!=1)
							pos_lens = [ ra_c1[ind_lens], dec_c1[ind_lens] ]
							pos_source = [ ra_c2[ind_source], dec_c2[ind_source] ]
							w_source = 1./(shape_noise**2+delta_e_c2[ind_source]**2) #shape_weight_c2[ind_source]
							#z_lens, z_source = z[ind_lens], z[ind_source]
							#w_lens, w_source = w[ind_lens], w[ind_source]
							shear_source, k_source = [ g1_c2[ind_source], -g2_c2[ind_source] ], None #k[ind_source] # important negative sign
							ng, varg = pos_shear_corr(pos_lens,pos_source,shear_source,same_cell=(c1==c2),w_source=w_source,num_threads=num_threads) #, k_source=k_source , w_lense=wl,w_source=ws)


							# file names, e.g. 'zsnb.r0.ps_1_3.hp_28_214.pkl'
							output_fname = realization_dir+'/ps_'+str(i)+'_'+str(j)+'/'+regime+'.r'+str(realization_number)+'.ps_'+str(i)+'_'+str(j)+'.hp_'+str(c1)+'_'+str(c2)+'.pkl' 
							usedir(output_fname.rsplit('/',1)[0]+'/') #if not os.path.exists(output_fname.rsplit('/',1)[0]): os.makedirs(output_fname.rsplit('/',1)[0])
							with open(output_fname, "wb") as f: pickle.dump([ng,varg], f)

							if not parallel:
								progress.refresh(jobid); jobid+=1

				# shear-shear correlation 	
				if irank in ss_iranks or not parallel:
					ss_irange = [ss_i[irank-len(pp_i)-len(ps_i)]] if parallel else range(ntomo) # len(pp_i) == len(pp_j)
					ss_jrange = [ss_j[irank-len(pp_i)-len(ps_i)]] if parallel else range(ntomo)
					for i in ss_irange: # omitting the first bin for sources
						for j in ss_jrange:
							if(j<=i): # is always true for our parallel mode 
								zlimi, zlimj = [zcuts[i], zcuts[i+1]], [zcuts[j], zcuts[j+1]]  
								indi, indj = (z_c1>zlimi[0])&(z_c1<=zlimi[1])&(imag_c1>=21)&(badshape_c1!=1), (z_c2>zlimj[0])&(z_c2<=zlimj[1])&(imag_c2>=21)&(badshape_c2!=1)
								#ki, kj = k[indi], k[indj]
								#wi, wj = w[indi], w[indj]
								posi, posj = [ ra_c1[indi], dec_c1[indi] ], [ ra_c2[indj], dec_c2[indj] ]
								sheari, shearj = [ g1_c1[indi], -g2_c1[indi] ], [ g1_c2[indj], -g2_c2[indj] ] # important negative
								wi, wj = 1./(shape_noise**2+delta_e_c1[indi]**2), 1./(shape_noise**2+delta_e_c2[indj]**2)
								gg, varg1, varg2 = shear_shear_corr(posi,posj,sheari,shearj,same_zshell=(i==j),same_cell=(c1==c2),unique_encounter=(c1>c2),w1=wi,w2=wj,num_threads=num_threads) #,k1=ki,k2=kj,w1=wi,w2=wj) 

								output_fname = realization_dir+'/ss_'+str(i)+'_'+str(j)+'/'+regime+'.r'+str(realization_number)+'.ss_'+str(i)+'_'+str(j)+'.hp_'+str(c1)+'_'+str(c2)+'.pkl' 
								usedir(output_fname.rsplit('/',1)[0]+'/') #if not os.path.exists(output_fname.rsplit('/',1)[0]): os.makedirs(output_fname.rsplit('/',1)[0])
								with open(output_fname, "wb") as f: pickle.dump([gg,varg1,varg2], f)

								if not parallel:
									progress.refresh(jobid); jobid+=1

				del ra_c1, ra_c2, dec_c1, dec_c2, z_c1, z_c2, imag_c1, imag_c2
				if (irank in pp_iranks) or (not parallel): del indi, indj, posi, posj, posr_c1, posr_c2, DD,RR,DR,RD
				if (irank in ss_iranks) or (not parallel): del g1_c1, g2_c1, g1_c2, g2_c2, indi, indj, posi, posj, sheari, shearj, gg
				if (irank in ps_iranks) or (not parallel): del g1_c2, g2_c2, ind_lens, ind_source, pos_lens, pos_source, shear_source, k_source, ng

			write_integers_np(fdir+'iranks_done.pkl',irank_done) #  append mode # note: it should be irank_done not irank since we lost track of irank by resetting it
		loop_ended = True

	print("rank",rank,"is done!")

	if parallel:
		COMM.Barrier()
		time.sleep(3)

	if rank==size-1 or not parallel: # rank==size-1 is the case with a single core anyway

		# consistency check

		l1_done = read_integers_np(fdir+'iranks_done.pkl',unique=False); l1_done.sort()
		l2_done = read_integers_np(fdir+'iranks_done.pkl',unique=True); l2_done.sort()

		counter_l1_done = Counter(l1_done)
		dup_times = [y_ for y_ in counter_l1_done.values() if y_!=1]
		ndup = len(dup_times)

		if not np.array_equal(l2_done,jobs): print("\nWarning: missing some correlations in the jobs reported as done in the log file!\n\niranks requested =",jobs,"\niranks calculated =",l2_done,"\n")
		if not np.array_equal(l1_done,l2_done): print("\nWarning:",ndup,"job calculations are reported as done more than one time in the log file! (not serious)\n\niranks calculated =",l2_done,"\n\niranks calculated (repetition allowed)=",l1_done,"\n\nCounter Done =",counter_l1_done,"\n")

		checklist = updated_checklist(tdir,njob)
		if sum(checklist)!=njob: print("\nWarning: missing some correlations in the jobs reported as running!\n\n# of iranks requested =",njob,"\n# of iranks put in the processing pool =",sum(checklist),"\n")

		t1 = datetime.datetime.now()
		print( "\n"+time.strftime('%a %H:%M:%S')+" :: All done! Elapsed time: "+str(datetime.timedelta(seconds=round((t1-t0).seconds))) )





##############################################################################
# functions needed
##############################################################################


def updated_checklist(directory,njob): # np.in64 if you expect a huge number
	checklist = np.zeros((njob,), dtype=bool) #[False for _ in range(njob)]
	taken = list(map(np.int32, os.listdir(directory))) # list is needed to be compatible with py3
	checklist[taken] = True
	return checklist

def getSize(filename):
	st = os.stat(filename)
	return st.st_size

def isNonEmptyFile(filename):
	return os.path.isfile(filename) and getSize(filename) #!=0 # if the first one is filse it does not even check the2nd one to give `No such a file` Error

def write_integers_np(filename,integer):
	f_handle = file(filename, 'ab') # append binary mode # b is important
	np.savetxt(f_handle, [integer], fmt='%d') # remove [] if you already have an array of integers 
	f_handle.close()

def read_integers_np(filename,unique=False):
	intlist = np.loadtxt(filename, dtype='i8').tolist() if isNonEmptyFile(filename) else [] # i4 might be more effivient but just in case we have a very large integer later
	if unique:
		return list(set(intlist)) # unique but unordered
	else:
		return intlist # repeated numbers allowed

def usedir(mydir):
# https://stackoverflow.com/questions/12468022/python-fileexists-error-when-making-directory	
	if not mydir.endswith('/'): mydir += '/' # important
	try:
		if not os.path.exists(os.path.dirname(mydir)):
			os.makedirs(os.path.dirname(mydir)) # sometimes b/w the line above and this line another core already made this directory and it leads to [Errno 17]
			print('-- made dir:',mydir)
	except OSError as err:
		pass # print('[handled error]',err)


def get_pairs(nside,cell_ids,blacklist,nest=True,unique=False):
	
	cell_ids_1 = []
	cell_ids_2 = []
	
	for cid in cell_ids:
		
		# for when we correlate a cell by itself
		cell_ids_1.append(cid)
		cell_ids_2.append(cid)

		# find the 8 nearest cell ids (SW, W, NW, N, NE, E, SE and S) and exclude non-existences (-1) and used ones
		adj_ids = (adj_id for adj_id in hp.get_all_neighbours(nside, cid, nest=nest) if adj_id != -1 and adj_id not in blacklist) 
		
		for adj_id in adj_ids:
	
			# cross cell correlations b/w cid and its neighbours
			cell_ids_1.append(cid)
			cell_ids_2.append(adj_id)
			
			# since it is already checked for neighbours
			if unique: blacklist.append(cid) # allow C1xC2 and C2xC1?

	return zip(cell_ids_1,cell_ids_2), len(cell_ids_1)


def radeclim2cell(ra_lim, dec_lim,num_points=None,pixinfo=None):
	rand_ra, rand_dec = fp.uniform_rect(ra_lim, dec_lim, num_points=num_points)
	cell_ids_touched = radec2pix(rand_ra,rand_dec,pixinfo=pixinfo)
	return list(set(cell_ids_touched)) # delete duplicates; won't preserve the order

def append_vector(big_vector, to_be_appended, flatten=False):
	for h in range(len(to_be_appended)):
		if flatten:
			big_vector.append(to_be_appended[h])
		else:
			big_vector.append([to_be_appended[h]])
	return big_vector

def inpix(ra,dec,pixinfo=None):
	# returns boolean array with 'True' values for the indices of galaxies that lie within the given healpix pixel
	pixel_numbers  = radec2pix(ra,dec,pixinfo=pixinfo) #hp.ang2pix(pixinfo['nside'],ra,dec,nest=pixinfo['nest'],lonlat=True) # longitude (~ra) and latitude (~dec) in degrees
	within_healpix = (pixel_numbers == pixinfo['pixel_number'])
	return within_healpix

def radec2pix(ra,dec,pixinfo=None):
	# pixinfo = {'nside': nside, 'nest': nest}
	pixnum = hp.ang2pix(pixinfo['nside'],ra,dec,nest=pixinfo['nest'],lonlat=True) # longitude (~ra) and latitude (~dec) in degrees
	return pixnum.tolist()

def sum_cumulative(thelist):
	return list(np.cumsum(thelist))
    #return list(itertools.accumulate(thelist)) only works in py3


def fair_division(nbasket,nball):
	''' fair distribution of jobs b/w available cores for an efficient mpi '''
	basket= [0]*nbasket
	while True:
		for i in range(nbasket):
			basket[i]+=1
			if sum(basket)==nball:
				return [0]+list(sum_cumulative(basket)) # [0]+`a list` (not an array!)

def reset_mpi_irank(irank,chunk_size):
	return irank-(irank//chunk_size)*chunk_size 


def pos_pos_corr(pos1,pos2,posr1,posr2,w1=None,w2=None,same_zshell=False,same_cell=False,unique_encounter=False,num_threads=0):

	nbins    = 6
	min_sep  = 0.05 # 3 arcmin
	max_sep  = 3.0 # 180 arcmin 
	bin_size = (max_sep-min_sep)/nbins # roughly
	bin_slop = 0.05/bin_size # 0.1 -> 0.05 # 2pt_pipeline for des used bin_slop: 0.01 here: https://github.com/des-science/2pt_pipeline/blob/master/pipeline/twopt_pipeline.yaml
	# num_threads = 5 #None #0 # should query the number of cpus your computer has
	logger = None

	if same_zshell and same_cell: # auto
		ra, dec = pos1 # either 1 or 2 works, they're the same
		ra_R, dec_R = posr1
		w = w1
		cat   = treecorr.Catalog(ra=ra, dec=dec, ra_units='degrees', dec_units='degrees', w=w) 
		cat_R = treecorr.Catalog(ra=ra_R, dec=dec_R, ra_units='degrees', dec_units='degrees')
	else: # cross
		ra1, dec1 = pos1
		ra2, dec2 = pos2
		ra_R1, dec_R1 = posr1
		ra_R2, dec_R2 = posr2
		cat1   = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='degrees', dec_units='degrees', w=w1) 
		cat2   = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='degrees', dec_units='degrees', w=w2)
		cat1_R = treecorr.Catalog(ra=ra_R1, dec=dec_R1, ra_units='degrees', dec_units='degrees')
		cat2_R = treecorr.Catalog(ra=ra_R2, dec=dec_R2, ra_units='degrees', dec_units='degrees')
	
	DD = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=bin_slop, sep_units='degrees', logger=logger)
	RR = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=bin_slop, sep_units='degrees', logger=logger)
	DR = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=bin_slop, sep_units='degrees', logger=logger)
	RD = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=bin_slop, sep_units='degrees', logger=logger)

	# *** two hp cells when corr is within the same zshell. c1xc2 enough, no c2xc1 later 

	if same_zshell and same_cell: # auto
		# same z, same pix
		DD.process_auto(cat,num_threads=num_threads)
		RR.process_auto(cat_R,num_threads=num_threads)
		DR.process_cross(cat, cat_R,num_threads=num_threads)
		RD = DR.copy() 
	elif same_zshell: # distribute the workload fairly b/w two ranks
		# same z, 2 pix:
		if unique_encounter: # the following two counts shouldn't be doubled up cuz they're the same in both directions
			DD.process_cross(cat1, cat2,num_threads=num_threads)
			RR.process_cross(cat1_R, cat2_R,num_threads=num_threads)
		else:
			DR.process_cross(cat1, cat2_R,num_threads=num_threads)
			DR.process_cross(cat2, cat1_R,num_threads=num_threads)
			RD = DR.copy()
	else: # different z  (can have different/same pix) when 2 cats have diff zshells it is enough to make them different even within the same hp pix
		DD.process_cross(cat1, cat2,num_threads=num_threads) # metric='Rperp')
		RR.process_cross(cat1_R, cat2_R,num_threads=num_threads)
		DR.process_cross(cat1, cat2_R,num_threads=num_threads)
		RD.process_cross(cat1_R, cat2,num_threads=num_threads) 
		# RD != DR here

	return DD, RR, DR, RD



def shear_shear_corr(pos1,pos2,shear1,shear2,k1=None,k2=None,w1=None,w2=None,same_zshell=False,same_cell=False,unique_encounter=False,num_threads=0):

	nbins = 6
	min_sep = 0.05 # 3 arcmin
	max_sep = 3.0 # 180 arcmin
	bin_size = (max_sep-min_sep)/nbins # roughly
	bin_slop = 0.05/bin_size # 0.1 -> 0.05 # 2pt_pipeline for des used bin_slop: 0.01 here: https://github.com/des-science/2pt_pipeline/blob/master/pipeline/twopt_pipeline.yaml
	# num_threads = 5 #None #0
	logger = None

	if same_zshell and same_cell: # auto
		ra, dec = pos1 # either 1 or 2 works
		g1, g2 = shear1
		k = k1
		w = w1
		cat = treecorr.Catalog(g1=g1, g2=g2, k=k, ra=ra, dec=dec, w=w, ra_units='degrees', dec_units='degrees')
	elif same_zshell: # just wanted to distrubute the workload fairly for two encounters (didn't want to make one of the cores idle)
		ra1, dec1 = np.array_split(pos1[0], 2), np.array_split(pos1[1], 2) # split in half
		ra2, dec2 = np.array_split(pos2[0], 2), np.array_split(pos2[1], 2)
		g1_1st, g2_1st = np.array_split(shear1[0], 2), np.array_split(shear1[1], 2)
		g1_2nd, g2_2nd = np.array_split(shear2[0], 2), np.array_split(shear2[1], 2)
		k1 = np.array_split(k1, 2) if (k1 is not None) else [None,None]
		k2 = np.array_split(k2, 2) if (k2 is not None) else [None,None]
		w1 = np.array_split(w1, 2) if (w1 is not None) else [None,None]
		w2 = np.array_split(w2, 2) if (w2 is not None) else [None,None]
		cat1 = [treecorr.Catalog(g1=g1_1st[h], g2=g2_1st[h], k=k1[h], ra=ra1[h], dec=dec1[h], w=w1[h], ra_units='degrees', dec_units='degrees') for h in [0,1]]
		cat2 = [treecorr.Catalog(g1=g1_2nd[h], g2=g2_2nd[h], k=k2[h], ra=ra2[h], dec=dec2[h], w=w2[h], ra_units='degrees', dec_units='degrees') for h in [0,1]]
	else:
		ra1, dec1 = pos1
		ra2, dec2 = pos2
		g1_1st, g2_1st = shear1
		g1_2nd, g2_2nd = shear2
		cat1 = treecorr.Catalog(g1=g1_1st, g2=g2_1st, k=k1, ra=ra1, dec=dec1, w=w1, ra_units='degrees', dec_units='degrees')
		cat2 = treecorr.Catalog(g1=g1_2nd, g2=g2_2nd, k=k2, ra=ra2, dec=dec2, w=w2, ra_units='degrees', dec_units='degrees')

	gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=bin_slop, sep_units='degrees', logger=logger)
	
	if same_zshell and same_cell:
		gg.process_auto(cat,num_threads=num_threads)
	elif same_zshell: # just wanted to distrubute the workload fairly for two encounters (didn't want to make one of the cores idle)
		if unique_encounter: # the following two counts shouldn't be doubled up cuz they're the same in both directions
			gg.process_cross(cat1[0],cat2[0],num_threads=num_threads)
			gg.process_cross(cat1[1],cat2[1],num_threads=num_threads)
		else: # in the other encounter cat1 and cat2 are switched but does not matter anyway
			gg.process_cross(cat2[0],cat1[1],num_threads=num_threads)
			gg.process_cross(cat2[1],cat1[0],num_threads=num_threads)
	else:
		gg.process_cross(cat1,cat2,num_threads=num_threads)

	if same_zshell and same_cell:
		varg1 = treecorr.calculateVarG(cat)
		varg2 = varg1
	elif same_cell:
		varg1 = treecorr.calculateVarG(cat1)
		varg2 = treecorr.calculateVarG(cat2)
	else:
		varg1 = np.nan
		varg2 = np.nan

	return gg, varg1, varg2



def pos_shear_corr(pos_lens,pos_source,shear_source,k_source=None,w_lense=None,w_source=None,same_cell=False,num_threads=0):

	nbins   = 6
	min_sep = 0.05 # 3 arcmin
	max_sep = 3.0 # 180 arcmin 
	bin_size = (max_sep-min_sep)/nbins # roughly
	bin_slop = 0.05/bin_size # 0.1 -> 0.05 # 2pt_pipeline for des used bin_slop: 0.01 here: https://github.com/des-science/2pt_pipeline/blob/master/pipeline/twopt_pipeline.yaml
	logger = None

	ra_lens, dec_lens = pos_lens
	ra_source, dec_source = pos_source
	g1_source, g2_source = shear_source

	# foreground (lens)
	cat_lens = treecorr.Catalog(ra=ra_lens, dec=dec_lens, w=w_lense, ra_units='degrees', dec_units='degrees') 
	
	# background (source)
	cat_source = treecorr.Catalog(ra=ra_source, dec=dec_source, w=w_source, g1=g1_source, g2=g2_source, k=k_source, ra_units='degrees', dec_units='degrees')

	ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=bin_slop, sep_units='degrees', logger=logger)
	ng.process_cross(cat_lens,cat_source,num_threads=num_threads)  # there's no process_auto for this object

	# one shear variance per pixel per source zbin
	varg = treecorr.calculateVarG(cat_source) if same_cell else np.nan 

	return ng, varg


if __name__== "__main__": #sys.argv[1]
	blend_percent=10
	calc_corrs(regime=sys.argv[1],pp=int(sys.argv[2]),ps=int(sys.argv[3]),ss=int(sys.argv[4]),slurm_jobid=str(sys.argv[5]),num_threads=int(sys.argv[6]),realization_number=0, blend_percent=blend_percent) # zsnb, 0 :: 4 regimes, 6 realizations

