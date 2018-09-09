from __future__ import print_function

# TODO: I could assign a core to each magnitude band in each cell, meaning 6 cores per cell

## imports
import sys

## adding DESCQA env
sys.path.insert(0, '/global/common/software/lsst/common/miniconda/py3-4.2.12/lib/python3.6/site-packages')

## Note: if you use Python 2, comment the line above and uncomment the line below
# sys.path.insert(0, '/global/common/cori/contrib/lsst/apps/anaconda/py2-envs/DESCQA/lib/python2.7/site-packages')

#import subprocess
import numpy as np
import scipy
import datetime
import time
import healpy as hp

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

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/gcr-catalogs')
import GCRCatalogs 

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/util')
import util

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/PhotoZDC1/src') # LSST error model
import photErrorModel as ephot

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/Jupytermeter') # loading bar
from jupytermeter import *

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/footprinter')
import footprinter as fp

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/epsnoise') # my version of epsnoise (I edited and added some functions)
import epsnoise 


# -------------------
# main program
# -------------------

def get_cells(realization_number=0,celli=None,cellf=None): # realization_number: 0, 1, 2 ,3 ,4, 5

	#print('-->>> entered core %d' % rank)
	realization_number = int(realization_number)
	celli = None if celli is None else int(celli)
	cellf = None if cellf is None else int(cellf)

	#realization_number = 0 # 0, 1, 2 ,3 ,4, 5

	## adjustments
	nyear = 1
	limiting_mag_i_lsst = 25.0 #24.0 # I went 2 mags fainter because Blendsim might displace some very faint galaxies and combined with other galaxies they can have imag < final_cut or so
	nNan_max = 2

	if rank==0: t0=time.process_time()

	nest = True
	nside = 8 # for healpix
	hpinfo = {'nside': nside, 'nest': nest}
	eps = 0.4 # degrees from the edges of the simulation

	# edges of the simulation
	rai, raf = [0,180] 
	deci, decf = [0,90] 

	# generate randoms such that at least one random position falls into each healpix cell for the iven resolution nside = 8 -> so 1000 is fine
	cell_ids_in_a_quarter = radeclim2cell([rai,raf], [deci,decf],num_points=6000,pixinfo=hpinfo) # Buzzard galaxies are all in a quarter of sky

	# find the edge cells to remove them later
	cell_ids_in_stripe = radeclim2cell([rai,rai+eps], [deci,decf],num_points=6000,pixinfo=hpinfo) 
	cell_ids_in_stripe.extend( radeclim2cell([rai,raf], [deci,deci+eps],num_points=6000,pixinfo=hpinfo) )
	cell_ids_in_stripe.extend( radeclim2cell([raf-eps,raf], [deci,decf],num_points=6000,pixinfo=hpinfo) )
	cell_ids_in_stripe.extend( radeclim2cell([rai,raf], [decf-eps,decf],num_points=6000,pixinfo=hpinfo) )

	blacklist = list(set(cell_ids_in_stripe)) # avoid cells that are close to the edges

	cell_ids = [aa for aa in cell_ids_in_a_quarter if aa not in blacklist]
	cell_ids.sort() # to have identical sorted lists for all cores

	if celli is not None and cellf is not None:
		cell_ids = cell_ids[celli:cellf+1] # since it should include celli and cellf as well

	## load 'buzzard' catalog
	realization = ['buzzard_v1.6', 'buzzard_v1.6_1', 'buzzard_v1.6_2', 'buzzard_v1.6_3', 'buzzard_v1.6_5', 'buzzard_v1.6_21']
	gc = GCRCatalogs.load_catalog(realization[realization_number])
	gc.healpix_pixels = cell_ids # !NEST ordering!
	gc.check_healpix_pixels()
	ncell = len(gc.healpix_pixels)

	## some hacks to get lsst magnitudes in older versions of GCRCatalogs:
	_abs_mask_func = lambda x: np.where(x==99.0, np.nan, x + 5 * np.log10(gc.cosmology.h))
	_mask_func = lambda x: np.where(x==99.0, np.nan, x)
	for i, b in enumerate('ugrizY'):
		gc._quantity_modifiers['Mag_true_{}_lsst_z0'.format(b)] = (_abs_mask_func, 'lsst/AMAG/{}'.format(i))
		gc._quantity_modifiers['mag_true_{}_lsst'.format(b)] = (_mask_func, 'lsst/TMAG/{}'.format(i)) #-----> TMAG to LMAG !!!!

	# if ncell % size != 0 the last processor will
	# do more than njob_per_core jobs 
	njob = ncell # TODO: I should make more jobs by assigning one band per core

	if rank<njob:

		mpi_index_list = fair_division(size,njob) # if parallel else fair_division(1,njob) # last core reserved for the progress bar and saving the outputs
		starting_index = mpi_index_list[rank] if parallel else 0
		ending_index = mpi_index_list[rank+1] if parallel else None #len(cell_ids) # not -1 since it ignores the last one

		if rank==0: # last core in parallel or just the only core in serial
			if size==njob+1:
				print("Warning: %d core is idle and has no job assigned to it. Use maximum %d cores.\n" % (size-njob,njob))
			elif size>njob:
				print("Warning: %d cores are idle and have no jobs assigned to them. Use maximum %d cores.\n" % (size-njob,njob))
			print("Total cells (HEALPix pixels): %d" % njob)
			print("Total cores to use: %d\nAn optimum number of cores is %d for this configuration.\n\njob #" % (size,njob)) # one core reserved for output
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
				print( arrow.expandtabs(len(str(mpi_index_list[-1]))+1) ) # some formatting hacks
			arrow = "prints"+"\t-------------> to core 0"	
			print(arrow)



		if rank==0: print("\nGetting lsst y%d observed magnitudes and errors and healing non-detections:" % nyear)

		# set up LSST error model from PhotozDC1 codebase
		errmodel = ephot.LSSTErrorModel()

		# manually customize parameters
		errmodel.nYrObs = nyear
		errmodel.sigmaSys = 0.005
		# https://docushare.lsstcorp.org/docushare/dsweb/Get/LPM-17
		# Nv1 (design spec.) { 56 (2.2) 80 (2.4) 184 (2.8) 184 (2.8) 160 (2.8) 160 (2.8) } / 10 = per year
		errmodel.nVisYr = {'LSST_u':5.6,'LSST_g':8,'LSST_r':18.4,'LSST_i':18.4,'LSST_z':16,'LSST_y':16}
		# Non-numpy functions like math.sqrt() don't play nicely with numpy arrays
		# So, I had to vectorize the function using numpy to speed it up
		vec_getObs = np.vectorize(errmodel.getObs)

		def funcsn1(mag,band):
			# gives magnitude for the magnitude error where S/N = 1
			# dMag = 2.5 log ( 1 + N/S ) = 0.7526
			_ , magerr = vec_getObs(mag,band)-2.5*np.log10(2)
			return magerr

		u_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_u'), maxiter=1000)
		g_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_g'), maxiter=1000)
		r_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_r'), maxiter=1000)
		i_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_i'), maxiter=1000)
		z_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_z'), maxiter=1000)
		y_SN1 = scipy.optimize.brentq(funcsn1, 20.0, 32.0, args=('LSST_y'), maxiter=1000)

		results_dir = '/global/cscratch1/sd/erfanxyz/projects/blending/buzzard_v1.6_lsst_y1/zsnb/'

		for cell_needed in gc.healpix_pixels[starting_index:ending_index]:

			output_fname = results_dir+'zsnb.'+str(cell_needed)+'_r'+str(realization_number)+'.fit' 

			# since Buzzard is split in healpix pixels before lensing is applied, if we need lensed galaxies that are in a specific healpix pixel, we have to load all neighboring pixels
			mag_true_u_lsst = list(gc.get_quantities(['mag_true_u_lsst'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0]
			mag_true_g_lsst = list(gc.get_quantities(['mag_true_g_lsst'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0]
			mag_true_r_lsst = list(gc.get_quantities(['mag_true_r_lsst'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0]
			mag_true_i_lsst = list(gc.get_quantities(['mag_true_i_lsst'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0]
			mag_true_z_lsst = list(gc.get_quantities(['mag_true_z_lsst'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0]
			mag_true_y_lsst = list(gc.get_quantities(['mag_true_Y_lsst'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0]


			# this will eventually replace -inf values with 99.0 as a flag for non-detections (i.e. zero or negative flux)
			u,g,r,i,z,y, eu,eg,er,ei,ez,ey = [np.repeat(99.0,len(mag_true_u_lsst)) for _ in range(6*2)]
			# they are all the same size

			u_finite = (mag_true_u_lsst!=-np.inf)
			g_finite = (mag_true_g_lsst!=-np.inf)
			r_finite = (mag_true_r_lsst!=-np.inf)
			i_finite = (mag_true_i_lsst!=-np.inf)
			z_finite = (mag_true_z_lsst!=-np.inf)
			y_finite = (mag_true_y_lsst!=-np.inf)

			finite_index = [u_finite,g_finite,r_finite,i_finite,z_finite,y_finite]	  

			u[u_finite],eu[u_finite] = vec_getObs(mag_true_u_lsst[u_finite],'LSST_u')

			g[g_finite],eg[g_finite] = vec_getObs(mag_true_g_lsst[g_finite],'LSST_g')
			r[r_finite],er[r_finite] = vec_getObs(mag_true_r_lsst[r_finite],'LSST_r')
			i[i_finite],ei[i_finite] = vec_getObs(mag_true_i_lsst[i_finite],'LSST_i')
			z[z_finite],ez[z_finite] = vec_getObs(mag_true_z_lsst[z_finite],'LSST_z')
			y[y_finite],ey[y_finite] = vec_getObs(mag_true_y_lsst[y_finite],'LSST_y')

			# zero/negative observed fluxes are flagged led to the ugrizy magnitudes ~>99
			mags_obs_lsst = np.array([u,g,r,i,z,y])
			NaN_index = (mags_obs_lsst>=98.9)
			nNaN = np.sum(NaN_index, axis=0) # number of NaN bands per object e.g. [0 0 0 ..., 0 2 1]

			u[u>=98.9] = u_SN1
			g[g>=98.9] = g_SN1
			r[r>=98.9] = r_SN1
			i[i>=98.9] = i_SN1
			z[z>=98.9] = z_SN1
			y[y>=98.9] = y_SN1

			eu[eu>=98.9] = 2.5*np.log10(2)
			eg[eg>=98.9] = 2.5*np.log10(2)
			er[er>=98.9] = 2.5*np.log10(2)
			ei[ei>=98.9] = 2.5*np.log10(2)
			ez[ez>=98.9] = 2.5*np.log10(2)
			ey[ey>=98.9] = 2.5*np.log10(2)

			# galaxies with maximum nNan_max bands with 99 values are accepted
			# limiting magnitude is also applied at the same time
			accepted = (nNaN <= nNan_max) & (i < limiting_mag_i_lsst)
			del mags_obs_lsst

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

			galaxy_id = (list(gc.get_quantities(['galaxy_id'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]
			zs = (list(gc.get_quantities(['redshift_true'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]
			ra = (list(gc.get_quantities(['ra'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]
			dec = (list(gc.get_quantities(['dec'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]
			galaxy_size = (list(gc.get_quantities(['size'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]
			gamma1 = (list(gc.get_quantities(['shear_1'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]
			gamma2 = (list(gc.get_quantities(['shear_2'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]
			kappa = (list(gc.get_quantities(['convergence'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]
			# e1 = (list(gc.get_quantities(['ellipticity_1'], # ! WILL ASSIGN IT MYSELF !
			#                        filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			#                        native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]
			# e2 = (list(gc.get_quantities(['ellipticity_2'], # ! WILL ASSIGN IT MYSELF !
			#                        filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			#                        native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]
			ug_rest = (list(gc.get_quantities(['Mag_true_u_lsst_z0'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted] \
					 -(list(gc.get_quantities(['Mag_true_g_lsst_z0'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted] 
					 # this is u-g rest frame color @ z=0 

			# - grab z=0.1 rest frame absolute magnitudes only to split the sample into red and blue
			Mag_true_g_des_z01 = (list(gc.get_quantities(['Mag_true_g_des_z01'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]
			Mag_true_r_des_z01 = (list(gc.get_quantities(['Mag_true_r_des_z01'],
			                       filters=[(lambda a, d: hp.ang2pix(nside, a, d, nest=nest, lonlat=True) == cell_needed, 'ra', 'dec')],
			                       native_filters=[(lambda p: p in [cell_needed]+hp.get_all_neighbours(nside, cell_needed, nest=nest).tolist(), 'healpix_pixel')]).values())[0])[accepted]

			Color_true_gr_des_z01 = Mag_true_g_des_z01 - Mag_true_r_des_z01
			g1, g2 = gamma1/(1-kappa), gamma2/(1-kappa) # assuming shear_{1,2} is not the reduced shear!
			g_complex = g1+1j*g2

			# - - - Here I use COSMOS ellipticity distribution to assign realistic ellipticities myself

			# mask = np.isfinite(Color_true_gr_des_z01)
			# numrows = sum(accepted)
			red = (Color_true_gr_des_z01 > 0.185 - 0.028 * Mag_true_r_des_z01)
			numred = sum(red)
			numblue = sum(~red)
			numrb = [numred,numblue] 
			eps_noisy = np.zeros(numred+numblue, dtype=complex)
			eps_sheared = np.zeros(numred+numblue, dtype=complex)
			shape_noise = np.zeros(numred+numblue, dtype=float)

			# Johnson SB model parameters for red (index=0) and blue (index=1) population:
			JSB_a = [1.91567133, -0.13654868] 
			JSB_b = [1.10292468, 0.94995499 ]
			JSB_loc = [0.00399965, -0.02860337]

			for rb, gtype in enumerate([red,~red]):
				eps_intrinsic = epsnoise.sampleEllipticity(numrb[rb], model='JohnsonSB', a=JSB_a[rb], b=JSB_b[rb], loc=JSB_loc[rb]) # I added JohnsonSB model to epsnoise
				ok = np.isfinite(g_complex) & np.isfinite(g_complex)
				print('rb',rb,'bad g_complex values :',sum(~ok))
				print('rb',rb,'min max g_complex :', np.min(g_complex),np.max(g_complex))
				eps_sheared[gtype] = epsnoise.addShear(eps_intrinsic, g_complex[gtype])
				nu = magerr2snr(ei[gtype]) # nu: significance of image i.e. S/N ratio based on imag_err
				try:
					eps_noisy[gtype] = epsnoise.addNoise(eps_sheared[gtype], nu, True) # I fixed this function in epsnoise to get rid of an excess number of galaxies arounf e~1 but Peter said you better remove them
				except Exception as exc:
					print('Failure in rank',rank,'cell',cell_needed,'rb',rb)
					raise exc

			e1_lensed_only = np.real(eps_sheared)
			e2_lensed_only = np.imag(eps_sheared)

			good = np.abs(eps_noisy) <= 0.999999999999 # not actually needed, just to print
			bad = np.abs(eps_noisy) > 0.999999999999
			print('rank %i: %f percent of shapes is useless' % (rank,100.*sum(bad)/len(eps_noisy)) )

			e1 = np.real(eps_noisy)
			e2 = np.imag(eps_noisy)

			shape_noise_ = {'red':np.std(eps_noisy[(red)&(~bad)]), 'blue':np.std(eps_noisy[(~red)&(~bad)])}
			print('sn red, blue',shape_noise_['red'],shape_noise_['blue'])

			shape_noise[red] = shape_noise_['red'] 
			shape_noise[~red] = shape_noise_['blue'] 

			print('g_complex.shape', g_complex.shape)
			print('num good',sum(good),'num bad',sum(bad),'N-bad',numred+numblue-sum(bad),'N-good',numred+numblue-sum(good))
			print('eps_sheared.shape', eps_sheared.shape)
			print('eps_noisy.shape', eps_noisy.shape)
			de1 = np.abs( np.real(eps_sheared) - e1 )
			de2 = np.abs( np.imag(eps_sheared) - e2 )
			de = np.abs( np.abs(eps_sheared) - np.abs(eps_noisy) )

			shape_weight = 1. / (shape_noise**2 + de**2)

			del eps_intrinsic, eps_sheared, eps_noisy, nu, g_complex
			red = red.astype(int) # {1: red, 0: blue}
			bad = bad.astype(int) # {1: useless shape, 0: useful shape}

			# - - - end reassigning ellipticities

			ug,gr,ri,iz,zy = u-g, g-r, r-i, i-z, z-y
			eug = np.sqrt(eu**2+eg**2)
			egr = np.sqrt(eg**2+er**2)
			eri = np.sqrt(er**2+ei**2)
			eiz = np.sqrt(ei**2+ez**2)
			ezy = np.sqrt(ez**2+ey**2)

			util.gen_write(output_fname,
					['id','u','g','r','i','z','y','ug_rest','u-g','g-r','r-i','i-z','z-y',
					 'eu','eg','er','ei','ez','ey','eu-g','eg-r','er-i','ei-z','ez-y',
					 'zs','ra','dec','size','gamma1','gamma2','kappa','g1','g2','e1','e2',
					 'delta_e','delta_e1','delta_e2','e1_lensed_only','e2_lensed_only','shape_noise','shape_weight','isred','badshape'], 
					[galaxy_id, u,g,r,i,z,y, ug_rest,ug,gr,ri,iz,zy, eu,eg,er,ei,ez,ey, 
					 eug,egr,eri,eiz,ezy, zs, ra, dec, galaxy_size, gamma1, gamma2, kappa,
					 g1, g2, e1, e2, de, de1, de2, e1_lensed_only, e2_lensed_only, shape_noise, shape_weight, red, bad], 
					 dtypes=['a11','','','','','','','','','','','','','','','',
							'','','','','','','','','','','','','','','','','','','','','','','','','','','i1','i1'])

			print('Core '+str(rank)+' is done writing the output catalog.')

	if rank==0: print('Waiting for all processors to write their outputs...')

	# when we use MPI_barrier() for synchronization, we are guaranteed that no process 
	# will leave the barrier until all processes have entered it
	if parallel: COMM.Barrier()
	time.sleep(2)

	#END: This need to print after all MPI_Send/MPI_Recv has been completed
	if rank==0: print( '\nAll done! Elapsed time: '+str(datetime.timedelta(seconds=round(time.process_time()-t0))) ) # time is underestimated?


# -------------------
# functions
# -------------------

def magerr2snr(magerr):
	# dm = 2.5 log ( 1 + N/S )
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
	return list(set(cell_ids_touched)) # delete duplicates; won't preserve the order

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

if __name__== "__main__":
	get_cells(realization_number=0) # realization_number: 0, 1, 2 ,3 ,4, 5

