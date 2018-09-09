import sys

## adding DESCQA env
sys.path.insert(0, '/global/common/software/lsst/common/miniconda/py3-4.2.12/lib/python3.6/site-packages')


sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/footprinter')
import footprinter as fp

import numpy as np
#from astropy.io import fits  
import fitsio as fio
import time
import datetime
from itertools import accumulate, compress
import healpy as hp

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/Jupytermeter') # loading bar
from jupytermeter import *

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/util')
from util import *

try:
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


# -----------------
# functions first
# -----------------


def sum_cumulative(thelist):
	return list(accumulate(thelist))

# fair distribution of jobs for an efficient mpi
def fair_division(nbasket,nball):
	basket= [0]*nbasket
	while True:
		for i in range(nbasket):
			basket[i]+=1
			if sum(basket)==nball:
				return [0]+sum_cumulative(basket)

def radec2pix(ra,dec,pixinfo=None):
	# pixinfo = {'nside': nside, 'nest': nest}
	pixnum = hp.ang2pix(pixinfo['nside'],ra,dec,nest=pixinfo['nest'],lonlat=True) # longitude (~ra) and latitude (~dec) in degrees
	return pixnum.tolist()

def radeclim2cell(ra_lim, dec_lim,num_points=5000,pixinfo=None):
	rand_ra, rand_dec = fp.uniform_rect(ra_lim, dec_lim, num_points=num_points)
	cell_ids_touched = radec2pix(rand_ra,rand_dec,pixinfo=pixinfo)
	return list(set(cell_ids_touched)) # delete duplicates; won't preserve the order


# -----------------------
# making random catalogs
# -----------------------

t0=time.process_time()

nrandom = 8e6   # num of random galaxies to use in calculating position-position correlation: w(theta)
#regime = 'zsnb' # regime (and realization version) does not matter for the random catalog because we only want the shape of the region
input_dir  = '/global/cscratch1/sd/erfanxyz/projects/blending/buzzard_v1.6_lsst_y1/'
output_dir = '/global/cscratch1/sd/erfanxyz/projects/blending/buzzard_v1.6_lsst_y1/rand/'

nest = True
nside = 8 # for healpix
hpinfo = {'nside': nside, 'nest': nest}
eps = 0.4 # degrees from the edges of the simulation

# edges of the simulation
rai, raf = [0,180] #[10,60] #[0,180]
deci, decf = [0,90] #[10,50] #[0,90]

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

ncell = len(cell_ids)

njob = ncell
jobs = range(njob)
njob_per_cell = 1 

last_core_io = 0 # whether the last core should be exclusively for outputs {1:yes, 0:no}

mpi_index_list = fair_division(size-last_core_io,njob) if parallel else fair_division(1,njob) # size -> size-1 :: last core could be only for the progress bar and saving the outputs

starting_index = mpi_index_list[rank] if parallel and rank<size-last_core_io else 0
ending_index = mpi_index_list[rank+1] if parallel and rank<size-last_core_io else 1

if rank==size-1: # last core in parallel or just the only core in serial
	print("Total mocks (HEALPix pixels): %d" % ncell)
	print("Total cores to use: %d\nAn optimum number of cores is %d for this configuration." % (size,njob+last_core_io)) # one core reserved for output? y/n
	if size==njob+1+last_core_io:
		print("Warning: %d core is idle and has no job assigned to it. Use maximum %d cores." % (size-(njob+last_core_io),njob+last_core_io))
	elif size>njob+last_core_io:
		print("Warning: %d cores are idle and have no jobs assigned to them. Use maximum %d cores." % (size-(njob+last_core_io),njob+last_core_io))
	print("Number of galaxies in each random catalog: %d\n\njob #" % nrandom)
	range_rank = range(size-last_core_io) if parallel else range(size)
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
	if last_core_io==1: print(arrow)
	print("\nSaving random catalogs:\n")

# -------------
# main process
# -------------

for irank in jobs[starting_index:ending_index]: # turns into only one iteration in serial!
	if rank==size-last_core_io and parallel: break # last core should or should not make any random catalog depending on last_core_io

	cell_ids_used = [cell_ids[irank//njob_per_cell]] if parallel else cell_ids
	if not parallel: progress = progress_bar(ncell,'catalogs',per=5, html=False)
	
	for m, mock in enumerate(cell_ids_used):
	
		output_file_name = output_dir+'rand.'+str(mock)+'.fit'
		pixinfo = {'nside': nside, 'pixel_number': mock, 'nest': nest}
		posr = fp.uniform_poly_fast(pixinfo=pixinfo, num_points=nrandom)

		gen_write(output_file_name, ['ra','dec'],[posr[0],posr[1]])
		print("%d random positions stored in %s" % (posr[2], output_file_name) )

		if not parallel: progress.refresh(m)

# when we use MPI_barrier() for synchronization, we are guaranteed that no process 
# will leave the barrier until all processes have entered it
COMM.Barrier()
time.sleep(2)

#END: This need to print after all MPI_Send/MPI_Recv has been completed
if rank==size-1: print( '\nAll done! Elapsed time: '+str(datetime.timedelta(seconds=round(time.process_time()-t0))) ) # it underestimates the elapsed time?!



