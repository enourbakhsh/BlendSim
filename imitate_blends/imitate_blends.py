import sys
#7 min with networkx

## adding a rich env
sys.path.insert(0, '/global/common/software/lsst/common/miniconda/py3-4.2.12/lib/python3.6/site-packages')
# /global/u1/e/erfanxyz/.conda/envs/erfanxyz-1/lib/python2.7/site-packages

# sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/footprinter')
# import footprinter as fp

import numpy as np
import fitsio as fio
import time
import datetime
from itertools import accumulate, chain
from astropy.coordinates import SkyCoord, Angle, search_around_sky
from numpy.lib.recfunctions import append_fields
from astropy.io import fits

import networkx 
from networkx.algorithms.components.connected import connected_components

# import multiprocessing
# from multiprocessing import Pool

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/Jupytermeter') # loading bar
from jupytermeter import *

sys.path.insert(0, '/global/homes/e/erfanxyz/myprojects/packages/util')
from util import *

try:
	# import nonsense
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


# ------------
# functions
# ------------


def sum_cumulative(thelist):
	#return list(np.cumsum(thelist)) # only option for py2
	return list(accumulate(thelist)) # faster but only works in py3


# fair distribution of jobs for an efficient mpi
def fair_division(nbasket,nball):
	basket= [0]*nbasket
	while True:
		for i in range(nbasket):
			basket[i]+=1
			if sum(basket)==nball:
				return [0]+sum_cumulative(basket)

# https://stackoverflow.com/questions/39965994/memory-friendly-way-to-add-a-field-to-a-structured-ndarray-without-duplicating
def add_field(data,newkey,values):
	if newkey in data.dtype.names:
		data[newkey] = values
	else:
		data = append_fields(data, newkey, values).filled()
	return data


def to_float(x):
	return [np.float(val) for val in x]


# source: https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
def gather_blends(theList): #slow

	# input MUST be a list of lists
	# L = [[0, 1, 3], [0, 1, 5, 7], [2], [0, 3], [4,8], [1, 9],[6,22],[1, 33],[4,99], [22,6], [77]]
	# gather_blends(L) will return:
	# [{2}, {0, 1, 3, 5, 7, 9, 33}, {4, 8, 99}, {77}, {6, 22}]

	#from itertools import chain

	lst = theList.copy() # it is important to copy()
	L = set(chain.from_iterable(lst)) 

	ff=True
	for each in L:
		components = [x for x in lst if each in x]
		for i in components:
			lst.remove(i)
		lst += [set(chain.from_iterable(components))]
	return lst


# source: https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
def gather_blends_ordered(theList): # an alternative function for gather_blends, <still slow>

	# input MUST be a list of lists
	# L = [[0, 1, 3], [0, 1, 5, 7], [2], [0, 3], [4,8], [1, 9],[6,22],[1, 33],[4,99], [22,6], [77]]
	# gather_blends(L) will return:
	# [{0, 1, 3, 5, 7, 9, 33}, {2}, {4, 8, 99}, {6, 22}, {77}]

	lst_ = theList.copy() # it is important to copy()
	out = []
	while len(lst_)>0:
		# `first, *rest = lst` construct is Python 3 only, swapping it with
		# `first, rest = lst[0], lst[1:]` seems to work fine on python 2.7 
		first, rest = lst_[0], lst_[1:] #first, *rest = l 
		first = set(first)
		lf = -1
		while len(first)>lf:
			lf = len(first)
			rest2 = []
			for r in rest:
				if len(first.intersection(set(r)))>0:
					first |= set(r)
				else:
					rest2.append(r)     
			rest = rest2
		out.append(first)
		lst_ = rest

	return out

# https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G
def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)
    for current in it:
        yield last, current
        last = current    
def gather_blends_networkx(theList): # an alternative function for gather_blends (super fast)
	G = to_graph(theList)
	return connected_components(G)


# -----------------------
# making blended catalogs
# -----------------------

t0 = datetime.datetime.now() #time.process_time()

maxsep = 1.0 #2.18 for 20% #1.5 for 10% #3.2# maximum separation for blends in arcsec
zregime = 'zs'
blend_percent = 5 # blend fraction in percent
realization_number = 0
limiting_mag_i_lsst = 23.0 # applied only on the isolated galaxies 

input_dir  = '/global/cscratch1/sd/erfanxyz/projects/blending/buzzard_v1.6_lsst_y1/'
output_dir = '/global/cscratch1/sd/erfanxyz/projects/blending/buzzard_v1.6_lsst_y1/'

# cell numbers in NEST ordering - all ids in a quarter of sky excluding the edge cells
cell_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 120, 121, 122, 123, 124, 126, 285, 286, 287, 308, 309, 311, 349, 350, 351, 365, 366, 367, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 429, 430, 431, 440, 442, 443]

ncell = len(cell_ids)
nest = True
nside = 8

njob = ncell
jobs = range(njob)
njob_per_cell = 1 

last_core_io = 0 # whether the last core should be exclusively for outputs {1:yes, 0:no}

mpi_index_list = fair_division(size-last_core_io,njob) if parallel else fair_division(1,njob) # size -> size-1 :: last core could be only for the progress bar and saving the outputs

starting_index = mpi_index_list[rank] if parallel and rank<size-last_core_io else 0
ending_index = mpi_index_list[rank+1] if parallel and rank<size-last_core_io else 1

if rank==size-1: # last core in parallel or just the only core in serial
	print("Total cells (HEALPix pixels): %d" % ncell)
	print("Total cores to use: %d\nAn optimum number of cores is %d for this configuration." % (size,njob+last_core_io)) # one core reserved for output? y/n
	if size==njob+1+last_core_io:
		print("Warning: %d core is idle and has no job assigned to it. Use maximum %d cores." % (size-(njob+last_core_io),njob+last_core_io))
	elif size>njob+last_core_io:
		print("Warning: %d cores are idle and have no jobs assigned to them. Use maximum %d cores." % (size-(njob+last_core_io),njob+last_core_io))
	print("Separation angle threshold for blends: %s arcsec\n\njob #" % maxsep)
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
	#print("\nImitating blends:\n")

	explained = {'zs':'spectroscopic', 'zp':'photometric'}
	if rank==size-1: print("\n"+time.strftime('%a %H:%M:%S')+" :: Imitating blends for the sample with %s redshifts:\n" % explained[zregime])

# -------------
# main process
# -------------

for irank in jobs[starting_index:ending_index]: # turns into only one iteration in serial!
	if rank==size-last_core_io and parallel: break # last core should or should not make any random catalog depending on last_core_io

	cell_ids_used = [cell_ids[irank//njob_per_cell]] if parallel else cell_ids
	if not parallel: progress = progress_bar(ncell,'catalogs',per=5, html=False)
	
	for m, cell in enumerate(cell_ids_used):
	
		input_file_name = input_dir+zregime+'nb/'+zregime+'nb.'+str(cell)+'_r'+str(realization_number)+'.fit' 
		output_file_name = output_dir+zregime+'nb/'+zregime+'nb.'+str(cell)+'.grouped_'+str(blend_percent)+'percent_r'+str(realization_number)+'.fit'

		# fits = fio.FITS(input_file_name,'rw') # slow method where output will just add columns to the same input
		mydata = fio.read(input_file_name) #or fits[1].read() 
		ra, dec = to_float(mydata['ra']), to_float(mydata['dec']) # since astropy's search gives error for the scientific notation we convert everything to a regular float
		del mydata # release some memory

		if rank==size-1: print("Read cells.") # #%d\n"%cell)

		# bad_radec = (np.abs(ra)>360) | (np.abs(dec)>90)
		# for b in ---: print('WARNING: invalid (ra,dec)=',ra[b],dec[b],'seen in cell #',cell)

		catalog = SkyCoord(ra, dec, unit='deg')
		centers = catalog # internal match
		idx_centers, idx_catalog, d2d, d3d = catalog.search_around_sky(centers, Angle(str(maxsep)+'s')) #,storekdtree='kdtree_sky') # d3d unused for now

		if rank==size-1: print("Detected the blend candidates.\n")

		print('idx_centers.shape',np.array(idx_centers).shape)


		t00 = datetime.datetime.now()

		freq = np.bincount(idx_centers)
		cumsum = [0]+sum_cumulative(freq)

		t11 = datetime.datetime.now()
		print( "\n"+time.strftime('%a %H:%M:%S')+" :: bincount Elapsed time: "+str(datetime.timedelta(seconds=round((t11-t00).seconds))) )

		seq = [list(idx_catalog[cumsum[i]:cumsum[i+1]]) for i in range(len(cumsum)-1) if len(idx_catalog[cumsum[i]:cumsum[i+1]])>1] # list is important

		t22 = datetime.datetime.now()
		print( "\n"+time.strftime('%a %H:%M:%S')+" :: seq Elapsed time: "+str(datetime.timedelta(seconds=round((t22-t11).seconds))) )


		print('found seq',seq[0:5])
		seq = gather_blends_networkx(seq)

		t33 = datetime.datetime.now()
		print( "\n"+time.strftime('%a %H:%M:%S')+" :: gather_blends_networkx Elapsed time: "+str(datetime.timedelta(seconds=round((t33-t22).seconds))) )

		print('gathered blends for cell',cell)
		gnarray = np.repeat(0,len(ra))
		gsarray = np.repeat(1,len(ra))

		GroupID, GroupSize = [], []
		gn=1 # zero reserved for single galaxies
		for eachset in seq: 
			#if (len(eachset)<2): continue # not a blend
			for galaxy_idx in eachset: 
				gnarray[galaxy_idx]=gn
				gsarray[galaxy_idx]=len(eachset)
			GroupID.append(gn)	
			GroupSize.append(len(eachset))
			gn+=1

		print('made gnarray and gsarray for cell',cell)

# NOTE: fitsio alone had memory issue adding two new columns. I had to combine it with astropy's fits

		mydata = fits.open(input_file_name)[1].data
		magcut_only_on_singles = (gnarray!=0) | ( (gnarray==0) & (mydata['i']<=limiting_mag_i_lsst) )
		# let's cut the unnecessary rows from the isolated galaxies; blended one need to be done after finalization in blendit.py 
		print("len(mydata['i']), sum(magcut_only_on_singles)",len(mydata['i']),sum(magcut_only_on_singles))

		orig_cols = mydata.columns
		del mydata
		new_cols  = fits.ColDefs([fits.Column(name='GroupID',   format='K', array=gnarray), # sys.maxsize = 9223372036854775807 # https://stackoverflow.com/questions/7604966/maximum-and-minimum-values-for-ints
		                          fits.Column(name='GroupSize', format='K', array=gsarray)])

		hdu = fits.BinTableHDU.from_columns(orig_cols + new_cols, name='catalog')
		hdu.data = hdu.data[magcut_only_on_singles] # had to do it like this at the end
		hdu.writeto(output_file_name, overwrite=True)

		del hdu
		t44 = datetime.datetime.now()
		print( "\n"+time.strftime('%a %H:%M:%S')+" :: adding new cols Elapsed time: "+str(datetime.timedelta(seconds=round((t44-t33).seconds))) )

		nblendsys=len(GroupID)
		GroupID = list(np.repeat(0,len(gnarray[gnarray==0])))+list(GroupID)
		GroupSize = list(np.repeat(1,len(gsarray[gsarray==1])))+list(GroupSize)
		del gnarray, gsarray
		gen_write(output_file_name,['GroupID','GroupSize'],[np.array(GroupID),np.array(GroupSize)],dtypes=['i4','i4'], extname='blends_stat', clobber=False) # just append it
		del GroupID, GroupSize	

		# # very slow method, do not do this:
		# fits[-1].insert_column(name='GroupID', data=gnarray)  # add the extra column
		# fits[-1].insert_column(name='GroupSize', data=gsarray)  # add another column
		# # create a new table extension and write unique pairs of ids and sizes
		# fits.write([GroupID,GroupSize], names=['GroupID','GroupSize'])
		# #fitsio.write(file_name, data, extname=extname, clobber=clobber) # this and gen_write are really faster but I didn't want to read unnecessary data for now
		# fits.close()

		print("%d blended systems detected for cell %d and grouped in %s" % (nblendsys, cell, output_file_name) )
		if not parallel: progress.refresh(m)

# when we use MPI_barrier() for synchronization, we are guaranteed that no process 
# will leave the barrier until all processes have entered it
if parallel:
	COMM.Barrier()
	time.sleep(5)

#END: This needs to print after all MPI_Send/MPI_Recv has been completed - automatic pass in serial mode
if rank==size-1:
	t1 = datetime.datetime.now()
	print( "\n"+time.strftime('%a %H:%M:%S')+" :: All done! Elapsed time: "+str(datetime.timedelta(seconds=round((t1-t0).seconds))) )



