#!/bin/bash -l
#SBATCH --constraint=haswell
#SBATCH -p debug
#SBATCH -N 64                              # Maximum 32 processors per node in cori
#SBATCH -t 00:30:00
#SBATCH -A m1727
#SBATCH -J get_rands
#SBATCH -o get_rands.out
#SBATCH -e get_rands.err
#SBATCH --mail-type=END                    # Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=erfan@ucdavis.edu      # Destination email address
#SBATCH --mem=120GB                        # Memory (avoid OOM error) --> bigmem queue

module load python/3.6-anaconda-4.4
module load openmpi
module load PrgEnv-intel

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# use --unbuffered to update output files without delay
srun -n 143 --unbuffered --cpus-per-task 4 --cpu_bind=cores python get_rands.py
