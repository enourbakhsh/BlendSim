#!/bin/bash -l

#SBATCH -A m1727
#SBATCH --mail-type=END                  # Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=erfan@ucdavis.edu    # Destination email address

start_time="$(date -u +%s)"
echo "SLURM_JOB_ID" $SLURM_JOB_ID

rm -f tmp/${SLURM_JOB_ID}_running/ctype-njob.pkl         # delete the file that stores the correlation type(s)
rm -f tmp/${SLURM_JOB_ID}_running/iranks_taken.pkl  # delete the file that keeps track of running jobs
rm -f tmp/${SLURM_JOB_ID}_running/iranks_done.pkl   # delete the file that keeps track of completed jobs

module load python #/2.7-anaconda-4.4

if $( echo ${LOADEDMODULES} | grep --quiet 'PrgEnv-intel' ); then
	echo $'Had to do: module swap PrgEnv-intel PrgEnv-gnu\n';
	module swap PrgEnv-intel PrgEnv-gnu
elif $( echo ${LOADEDMODULES} | grep --quiet 'PrgEnv-cray' ); then
	echo $'Had to do: module swap PrgEnv-cray PrgEnv-gnu\n';
	module swap PrgEnv-cray PrgEnv-gnu
elif $( echo ${LOADEDMODULES} | grep --quiet 'PrgEnv-gnu' ); then
	echo $'PrgEnv-gnu is already loaded\n';
else
	echo $'Had to do: module load PrgEnv-gnu\n';
	module load PrgEnv-gnu
fi

# load an environment that has mpi4py and treecorr with OpenMP support

if [ -n "$SLURM_CONSTRAINT" ]; then
  source activate erfanxyz-1       # Cori
else
  source activate erfanxyz-edison  # Edison
fi

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  omp_threads=$((1*$SLURM_CPUS_PER_TASK)) # 2X for hyper_threading (but we already did the multiplication in cpu_per_task) so we don't need it here
else
  omp_threads=1
fi

export OMP_NUM_THREADS=$omp_threads
export OMP_PLACES=threads

srun --unbuffered python calc_corr.py $1 $2 $3 $4 $SLURM_JOB_ID $omp_threads

end_time="$(date -u +%s)"
secs="$(($end_time-$start_time))"
mv tmp/${SLURM_JOB_ID}_${SLURM_JOB_NAME}_running tmp/${SLURM_JOB_ID}_${SLURM_JOB_NAME}_done
mv ${SLURM_JOB_NAME}.out tmp/${SLURM_JOB_ID}_${SLURM_JOB_NAME}_done/${SLURM_JOB_NAME}.out
mv ${SLURM_JOB_NAME}.err tmp/${SLURM_JOB_ID}_${SLURM_JOB_NAME}_done/${SLURM_JOB_NAME}.err
printf 'SLURM Runtime: %02dh:%02dm:%02ds\n' $(($secs/3600)) $(($secs%3600/60)) $(($secs%60))
