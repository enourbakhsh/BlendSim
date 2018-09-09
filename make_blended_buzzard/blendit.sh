#!/bin/bash -l
#SBATCH --constraint=haswell
#SBATCH -p debug
#SBATCH -N 43 #36                                # Maximum 32 processors per node in cori
#SBATCH -t 00:30:00
#SBATCH -A m1727
#SBATCH -J blendit_5p
#SBATCH -o blendit_5p.out
#SBATCH -e blendit_5p.err
#SBATCH --mail-type=END                    # Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=erfan@ucdavis.edu      # Destination email address
#SBATCH --ntasks=860 #715                  # number of MPI tasks/jobs


start_time="$(date -u +%s)"
echo "SLURM_JOB_ID" $SLURM_JOB_ID

HOSTNAME=$(echo $NERSC_HOST)
if [[ $(echo $HOSTNAME | grep -c "edison") > 0 ]]; then
    echo $'Edison is your host';
    HOST=edison
#   source activate erfanxyz-edison  # Edison
#   NNODE=40
#   DNODE=$((NNODE*2))
#   NODETYPE=ivybridge
elif [[ $(echo $HOSTNAME |grep -c "cori") > 0 ]]; then
    echo $'Cori is your host';
#   HOST=cori
#   source activate erfanxyz-1       # Cori
#   NNODE=4
#   DNODE=$((NNODE*8))
#   NODETYPE=haswell
else
    echo "unknown host",$HOST
    exit
fi

module load python/3.6-anaconda-4.4 #/2.7-anaconda-4.4

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

# use --unbuffered to update output files without delay
srun --unbuffered python blendit.py 

end_time="$(date -u +%s)"
secs="$(($end_time-$start_time))"
printf 'SLURM Runtime: %02dh:%02dm:%02ds\n' $(($secs/3600)) $(($secs%3600/60)) $(($secs%60))
