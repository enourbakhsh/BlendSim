#!/bin/bash -l
#SBATCH --constraint=haswell
#SBATCH -p debug
#SBATCH -N 2 #40 #46
#SBATCH -t 00:30:00
#SBATCH -A m1727
#SBATCH -J zsnb_all3_scut
#SBATCH --output=zsnb_all3_multinest.out
#SBATCH --error=zsnb_all3_multinest.err
#SBATCH --cpus-per-task=6 #64 cori, 48 edison
#SBATCH --ntasks-per-node=8 # with 64 nodes --> -n 1536

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

source ~/myprojects/packages/cosmosis/config/setup-cosmosis-nersc

# You need only do these bits once
export COSMOSIS_OMP=1

# if $NERSC_HOST == "edison"

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  omp_threads=$((1*$SLURM_CPUS_PER_TASK)) # 2X for hyper_threading (but we already did the multiplication in cpu_per_task) so we don't need it here
else
  omp_threads=1
fi
export OMP_NUM_THREADS=$omp_threads
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

srun --unbuffered --cpu_bind=cores cosmosis --mpi /global/u1/e/erfanxyz/myprojects/cosmosis-run/bz1.6/analyze.ini
# hyperthreading is used

end_time="$(date -u +%s)"
secs="$(($end_time-$start_time))"
printf 'SLURM Runtime: %02dh:%02dm:%02ds\n' $(($secs/3600)) $(($secs%3600/60)) $(($secs%60))
