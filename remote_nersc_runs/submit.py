import os
import yamlplus as yp
# from fstring import fstring

# load configurations
config_file = 'config.yaml'
cfg = yp.load(open(config_file))
globals().update(cfg)

# htfactor = 2 if hyperthreading else 1

if not parallel:
	print(f"Warning: the job is not running in parallel mode. Set `parallel` to True in `{config_file}` to get the best efficiency.")

# make the sbatch file
with open(bashfname, 'w') as f:
	f.writelines("#!/bin/bash\n")
	f.writelines(f"#SBATCH -A {project}\n")
	if qos:
		f.writelines(f"#SBATCH -q {qos}\n") # -q instead of -p on NERSC premium?
	# if partition:
	# 	f.writelines(f"#SBATCH -p {partition}\n")
	if constraint:
		f.writelines(f"#SBATCH -C {constraint}\n")
	f.writelines(f"#SBATCH -N {nodes}\n")
	f.writelines(f"#SBATCH -n {tasks}\n")
	f.writelines(f"#SBATCH -c {cpus_per_task}\n")
	f.writelines(f"#SBATCH -t {runtime}\n")
	f.writelines(f"#SBATCH -J {jobname}\n")
	f.writelines(f"#SBATCH -o {outfname}\n")
	f.writelines(f"#SBATCH -e {errfname}\n")
	f.writelines(f"#SBATCH --mem={memory_max}\n")
	f.writelines(f"#SBATCH --mail-type={mailtype}\n")
	f.writelines(f"#SBATCH --mail-user={email}\n")
	f.writelines("\ndate")
	f.writelines("""\nprintf "host name: "\n
if [ ! -z "$NERSC_HOST" ]; then
  printf "$NERSC_HOST - "
fi\n\nhostname\n\n""")
	f.writelines("""start_time="$(date -u +%s)"\n""")
	f.writelines("""echo "SLURM_JOB_ID:" $SLURM_JOB_ID\n""")
	# f.writelines("module load python\n")
	f.writelines(f"source activate {conda_env}\n")
	f.writelines(f"""
echo "Running on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS tasks, each with $SLURM_CPUS_PER_TASK cores."
if $( echo ${{LOADEDMODULES}} | grep --quiet 'PrgEnv-intel' ); then
  echo $'Had to do: module swap PrgEnv-intel PrgEnv-gnu\\n';
  module swap PrgEnv-intel PrgEnv-gnu
elif $( echo ${{LOADEDMODULES}} | grep --quiet 'PrgEnv-cray' ); then
  echo $'Had to do: module swap PrgEnv-cray PrgEnv-gnu\\n';
  module swap PrgEnv-cray PrgEnv-gnu
elif $( echo ${{LOADEDMODULES}} | grep --quiet 'PrgEnv-gnu' ); then
  echo $'PrgEnv-gnu is already loaded\\n';
else
  echo $'Had to do: module load PrgEnv-gnu\\n';
  module load PrgEnv-gnu
fi\n
if [ -n "$SLURM_CPUS_PER_TASK" ] && [ $SLURM_CPUS_PER_TASK != 1 ]; then
  omp_threads=$(($SLURM_CPUS_PER_TASK)) # 2X for hyperthreading (if you already did the multiplication in cpu_per_task you don't need to multiply by 2 here)
else
  omp_threads=1
fi\n
export OMP_NUM_THREADS=$omp_threads
export OMP_PLACES=threads\n
""")
	f.writelines(f"srun --unbuffered --cpu_bind=cores python {pyfname}")
	f.writelines("""\n
#sacct -j $SLURM_JOB_ID --format JobID,Partition,Submit,Start,End,NodeList%40,ReqMem,MaxRSS,MaxRSSNode,MaxRSSTask,MaxVMSize,ExitCode\n
date
end_time="$(date -u +%s)"
secs="$(($end_time-$start_time))"
printf 'SLURM runtime: %02dh:%02dm:%02ds\\n' $(($secs/3600)) $(($secs%3600/60)) $(($secs%60))
""")

# submit the job using python
# os.system("module load python")
# os.system(f"source activate {conda_env}")
# os.system(f"sbatch {bashfname}")

# ------------------
# - email settings
# ------------------

# body = "Hi Erfan,\n\nPlease take a look at the attached error and output files.\n\nSent by Erfan's automated script :)"

bash_lines = f"""
JID=$(sbatch {bashfname})
echo $JID
sleep 20s # needed
ST="PENDING"
while [ "$ST" != "COMPLETED" ] ; do
   ST=$(sacct -j ${{JID##* }} -o State | awk 'FNR == 3 {{print $1}}')
   sleep 3m
   if [ "$ST" == "FAILED" ]; then
      echo 'Job final status:' $ST, exiting...
      exit 122
   fi
echo $ST
echo -e "$ST" | mailx -s "$JID" -a {outfname} -a {errfname} "erfan@ucdavis.edu"
"""

os.system(bash_lines)

# echo -e "attached are out and err files" | mailx -s "Batch job COMPLETED" -a "get_observed_0_0_r3.out" -a "get_observed_0_0_r3.err" "tyson@physics.ucdavis.edu erfan@ucdavis.edu"

