#!/bin/bash

# NOTE: 48 cores per node in Edison means hyperthreading because there are 24 physical cores in Edison

# -----------------
# machine info
# -----------------

hn=$(hostname)
constraint='haswell' # only important for cori

# -----------------
# initialization
# -----------------

pp=0 
ps=0 
ss=0 

# -----------------
# catalog regime
# -----------------

regimes=() 
# uncomment the ones you don't want:
# regimes+=('zsnb')
regimes+=('zsb')
# regimes+=('zpnb')
# regimes+=('zpb')

# -----------------
# correlation type
# -----------------

ctypes=() 
# uncomment the ones you don't want:
# ctypes+=('pp')
# ctypes+=('ps')
ctypes+=('ss')

echo
echo regimes: {"${regimes[@]}"}
echo ctypes: {"${ctypes[@]}"}
echo

declare -A pp=( [pp]=1 [ps]=0 [ss]=0 )
declare -A ps=( [pp]=0 [ps]=1 [ss]=0 )
declare -A ss=( [pp]=0 [ps]=0 [ss]=1 )

declare -A           nodes=( [pp]=70  [ps]=30 [ss]=30 ) # pp zsb at cori N:55 otherwise for edison increase N to 70 26min
declare -A   cpus_per_task=( [pp]=6   [ps]=6  [ss]=6  ) # 6 dison - 8 cori
declare -A ntasks_per_node=( [pp]=8   [ps]=8  [ss]=8  )

# declare -A           nodes=( [pp]=70  [ps]=24 [ss]=22 ) # pp zsb at cori N:55 otherwise for edison increase N to 70 26min
# declare -A ntasks_per_node=( [pp]=8   [ps]=8  [ss]=8  )
# declare -A   cpus_per_task=( [pp]=6   [ps]=6  [ss]=6  ) # 6 dison - 8 cori

for regime in "${regimes[@]}"; do
	for ctype in "${ctypes[@]}"; do # I decided to submit a separate job for each ctype - you can do all of them together, though
		if [[ $hn = "cori"* ]]; then
			sbatch --nodes=${nodes["$ctype"]} --ntasks-per-node=${ntasks_per_node["$ctype"]} --cpus-per-task=${cpus_per_task["$ctype"]} -p debug -t 00:30:00 -J ${regime}_${ctype} -o ${regime}_${ctype}.out -e ${regime}_${ctype}.err --constraint=${constraint} calc_corr.sh $regime ${pp["${ctype}"]} ${ps["${ctype}"]} ${ss["${ctype}"]} 
		else # edison
			sbatch --nodes=${nodes["$ctype"]} --ntasks-per-node=${ntasks_per_node["$ctype"]} --cpus-per-task=${cpus_per_task["$ctype"]} -p debug -t 00:30:00 -J ${regime}_${ctype} -o ${regime}_${ctype}.out -e ${regime}_${ctype}.err calc_corr.sh $regime ${pp["${ctype}"]} ${ps["${ctype}"]} ${ss["${ctype}"]} 
		fi
		echo "$hn" is your host :: "$regime"_"$ctype" :: ${nodes["${ctype}"]} nodes :: ${ntasks_per_node["$ctype"]} tasks per node :: ${cpus_per_task["$ctype"]} cpus per task
		echo
	done 
done

squeue -u erfanxyz
echo