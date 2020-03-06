cd /global/project/projectdirs/m1727/erfanxyz_home/myprojects/packages
mkdir gcr-catalogs-buzzard-2
02:okdel erfan$ git clone https://github.com/LSSTDESC/gcr-catalogs.git .

cd ~/erfan_projects/get_observed
module load python
source activate for_erfan_py3

wget -N https://github.com/cosmicshear/BlendSim/raw/master/remote_nersc_runs/get_observed.py
wget -N https://github.com/cosmicshear/BlendSim/raw/master/remote_nersc_runs/config.yaml

python submit.py
