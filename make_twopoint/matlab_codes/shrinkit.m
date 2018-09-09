function [] = shrinkit(fname,M)

%M = 1000; % number of realizations for nercome
datavector = readNPY(fname); %('/Users/erfan/Desktop/py/fakeblends/get_tiles/downloaded/zsnb/covmat.zsnb_r0/corrsupervec.npy');
[~,n] = size(datavector); 
[fpath,name,ext] = fileparts(fname);

%nercome shrinkage estimation
[shrunk_covmat_nercome,vals] = nercome(datavector, M); %, 1, round((2./3.)*n) );
mat2np(shrunk_covmat_nercome, strcat(fpath,'/shrunk_covmat_nercome.pkl'), 'float64');
