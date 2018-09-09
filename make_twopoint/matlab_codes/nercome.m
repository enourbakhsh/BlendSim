function [Sigmahat, values] = nercome(Y, M, m, splits)

%%% NERCOME as in Lam (2015), with M permutation of data. 

%%% function [Sigmahat, values, P1, Stilde2avg] = nercome(Y, M, m, splits) created by Clifford Lam 14-09-2014
%%% The split m has default value 7, to be chosen by minimizing ||Sigmahat_m -
%%% Stilde2avg||_F.
%%% 
%%% Input : Y - The data matrix, with size p x n
%%%         M - The number of permutations to perform.
%%%         m - The number of splits to search for.
%%%    splits - The split locations specified by user, has to have length m. 
%%% Output : Sigmahat - The NERCOME covariance matrix estimator. 
%%%          values   - The Frobenius error norm under different splits.



[p,n] = size(Y);
data = Y - mean(Y,2)*ones(1,n); %%% Centering to 0



if nargin==1
    M=50; m=7; splits = round([2*n^.5, 0.2*n, 0.4*n, 0.6*n, 0.8*n, n-2.5*n^.5, n-1.5*n^.5]);
end

if nargin==2
    m=7; splits = round([2*n^.5, 0.2*n, 0.4*n, 0.6*n, 0.8*n, n-2.5*n^.5, n-1.5*n^.5]);
end



values = zeros(m,1);
Shattemp = zeros(p,p,m);

%P1 = zeros(p,p,M); %%% Added for effsim1.m etc


for i=1:m
     Stilde2avg = zeros(p,p); 
     for  j=1:M
         permset = randperm(size(Y,2));
         Ytemp = data(:,permset);
         X1 = Ytemp(:,1:splits(i)) - mean(Ytemp(:,1:splits(i)),2)*ones(1,splits(i));
         X2 = Ytemp(:,splits(i)+1:end) - mean(Ytemp(:,splits(i)+1:end),2)*ones(1,n-splits(i));
         S1 = X1*X1'/splits(i);
         [U1, D1] = svd(S1);
         Stilde2 = X2*X2'/(n-splits(i));
         d1 = diag(U1'*Stilde2*U1);
         Shattemp(:,:,i) = Shattemp(:,:,i) + U1*diag(d1)*U1'/M;
         %P1(:,:,j) = U1; %%% Added for effsim1.m etc
         Stilde2avg = Stilde2avg + Stilde2/M; 
     end
     values(i) = sum(sum((Shattemp(:,:,i)-Stilde2avg).^2)); 
end

[a,b] = min(values);
Sigmahat = Shattemp(:,:,b);




