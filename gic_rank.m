% PCA rank selection by GIC (Hung, 07/27/2020)
%
% ----- INPUT -----
% x: n by p data matrix
% upperbound: <= p-1, search over [0, upperbound], defaul = floor(min([n,p])/2)
%
% ----- OUTPUT -----
% opt_rank: selected rank 
% GIC: penalized minus-loglikelihood
% b_gic: GIC penalty

function [opt_rank, GIC, b_gic] = gic_rank(x, upperbound)

[n,p] = size(x);

if ~exist('upperbound','var')
    upperbound = floor(min([n,p])/2);
end

[uu, ll] = svd(cov(x));
ll = diag(ll);  % sample eigenvalues
b_gic = zeros(upperbound+1,1);
for r = 0:upperbound
    sigma_r2 = sum(ll(r+1:end))/(p-r);  % MLE of sigma_r^2    
    ll_e = [ll(1:r); sigma_r2*ones(p-r,1)];
    likelihood(r+1) = -sum(log(ll_e))/2;  % log-likelihood
    for j = 1:r
        b_gic(r+1) = b_gic(r+1) + (ll(j)/sigma_r2-1)*sum(ll(r+1:end)./(ll(j)-ll(r+1:end)));
    end
    b_gic(r+1) = b_gic(r+1) + r*(r-1)/2 + r + mean(ll(r+1:end).^2)/(mean(ll(r+1:end)))^2 + p;
    GIC(r+1) = likelihood(r+1) - b_gic(r+1)/n;
end
[max_val, opt_rank] = max(GIC);
opt_rank = opt_rank-1;
end






