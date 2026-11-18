function model=pca_train(X,cfg)
[N,D]=size(X); k=cfg.n_dims;
assert(k>=1 && k==floor(k) && k<=min(N-1,D),'Invalid PCA dimension.');
model.mean_vec=mean(X,1); A=bsxfun(@minus,X,model.mean_vec);
% Economy SVD: never construct the pixel-by-pixel covariance matrix.
[~,S,V]=svd(A,'econ'); lambda=diag(S).^2/(N-1);
assert(~isempty(lambda) && lambda(1)>0,'Zero-variance training data.');
numericalRank=sum(lambda>lambda(1)*cfg.relative_tolerance);
assert(k<=numericalRank,'Requested PCA dimension exceeds numerical rank.');
model.W=V(:,1:k); model.eigenvalues=lambda(1:k);
model.var_ratio=lambda(1:k)/sum(lambda); model.n_dims=k;
model.stored_scalars=numel(model.mean_vec)+numel(model.W);
end
