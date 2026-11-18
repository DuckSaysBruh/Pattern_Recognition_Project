function model=bayes_train(X,y,cfg)
model.classes=[0;1]; D=size(X,2); K=2;
model.means=cell(K,1); model.covs=cell(K,1); model.chol=cell(K,1);
model.priors=zeros(K,1);
assert(cfg.ridge_absolute>0,'Bayes ridge must be positive.');
for j=1:K
    A=X(y==model.classes(j),:); n=size(A,1);
    assert(n>=2,'Not enough class samples.');
    model.means{j}=mean(A,1); model.priors(j)=n/size(X,1);
    B=bsxfun(@minus,A,model.means{j});
    % Unbiased covariance, denominator n-1 (not MLE).
    C=(B'*B)/(n-1); C=(C+C')/2+cfg.ridge_absolute*eye(D);
    [U,p]=chol(C); assert(p==0,'Bayes covariance is not positive definite.');
    model.covs{j}=C; model.chol{j}=U;
end
model.param_count=K*(D+D^2)+K;
model.free_param_count=K*D+K*D*(D+1)/2+(K-1);
model.kind='bayes';
end
