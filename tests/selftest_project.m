function selftest_project()
% Synthetic tests; no course images required. Run before the first experiment.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(root));
cfg=config_project(); cfg.pca.n_dims=3; cfg.mlp.epochs=30;
rng(7); y=[zeros(20,1);ones(20,1)]; X=randn(40,12)+y*ones(1,12);
ids=arrayfun(@(i) sprintf('person_%d',ceil(i/2)),(1:40)','UniformOutput',false);
f=make_folds(y,ids,cfg.cv);
for k=1:5, assert(isempty(intersect(ids(f==k),ids(f~=k)))); end
[A,state]=normalize_data(X(1:30,:),cfg.normalize);
B=normalize_data(X(31:end,:),state); assert(all(isfinite(B(:))));
p=pca_train(A,cfg.pca); assert(norm(p.W'*p.W-eye(3),'fro')<1e-8);
% Dual ridge LDA agrees with a direct small-dimensional reference.
l=lda_train(X,y,cfg.lda); R=X;
for c=0:1, ix=y==c; R(ix,:)=bsxfun(@minus,X(ix,:),mean(X(ix,:),1)); end
R=R/sqrt(38); delta=(mean(X(y==1,:),1)-mean(X(y==0,:),1))';
w=(R'*R+l.lambda*eye(12))\delta; w=w/norm(w);
assert(abs(w'*l.W)>1-1e-8,'Dual LDA disagrees with direct solve.');
[a,b]=compute_roc([0;1],[0.1;0.9]); assert(abs(compute_auc(a,b)-1)<1e-12);
[a,b]=compute_roc([0;1],[0.5;0.5]); assert(abs(compute_auc(a,b)-0.5)<1e-12);
[a,b]=compute_roc([0;1],[0.9;0.1]); assert(abs(compute_auc(a,b))<1e-12);
methods={'pca_bayes','lda_bayes','pca_mlp','lda_mlp'};
for i=1:4
    [model,info]=pipeline_train(X,y,methods{i},cfg);
    [pred,s]=pipeline_predict(model,X);
    assert(numel(pred)==40 && all(isfinite(s)) && all(s>=0 & s<=1));
    assert(info.train_seconds>=0);
end
fprintf('All synthetic tests passed. This does not validate the real dataset.\n');
end
