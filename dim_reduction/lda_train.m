function model=lda_train(X,y,cfg)
% Binary ridge Fisher LDA, solved in sample space via Woodbury.
% Sw=R'*R (within-class covariance); w=(Sw+lambda*I)\delta.
% No D-by-D arrays and no preceding PCA.
[N,D]=size(X); assert(isequal(unique(y),[0;1]),'Binary labels required.');
assert(N>2,'Too few samples.');
model.mean_vec=mean(X,1); R=X;
means=zeros(2,D);
for c=0:1
    ix=(y==c); means(c+1,:)=mean(X(ix,:),1);
    R(ix,:)=bsxfun(@minus,X(ix,:),means(c+1,:));
end
R=R/sqrt(N-2);
scale=sum(R(:).^2)/D;
lambda=max(cfg.ridge_absolute,cfg.ridge_relative*scale);
assert(lambda>0,'LDA ridge must be positive.');
delta=(means(2,:)-means(1,:))';
G=R*R'; G=(G+G')/2;
w=(delta-R'*((G+lambda*eye(N))\(R*delta)))/lambda;
normW=norm(w); assert(isfinite(normW) && normW>0,'Degenerate LDA direction.');
model.W=w/normW; model.n_dims=1; model.lambda=lambda;
model.stored_scalars=numel(model.mean_vec)+numel(model.W);
end
