function [pred,scores]=bayes_predict(model,X)
N=size(X,1); D=size(X,2); logp=zeros(N,2);
for j=1:2
    B=bsxfun(@minus,X,model.means{j}); U=model.chol{j};
    A=B/U;
    logp(:,j)=-0.5*sum(A.^2,2)-sum(log(diag(U))) ...
        -0.5*D*log(2*pi)+log(model.priors(j));
end
[~,ix]=max(logp,[],2); pred=model.classes(ix);
p=exp(bsxfun(@minus,logp,max(logp,[],2)));
p=bsxfun(@rdivide,p,sum(p,2)); scores=p(:,2);
end
