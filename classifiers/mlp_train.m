function model=mlp_train(X,y,cfg)
% Full-batch gradient descent, sigmoid hidden/output, binary cross entropy.
% Training-only reduced-feature scaling is part of this classifier.
assert(isequal(unique(y),[0;1]),'Binary classes required.');
rng(cfg.seed,'twister'); [N,D]=size(X); H=cfg.hidden_size;
assert(H>=1 && H==floor(H),'Invalid hidden size.');
model.input_mu=mean(X,1); model.input_sigma=std(X,0,1);
model.input_sigma(model.input_sigma<1e-8)=1;
A=bsxfun(@rdivide,bsxfun(@minus,X,model.input_mu),model.input_sigma);
W1=randn(D,H)*sqrt(2/(D+H)); b1=zeros(1,H);
W2=randn(H,1)*sqrt(2/(H+1)); b2=0;
model.loss_history=zeros(cfg.epochs,1);
for ep=1:cfg.epochs
    hidden=stable_sigmoid(bsxfun(@plus,A*W1,b1));
    logits=hidden*W2+b2; p=stable_sigmoid(logits);
    loss=mean(max(logits,0)-y.*logits+log1p(exp(-abs(logits)))) ...
        +0.5*cfg.l2*(sum(W1(:).^2)+sum(W2(:).^2));
    model.loss_history(ep)=loss;
    d2=(p-y)/N;
    gW2=hidden'*d2+cfg.l2*W2; gb2=sum(d2);
    d1=(d2*W2').*hidden.*(1-hidden);
    gW1=A'*d1+cfg.l2*W1; gb1=sum(d1,1);
    W1=W1-cfg.learning_rate*gW1; b1=b1-cfg.learning_rate*gb1;
    W2=W2-cfg.learning_rate*gW2; b2=b2-cfg.learning_rate*gb2;
    assert(isfinite(loss) && all(isfinite(W1(:))),'MLP diverged.');
end
model.W1=W1; model.b1=b1; model.W2=W2; model.b2=b2;
model.kind='mlp'; model.param_count=D*H+H+H+1;
model.free_param_count=model.param_count;
model.preprocessing_stored_scalars=2*D;
end
function p=stable_sigmoid(z)
p=zeros(size(z)); ix=z>=0; p(ix)=1./(1+exp(-z(ix)));
e=exp(z(~ix)); p(~ix)=e./(1+e);
end
