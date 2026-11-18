function [pred,scores]=mlp_predict(model,X)
A=bsxfun(@rdivide,bsxfun(@minus,X,model.input_mu),model.input_sigma);
h=sigmoid_stable(bsxfun(@plus,A*model.W1,model.b1));
scores=sigmoid_stable(h*model.W2+model.b2); pred=double(scores>=0.5);
end
function p=sigmoid_stable(z)
p=zeros(size(z)); ix=z>=0; p(ix)=1./(1+exp(-z(ix)));
e=exp(z(~ix)); p(~ix)=e./(1+e);
end
