function [fpr,tpr]=compute_roc(y,scores)
% Tied scores enter together. Threshold moves from +Inf downwards.
y=y(:); scores=scores(:);
assert(numel(y)==numel(scores) && all(isfinite(scores)),'Invalid scores.');
assert(isequal(unique(y),[0;1]),'ROC requires both classes.');
[s,ix]=sort(scores,'descend'); yy=y(ix);
ends=[find(diff(s)~=0); numel(s)];
tp=cumsum(yy==1); fp=cumsum(yy==0);
tpr=[0; tp(ends)/sum(y==1)]; fpr=[0; fp(ends)/sum(y==0)];
end
