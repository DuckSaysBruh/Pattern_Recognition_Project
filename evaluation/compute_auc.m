function a=compute_auc(fpr,tpr)
assert(numel(fpr)==numel(tpr) && all(diff(fpr)>=0),'Invalid ordered ROC.');
a=trapz(fpr,tpr); assert(a>=-1e-12 && a<=1+1e-12,'Invalid AUC.');
a=max(0,min(1,a));
end
