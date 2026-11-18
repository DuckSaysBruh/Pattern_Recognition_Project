function [pred,scores]=pipeline_predict(model,X)
Z=normalize_data(X,model.normalization);
parts=strsplit(model.method,'_');
if strcmp(parts{1},'pca'), A=pca_project(model.reduction,Z);
else, A=lda_project(model.reduction,Z); end
if strcmp(parts{2},'bayes'), [pred,scores]=bayes_predict(model.classifier,A);
else, [pred,scores]=mlp_predict(model.classifier,A); end
assert(iscolumn(pred) && iscolumn(scores),'Predictions must be columns.');
assert(all(isfinite(scores)) && all(scores>=0 & scores<=1),'Invalid posterior scores.');
end
