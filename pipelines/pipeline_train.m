function [model,info]=pipeline_train(X,y,method,cfg)
parts=strsplit(method,'_'); assert(numel(parts)==2,'Invalid method.');
model.method=method; model.cfg=cfg;
t=tic;
[Z,model.normalization]=normalize_data(X,cfg.normalize);
info.normalize_seconds=toc(t);
t=tic;
if strcmp(parts{1},'pca')
    model.reduction=pca_train(Z,cfg.pca);
    A=pca_project(model.reduction,Z);
elseif strcmp(parts{1},'lda')
    model.reduction=lda_train(Z,y,cfg.lda);
    A=lda_project(model.reduction,Z);
else
    error('Unknown reduction.');
end
info.reduction_seconds=toc(t);
clear Z;
t=tic;
if strcmp(parts{2},'bayes')
    model.classifier=bayes_train(A,y,cfg.bayes);
elseif strcmp(parts{2},'mlp')
    model.classifier=mlp_train(A,y,cfg.mlp);
else
    error('Unknown classifier.');
end
info.classifier_seconds=toc(t);
info.train_seconds=info.normalize_seconds+info.reduction_seconds+info.classifier_seconds;
info.feature_dim=size(A,2);
info.classifier_param_count=model.classifier.param_count;
info.classifier_free_param_count=model.classifier.free_param_count;
info.reduction_stored_scalars=model.reduction.stored_scalars;
info.normalization_stored_scalars=numel(model.normalization.mu)+numel(model.normalization.sigma);
info.classifier_preprocessing_scalars=0;
if isfield(model.classifier,'preprocessing_stored_scalars')
    info.classifier_preprocessing_scalars=model.classifier.preprocessing_stored_scalars;
end
info.pipeline_logical_stored_scalars=info.reduction_stored_scalars+ ...
    info.normalization_stored_scalars+info.classifier_param_count+ ...
    info.classifier_preprocessing_scalars;
model.label_meaning='0=male, 1=female';
model.train_info=info;
end
