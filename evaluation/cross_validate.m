function r=cross_validate(X,y,meta,foldID,method,cfg)
K=cfg.cv.num_folds; N=numel(y);
r.method=method; r.predictions=nan(N,1); r.scores=nan(N,1);
r.fold_fpr=cell(K,1); r.fold_tpr=cell(K,1); r.train_info=cell(K,1);
values=zeros(K,17);
for f=1:K
    tr=foldID~=f; va=foldID==f;
    tp=unique(meta.person_id(tr)); vp=unique(meta.person_id(va)); overlap=numel(intersect(tp,vp));
    if strcmp(cfg.cv.mode,'person_grouped'), assert(overlap==0,'Person overlap.'); end
    fprintf('Fold %d/%d: train=%d, validation=%d, person overlap=%d\n', ...
        f,K,sum(tr),sum(va),overlap);
    localCfg=cfg; localCfg.mlp.seed=cfg.mlp.seed+f;
    [model,info]=pipeline_train(X(tr,:),y(tr),method,localCfg);
    A=X(va,:);
    pipeline_predict(model,A); % warm-up
    times=zeros(cfg.timing.repeats,1);
    for j=1:numel(times)
        t=tic; [pred,scores]=pipeline_predict(model,A); times(j)=toc(t);
    end
    seconds=median(times);
    r.predictions(va)=pred; r.scores(va)=scores;
    [fpr,tpr]=compute_roc(y(va),scores); auc=compute_auc(fpr,tpr);
    acc=compute_accuracy(y(va),pred);
    r.fold_fpr{f}=fpr; r.fold_tpr{f}=tpr; r.train_info{f}=info;
    explained=NaN;
    if isfield(model.reduction,'var_ratio'), explained=100*sum(model.reduction.var_ratio); end
    values(f,:)=[f,sum(tr),sum(va),numel(tp),numel(vp),overlap, ...
        info.feature_dim,acc,auc,info.classifier_param_count, ...
        info.pipeline_logical_stored_scalars,info.normalize_seconds, ...
        info.reduction_seconds,info.classifier_seconds,info.train_seconds, ...
        seconds/sum(va)*1000,explained];
    fprintf('  Accuracy %.2f%% | AUC %.4f | train %.3fs | infer %.3f ms/image\n', ...
        100*acc,auc,info.train_seconds,seconds/sum(va)*1000);
    clear model A;
end
assert(all(isfinite(r.predictions)) && all(isfinite(r.scores)),'Missing OOF predictions.');
r.fold_table=array2table(values,'VariableNames', ...
    {'Fold','NTrain','NValidation','NTrainPersons','NValidationPersons','PersonOverlap', ...
    'FeatureDim','Accuracy','AUC','ClassifierStoredParameters','PipelineLogicalStoredScalars', ...
    'NormalizeSeconds','ReductionSeconds','ClassifierSeconds','TrainSeconds', ...
    'InferMsPerImage','PCARetainedVariancePercent'});
[r.roc_fpr,r.roc_tpr]=compute_roc(y,r.scores);
r.pooled_auc=compute_auc(r.roc_fpr,r.roc_tpr);
r.pooled_accuracy=compute_accuracy(y,r.predictions);
r.confusion_matrix=[sum(y==0 & r.predictions==0),sum(y==0 & r.predictions==1); ...
    sum(y==1 & r.predictions==0),sum(y==1 & r.predictions==1)];
end
