function runDir = main_train(cfg)
% Interactive method selection or scripted main_train(cfg).
projectRoot=fileparts(mfilename('fullpath')); addpath(genpath(projectRoot));
if nargin==0
    cfg=config_project();
    fprintf('\n1 PCA + Bayes\n2 LDA + Bayes\n3 PCA + MLP\n4 LDA + MLP\n5 All four\n0 Exit\n');
    choice=input('Select method [0-5]: ');
    if isequal(choice,0), runDir=''; return; end
    assert(isnumeric(choice)&&isscalar(choice)&&ismember(choice,1:5),'Invalid selection.');
    names={'pca_bayes','lda_bayes','pca_mlp','lda_mlp'};
    if choice==5, cfg.methods=names; else, cfg.methods=names(choice); end
end
if isempty(cfg.train_path)
    cfg.train_path=uigetdir(projectRoot,'Select TRAINING images');
    if isequal(cfg.train_path,0), error('No training folder selected.'); end
end
assert(iscell(cfg.methods)&&~isempty(cfg.methods),'cfg.methods must be a nonempty cell array.');
valid={'pca_bayes','lda_bayes','pca_mlp','lda_mlp'};
assert(all(ismember(cfg.methods,valid)),'Unknown method.');
if numel(cfg.methods)==1
    methodTag=cfg.methods{1};
elseif numel(cfg.methods)==4 && numel(unique(cfg.methods))==4
    methodTag='all4';
else
    methodTag=strjoin(cfg.methods,'-');
end
runTag=sprintf('train_%s_%dx%d_pca%d_%s',methodTag, ...
    cfg.image_size(1),cfg.image_size(2),cfg.pca.n_dims,cfg.cv.mode);
runDir=create_run_dir(cfg.output_root,runTag);
diary(fullfile(runDir,'console_log.txt'));
diaryCleanup=onCleanup(@() diary('off')); %#ok<NASGU>
fprintf('Output: %s\n',runDir);
t=tic; [X,y,meta]=load_data(cfg.train_path,cfg,true); loadSeconds=toc(t);
fprintf('%d images, %d pixels/image, males=%d, females=%d\n', ...
    size(X,1),size(X,2),sum(y==0),sum(y==1));
if numel(y)~=200 || sum(y==0)~=100 || sum(y==1)~=100
    warning('Expected 200 training images, 100 per class. Check dataset.');
end
foldID=make_folds(y,meta.person_id,cfg.cv);
manifest=table(meta.files,meta.person_id,y,foldID, ...
    'VariableNames',{'File','PersonID','Label','ValidationFold'});
writetable(manifest,fullfile(runDir,'dataset_and_folds.csv'));
save(fullfile(runDir,'config_and_folds.mat'),'cfg','meta','y','foldID');
versionInfo=version; products=ver;
save(fullfile(runDir,'environment.mat'),'versionInfo','products','loadSeconds');
results=cell(numel(cfg.methods),1);
for i=1:numel(cfg.methods)
    method=cfg.methods{i}; methodDir=fullfile(runDir,method); mkdir(methodDir);
    fprintf('\n===== %s =====\n',method);
    r=cross_validate(X,y,meta,foldID,method,cfg); results{i}=r;
    writetable(r.fold_table,fullfile(methodDir,'fold_results.csv'));
    predTable=table(meta.files,meta.person_id,y,foldID,r.predictions,r.scores, ...
        'VariableNames',{'File','PersonID','TrueLabel','Fold','Prediction','FemaleScore'});
    writetable(predTable,fullfile(methodDir,'oof_predictions.csv'));
    writetable(predTable(y~=r.predictions,:),fullfile(methodDir,'misclassified.csv'));
    save(fullfile(methodDir,'cv_results.mat'),'r');
    plot_roc({r},fullfile(methodDir,'roc.png'));
    fprintf('Final fit using all training images...\n');
    [model,trainInfo]=pipeline_train(X,y,method,cfg);
    model.training_person_ids=unique(meta.person_id);
    model.training_files=meta.files; model.matlab_version=versionInfo;
    save(fullfile(methodDir,'final_model.mat'),'model','trainInfo','-v7.3');
    ncheck=min(4,size(X,1)); [p1,s1]=pipeline_predict(model,X(1:ncheck,:));
    loaded=load(fullfile(methodDir,'final_model.mat'),'model');
    [p2,s2]=pipeline_predict(loaded.model,X(1:ncheck,:));
    assert(isequal(p1,p2)&&max(abs(s1-s2))<1e-12,'Model reload mismatch.');
    fprintf('Model saved and reload predictions verified.\n'); clear model loaded;
end
m=numel(results); Method=cell(m,1); AccuracyMean=zeros(m,1); AccuracyStd=zeros(m,1);
AUCMean=zeros(m,1); AUCStd=zeros(m,1); PooledOOFAUC=zeros(m,1);
FeatureDim=zeros(m,1); ClassifierStoredParameters=zeros(m,1);
MeanTrainSeconds=zeros(m,1); MeanInferMsPerImage=zeros(m,1);
for i=1:m
    r=results{i}; Method{i}=r.method;
    AccuracyMean(i)=mean(r.fold_table.Accuracy); AccuracyStd(i)=std(r.fold_table.Accuracy);
    AUCMean(i)=mean(r.fold_table.AUC); AUCStd(i)=std(r.fold_table.AUC);
    PooledOOFAUC(i)=r.pooled_auc; FeatureDim(i)=r.fold_table.FeatureDim(1);
    ClassifierStoredParameters(i)=r.fold_table.ClassifierStoredParameters(1);
    MeanTrainSeconds(i)=mean(r.fold_table.TrainSeconds);
    MeanInferMsPerImage(i)=mean(r.fold_table.InferMsPerImage);
end
summary=table(Method,AccuracyMean,AccuracyStd,AUCMean,AUCStd,PooledOOFAUC, ...
    FeatureDim,ClassifierStoredParameters,MeanTrainSeconds,MeanInferMsPerImage);
disp(summary); writetable(summary,fullfile(runDir,'summary.csv'));
plot_roc(results,fullfile(runDir,'roc_comparison.png'));
save(fullfile(runDir,'experiment.mat'),'results','summary','cfg','foldID','meta','y');
fprintf('\nFinished: %s\n',runDir);
end
