function testDir=main_test(modelPath,dataPath,labelCSV)
% Any supported filename is allowed. Labels come ONLY from an optional CSV.
% Interactive: main_test. Scripted no labels: main_test(modelPath,dataPath,'').
% CSV columns: File,Label. File is a unique basename including extension.
projectRoot=fileparts(mfilename('fullpath')); addpath(genpath(projectRoot));
if nargin<1 || isempty(modelPath)
    [f,p]=uigetfile('*.mat','Select final_model.mat');
    if isequal(f,0), testDir=''; return; end
    modelPath=fullfile(p,f);
end
if nargin<2 || isempty(dataPath)
    dataPath=uigetdir(projectRoot,'Select TEST images');
    if isequal(dataPath,0), testDir=''; return; end
end
if nargin<3
    answer=questdlg('Do you have a ground-truth label CSV?', ...
        'Test labels','Choose CSV','Predictions only','Cancel','Predictions only');
    if isempty(answer)||strcmp(answer,'Cancel'), testDir=''; return; end
    labelCSV='';
    if strcmp(answer,'Choose CSV')
        [f,p]=uigetfile('*.csv','Select CSV with File,Label columns');
        if isequal(f,0), testDir=''; return; end
        labelCSV=fullfile(p,f);
    end
end
S=load(modelPath,'model'); assert(isfield(S,'model'),'MAT file has no model.'); model=S.model;
% requireLabels=false permits arbitrary test basenames.
% Discard filename-inferred labels deliberately: only explicit CSV is ground truth.
[X,~,meta]=load_data(dataPath,model.cfg,false); N=size(X,1); y=nan(N,1);
baseNames=cell(N,1);
for i=1:N
    [~,b,e]=fileparts(meta.files{i}); baseNames{i}=[b e];
end
labelSource='none';
if ~isempty(labelCSV)
    assert(numel(unique(baseNames))==N, ...
        'CSV matching requires unique basenames across all test subfolders.');
    T=readtable(labelCSV,'TextType','string');
    assert(all(ismember({'File','Label'},T.Properties.VariableNames)), ...
        'CSV must contain File and Label columns.');
    names=cellstr(string(T.File));
    assert(numel(unique(names))==numel(names),'Duplicate File entries in CSV.');
    if isnumeric(T.Label), csvLabels=double(T.Label);
    else, csvLabels=str2double(string(T.Label)); end
    assert(all(isfinite(csvLabels))&&all(ismember(csvLabels,[0 1])), ...
        'CSV Label values must be 0 (male) or 1 (female).');
    [found,ix]=ismember(baseNames,names);
    assert(all(found),'Some test images have no matching CSV row (check extension/case).');
    y=csvLabels(ix); y=y(:); labelSource=labelCSV;
end
pipeline_predict(model,X); times=zeros(model.cfg.timing.repeats,1);
for j=1:numel(times)
    t=tic; [pred,scores]=pipeline_predict(model,X); times(j)=toc(t);
end
inferMs=1000*median(times)/N;
testDir=create_run_dir(fileparts(modelPath),['test_' model.method]);
P=table(meta.files,y,pred,scores, ...
    'VariableNames',{'File','TrueLabel','Prediction','FemaleScore'});
writetable(P,fullfile(testDir,'test_predictions.csv'));
report.method=model.method; report.n_images=N;
report.infer_ms_per_image=inferMs; report.label_source=labelSource;
report.accuracy=NaN; report.auc=NaN;
if all(isfinite(y))
    report.accuracy=compute_accuracy(y,pred);
    report.confusion_matrix=[sum(y==0 & pred==0),sum(y==0 & pred==1); ...
        sum(y==1 & pred==0),sum(y==1 & pred==1)];
    fprintf('Test accuracy %.2f%%\n',100*report.accuracy);
    if numel(unique(y))==2
        [report.roc_fpr,report.roc_tpr]=compute_roc(y,scores);
        report.pooled_auc=compute_auc(report.roc_fpr,report.roc_tpr);
        report.auc=report.pooled_auc;
        plot_roc({report},fullfile(testDir,'test_roc.png'));
        fprintf('Test AUC %.4f\n',report.auc);
    else
        fprintf('Only one true class: ROC/AUC not reported.\n');
    end
else
    fprintf('No ground-truth CSV: predictions only; Accuracy/ROC/AUC not reported.\n');
end
summary=table({model.method},N,report.accuracy,report.auc,inferMs, ...
    'VariableNames',{'Method','NImages','Accuracy','AUC','InferMsPerImage'});
writetable(summary,fullfile(testDir,'test_summary.csv'));
% Template uses basenames; fill Label manually from verified ground truth.
if numel(unique(baseNames))==N
    labelTemplate=table(baseNames,nan(N,1),'VariableNames',{'File','Label'});
    writetable(labelTemplate,fullfile(testDir,'labels_template.csv'));
end
save(fullfile(testDir,'test_results.mat'),'report','P','modelPath','dataPath','labelCSV');
fprintf('Inference %.3f ms/image; output: %s\n',inferMs,testDir);
end
