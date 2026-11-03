%% main_train.m  –  Member 1 (Integration / Project Lead)
% Gender Recognition Grand Challenge  |  CVR Lab
%
% DESCRIPTION
%   Loads the training dataset, defines all four pipelines, runs 5-fold
%   cross-validation, prints a comparison table, and saves ROC plots.
%
% USAGE
%   Run from the project root in MATLAB:
%     >> main_train
%
% OUTPUT
%   results/roc_cv.png      – ROC curves from 5-fold CV
%   results/cv_results.mat  – struct array with all CV metrics

clear; clc; close all;
addpath(genpath(pwd));   % add data/, pipelines/, dim_reduction/, classifiers/, evaluation/
if ~exist('results','dir'), mkdir('results'); end

%% ── Configuration ────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = 'train';     % folder containing training PNG images
IMG_SIZE        = [64, 64];    % resize all images to [H W] before flattening
K_FOLDS         = 5;           % number of cross-validation folds

PCA_DIMS   = 50;    % number of principal components to retain (tune)
MLP_HIDDEN = 64;    % MLP hidden-layer units                    (tune)
MLP_LR     = 0.01;  % MLP learning rate                         (tune)
MLP_EPOCHS = 200;   % MLP training epochs                       (tune)

%% ── Step 1: Load & Normalise Training Data ───────────────────────────────────
fprintf('Loading training data from "%s"...\n', TRAIN_DATA_PATH);
[X, y]             = load_data(TRAIN_DATA_PATH, IMG_SIZE);
[X, mu, sigma]     = normalize_data(X);
fprintf('Loaded %d samples | %d features | %d males | %d females\n\n', ...
    size(X,1), size(X,2), sum(y==0), sum(y==1));

%% ── Step 2: Define the 4 Pipelines ──────────────────────────────────────────
% Each pipeline_fn signature: (X_train, y_train, X_test)
%   returns: y_pred  [N×1 double]  hard predictions (0/1)
%            scores  [N×1 double]  posterior probability of class 1 (for ROC)
%            info    struct  with fields:
%                      .feature_dim  – dimensionality after dim-reduction
%                      .param_count  – total trainable parameters in classifier
%                      .train_time   – seconds to train (DR + classifier)
%                      .infer_time   – seconds per sample at inference

pipeline_names = {'PCA + Bayes', 'LDA + Bayes', 'PCA + MLP', 'LDA + MLP'};
pipeline_fns   = {
    @(Xtr,ytr,Xte) pipeline_pca_bayes(Xtr, ytr, Xte, PCA_DIMS),
    @(Xtr,ytr,Xte) pipeline_lda_bayes(Xtr, ytr, Xte),
    @(Xtr,ytr,Xte) pipeline_pca_mlp  (Xtr, ytr, Xte, PCA_DIMS, MLP_HIDDEN, MLP_LR, MLP_EPOCHS),
    @(Xtr,ytr,Xte) pipeline_lda_mlp  (Xtr, ytr, Xte,           MLP_HIDDEN, MLP_LR, MLP_EPOCHS),
};
N_PIPELINES = numel(pipeline_names);

%% ── Step 3: 5-Fold Cross-Validation ─────────────────────────────────────────
cv_results(1:N_PIPELINES) = deal(struct( ...
    'name','', 'accuracy',0, 'auc',0, ...
    'feature_dim',0, 'param_count',0, 'train_time',0, 'infer_time',0, ...
    'roc_fpr',[], 'roc_tpr',[]));

for i = 1:N_PIPELINES
    fprintf('[%d/%d] 5-fold CV  →  %s\n', i, N_PIPELINES, pipeline_names{i});
    cv_results(i)      = cross_validate(pipeline_fns{i}, X, y, K_FOLDS);
    cv_results(i).name = pipeline_names{i};
    fprintf('  Accuracy: %.4f  AUC: %.4f  Feat-dim: %d  Params: %d\n', ...
        cv_results(i).accuracy, cv_results(i).auc, ...
        cv_results(i).feature_dim, cv_results(i).param_count);
    fprintf('  Train: %.3fs   Infer/sample: %.5fs\n', ...
        cv_results(i).train_time, cv_results(i).infer_time);
end

%% ── Step 4: ROC Curves ───────────────────────────────────────────────────────
figure('Name','ROC – CV', 'Position',[100,100,700,550]);
hold on;
clrs = lines(N_PIPELINES);
leg  = cell(1, N_PIPELINES);
for i = 1:N_PIPELINES
    plot(cv_results(i).roc_fpr, cv_results(i).roc_tpr, ...
        'Color', clrs(i,:), 'LineWidth', 2);
    leg{i} = sprintf('%s (AUC=%.3f)', cv_results(i).name, cv_results(i).auc);
end
plot([0 1],[0 1],'k--','LineWidth',1);
xlabel('False Positive Rate');  ylabel('True Positive Rate');
title('ROC Curves — 5-fold CV on Training Data');
legend(leg, 'Location','southeast');  grid on;
saveas(gcf, fullfile('results','roc_cv.png'));
fprintf('\nROC plot saved → results/roc_cv.png\n');

%% ── Step 5: Summary Table ───────────────────────────────────────────────────
sep = repmat('─', 1, 88);
fprintf('\n%s\n', sep);
fprintf('%-20s %9s %9s %11s %11s %11s %11s\n', ...
    'Method','Accuracy','AUC','Feat-dim','Params','Train(s)','Infer/s(s)');
fprintf('%s\n', sep);
for i = 1:N_PIPELINES
    r = cv_results(i);
    fprintf('%-20s %9.4f %9.4f %11d %11d %11.4f %11.5f\n', ...
        r.name, r.accuracy, r.auc, ...
        r.feature_dim, r.param_count, r.train_time, r.infer_time);
end
fprintf('%s\n', sep);

%% ── Save ─────────────────────────────────────────────────────────────────────
save(fullfile('results','cv_results.mat'), 'cv_results', 'mu', 'sigma', 'IMG_SIZE');
fprintf('CV results saved → results/cv_results.mat\n');
