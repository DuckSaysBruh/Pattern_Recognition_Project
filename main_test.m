%% main_test.m  –  Member 1 (Integration / Project Lead)
% Gender Recognition Grand Challenge  |  CVR Lab
%
% DESCRIPTION
%   Trains each pipeline on the FULL training set, then evaluates on the
%   held-out test set.  Reports accuracy, ROC, AUC, and inference time.
%
% USAGE
%   Run from the project root AFTER main_train.m (uses saved mu/sigma):
%     >> main_test
%
% OUTPUT
%   results/roc_test.png      – ROC curves on test set
%   results/test_results.mat  – struct array with all test metrics

clear; clc; close all;
addpath(genpath(pwd));
if ~exist('results','dir'), mkdir('results'); end

%% ── Configuration ────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = 'train';
TEST_DATA_PATH  = 'test';
IMG_SIZE        = [64, 64];

PCA_DIMS   = 50;
MLP_HIDDEN = 64;
MLP_LR     = 0.01;
MLP_EPOCHS = 200;

%% ── Step 1: Load Data ────────────────────────────────────────────────────────
if ~exist(TEST_DATA_PATH, 'dir')
    error(['main_test: test folder "%s" not found.\n' ...
           'Place the test dataset images there first, then re-run.'], ...
          TEST_DATA_PATH);
end

fprintf('Loading training data...\n');
[X_train, y_train]        = load_data(TRAIN_DATA_PATH, IMG_SIZE);
[X_train, mu, sigma]      = normalize_data(X_train);

fprintf('Loading test data...\n');
[X_test, y_test]          = load_data(TEST_DATA_PATH, IMG_SIZE);
[X_test, ~, ~]            = normalize_data(X_test, mu, sigma);  % use training stats

fprintf('Train: %d samples | Test: %d samples\n', size(X_train,1), size(X_test,1));

%% ── Step 2: Run All 4 Pipelines (Full Train → Test) ─────────────────────────
pipeline_names = {'PCA + Bayes', 'LDA + Bayes', 'PCA + MLP', 'LDA + MLP'};
pipeline_fns   = {
    @(Xtr,ytr,Xte) pipeline_pca_bayes(Xtr, ytr, Xte, PCA_DIMS),
    @(Xtr,ytr,Xte) pipeline_lda_bayes(Xtr, ytr, Xte),
    @(Xtr,ytr,Xte) pipeline_pca_mlp  (Xtr, ytr, Xte, PCA_DIMS, MLP_HIDDEN, MLP_LR, MLP_EPOCHS),
    @(Xtr,ytr,Xte) pipeline_lda_mlp  (Xtr, ytr, Xte,           MLP_HIDDEN, MLP_LR, MLP_EPOCHS),
};
N_PIPELINES = numel(pipeline_names);

test_results(1:N_PIPELINES) = deal(struct( ...
    'name','', 'accuracy',0, 'auc',0, 'infer_time',0, ...
    'roc_fpr',[], 'roc_tpr',[]));

for i = 1:N_PIPELINES
    fprintf('\n[%d/%d] Testing:  %s\n', i, N_PIPELINES, pipeline_names{i});
    [y_pred, scores, info] = pipeline_fns{i}(X_train, y_train, X_test);

    acc        = compute_accuracy(y_test, y_pred);
    [fpr, tpr] = compute_roc(y_test, scores);
    auc        = compute_auc(fpr, tpr);

    test_results(i).name       = pipeline_names{i};
    test_results(i).accuracy   = acc;
    test_results(i).auc        = auc;
    test_results(i).infer_time = info.infer_time;
    test_results(i).roc_fpr    = fpr;
    test_results(i).roc_tpr    = tpr;

    fprintf('  Accuracy: %.4f  AUC: %.4f  Infer/sample: %.5fs\n', ...
        acc, auc, info.infer_time);
end

%% ── Step 3: ROC Curves ───────────────────────────────────────────────────────
figure('Name','ROC – Test Set', 'Position',[100,100,700,550]);
hold on;
clrs = lines(N_PIPELINES);
leg  = cell(1, N_PIPELINES);
for i = 1:N_PIPELINES
    plot(test_results(i).roc_fpr, test_results(i).roc_tpr, ...
        'Color', clrs(i,:), 'LineWidth', 2);
    leg{i} = sprintf('%s (AUC=%.3f)', test_results(i).name, test_results(i).auc);
end
plot([0 1],[0 1],'k--','LineWidth',1);
xlabel('False Positive Rate');  ylabel('True Positive Rate');
title('ROC Curves — Test Set');
legend(leg, 'Location','southeast');  grid on;
saveas(gcf, fullfile('results','roc_test.png'));
fprintf('\nROC plot saved → results/roc_test.png\n');

%% ── Step 4: Summary Table ───────────────────────────────────────────────────
sep = repmat('─', 1, 58);
fprintf('\n%s\n', sep);
fprintf('%-20s %10s %10s %11s\n', 'Method','Accuracy','AUC','Infer/s(s)');
fprintf('%s\n', sep);
for i = 1:N_PIPELINES
    r = test_results(i);
    fprintf('%-20s %10.4f %10.4f %11.5f\n', r.name, r.accuracy, r.auc, r.infer_time);
end
fprintf('%s\n', sep);

%% ── Save ─────────────────────────────────────────────────────────────────────
save(fullfile('results','test_results.mat'), 'test_results');
fprintf('Test results saved → results/test_results.mat\n');
