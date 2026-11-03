function results = cross_validate(pipeline_fn, X, y, k)
% CROSS_VALIDATE  k-fold cross-validation wrapper for any pipeline.
%   Member 6 (Evaluation Framework)
%
% ALGORITHM OUTLINE
%   1. Randomly shuffle the dataset indices (set seed for reproducibility).
%   2. Split indices into k roughly equal folds.
%   3. For fold f = 1 … k:
%        a. Val   indices = fold f
%        b. Train indices = all other folds
%        c. Call: [y_pred, scores, info] = pipeline_fn(X_tr, y_tr, X_val)
%        d. Compute accuracy:          compute_accuracy(y_val, y_pred)
%        e. Compute ROC:               compute_roc(y_val, scores)
%        f. Compute AUC:               compute_auc(fpr, tpr)
%        g. Record info.feature_dim, info.param_count,
%                  info.train_time, info.infer_time
%   4. Average all metrics across k folds.
%   5. Average ROC curve: interpolate each fold's TPR to a common FPR
%      grid [0:0.01:1], then take the mean.
%
% INPUT
%   pipeline_fn – function handle with signature:
%                   (X_train, y_train, X_test) →
%                   (y_pred [N×1], scores [N×1], info struct)
%   X           – [N × D] full feature matrix
%   y           – [N × 1] label vector
%   k           – number of folds (default 5)
%
% OUTPUT
%   results – struct with fields:
%     .accuracy    – scalar  mean accuracy across folds
%     .auc         – scalar  mean AUC across folds
%     .feature_dim – scalar  feature dimension (from last fold's info)
%     .param_count – scalar  parameter count   (from last fold's info)
%     .train_time  – scalar  mean training time across folds
%     .infer_time  – scalar  mean per-sample inference time across folds
%     .roc_fpr     – [M×1]  common FPR grid  (e.g. linspace(0,1,101))
%     .roc_tpr     – [M×1]  mean TPR at each FPR point
%
% CALLED BY
%   main_train.m

    if nargin < 4, k = 5; end

    N = size(X, 1);

    % ── TODO: Implement k-fold cross-validation ───────────────────────────────
    %
    % rng(42);   % for reproducibility
    % idx    = randperm(N);
    % folds  = mat2cell(idx, 1, diff(round(linspace(0, N, k+1))));
    %          % or: use  cvpartition(y, 'KFold', k)  for stratified splits
    %
    % fpr_grid = linspace(0, 1, 101)';   % common FPR axis
    %
    % acc_all  = zeros(k, 1);
    % auc_all  = zeros(k, 1);
    % tpr_all  = zeros(101, k);
    % traint   = zeros(k, 1);
    % infert   = zeros(k, 1);
    % fdim     = 0;
    % pcnt     = 0;
    %
    % for f = 1:k
    %     val_idx   = folds{f};
    %     train_idx = cell2mat(folds([1:f-1, f+1:k]));
    %
    %     X_tr = X(train_idx, :);   y_tr = y(train_idx);
    %     X_va = X(val_idx,   :);   y_va = y(val_idx);
    %
    %     [y_pred, scores, info] = pipeline_fn(X_tr, y_tr, X_va);
    %
    %     acc_all(f)   = compute_accuracy(y_va, y_pred);
    %     [fpr, tpr]   = compute_roc(y_va, scores);
    %     auc_all(f)   = compute_auc(fpr, tpr);
    %     tpr_all(:,f) = interp1(fpr, tpr, fpr_grid, 'linear', 'extrap');
    %     traint(f)    = info.train_time;
    %     infert(f)    = info.infer_time;
    %     fdim         = info.feature_dim;
    %     pcnt         = info.param_count;
    %
    %     fprintf('  Fold %d/%d  acc=%.4f  AUC=%.4f\n', f, k, acc_all(f), auc_all(f));
    % end
    %
    % results.accuracy    = mean(acc_all);
    % results.auc         = mean(auc_all);
    % results.feature_dim = fdim;
    % results.param_count = pcnt;
    % results.train_time  = mean(traint);
    % results.infer_time  = mean(infert);
    % results.roc_fpr     = fpr_grid;
    % results.roc_tpr     = mean(tpr_all, 2);

    error('cross_validate: TODO — Member 6 must implement this function.');
end
