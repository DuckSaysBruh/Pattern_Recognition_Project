function [y_pred, scores, info] = pipeline_pca_bayes(X_train, y_train, X_test, n_dims)
% PIPELINE_PCA_BAYES  PCA dimensionality reduction + Bayes classifier.
%   Member 1 (Integration) — glues together:
%     dim_reduction/pca_train.m      (Member 2)
%     dim_reduction/pca_project.m    (Member 2)
%     classifiers/bayes_train.m      (Member 4)
%     classifiers/bayes_predict.m    (Member 4)
%
% INPUT
%   X_train  – [N_tr × D] normalised training feature matrix
%   y_train  – [N_tr × 1] training labels  (0/1)
%   X_test   – [N_te × D] normalised test   feature matrix
%   n_dims   – number of principal components to retain
%
% OUTPUT
%   y_pred  – [N_te × 1] predicted labels (0/1)
%   scores  – [N_te × 1] posterior probability of class 1 (female)
%   info    – struct with fields:
%               .feature_dim  – dimension after PCA
%               .param_count  – Bayes model parameter count
%               .train_time   – seconds for PCA fit + Bayes fit
%               .infer_time   – seconds PER SAMPLE at inference

    %% Training phase
    t_train = tic;
    pca_model  = pca_train(X_train, n_dims);
    X_tr_red   = pca_project(pca_model, X_train);
    clf_model  = bayes_train(X_tr_red, y_train);
    info.train_time  = toc(t_train);
    info.feature_dim = size(X_tr_red, 2);
    info.param_count = clf_model.param_count;

    %% Inference phase
    t_infer = tic;
    X_te_red        = pca_project(pca_model, X_test);
    [y_pred, scores] = bayes_predict(clf_model, X_te_red);
    info.infer_time = toc(t_infer) / max(size(X_test,1), 1);
end
