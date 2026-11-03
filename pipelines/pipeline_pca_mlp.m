function [y_pred, scores, info] = pipeline_pca_mlp(X_train, y_train, X_test, ...
                                                    n_dims, hidden_size, lr, epochs)
% PIPELINE_PCA_MLP  PCA dimensionality reduction + MLP classifier.
%   Member 1 (Integration) — glues together:
%     dim_reduction/pca_train.m      (Member 2)
%     dim_reduction/pca_project.m    (Member 2)
%     classifiers/mlp_train.m        (Member 5)
%     classifiers/mlp_predict.m      (Member 5)
%
% INPUT
%   X_train     – [N_tr × D] normalised training feature matrix
%   y_train     – [N_tr × 1] training labels  (0/1)
%   X_test      – [N_te × D] normalised test   feature matrix
%   n_dims      – number of principal components to retain
%   hidden_size – number of hidden units in MLP
%   lr          – MLP learning rate
%   epochs      – number of MLP training epochs
%
% OUTPUT
%   y_pred  – [N_te × 1] predicted labels (0/1)
%   scores  – [N_te × 1] posterior probability of class 1 (female)
%   info    – struct with fields:
%               .feature_dim  – dimension after PCA
%               .param_count  – MLP total trainable parameters
%               .train_time   – seconds for PCA fit + MLP fit
%               .infer_time   – seconds PER SAMPLE at inference

    %% Training phase
    t_train = tic;
    pca_model  = pca_train(X_train, n_dims);
    X_tr_red   = pca_project(pca_model, X_train);
    clf_model  = mlp_train(X_tr_red, y_train, hidden_size, lr, epochs);
    info.train_time  = toc(t_train);
    info.feature_dim = size(X_tr_red, 2);
    info.param_count = clf_model.param_count;

    %% Inference phase
    t_infer = tic;
    X_te_red        = pca_project(pca_model, X_test);
    [y_pred, scores] = mlp_predict(clf_model, X_te_red);
    info.infer_time = toc(t_infer) / max(size(X_test,1), 1);
end
