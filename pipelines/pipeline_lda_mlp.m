function [y_pred, scores, info] = pipeline_lda_mlp(X_train, y_train, X_test, ...
                                                    hidden_size, lr, epochs)
% PIPELINE_LDA_MLP  LDA dimensionality reduction + MLP classifier.
%   Member 1 (Integration) — glues together:
%     dim_reduction/lda_train.m      (Member 3)
%     dim_reduction/lda_project.m    (Member 3)
%     classifiers/mlp_train.m        (Member 5)
%     classifiers/mlp_predict.m      (Member 5)
%
% INPUT
%   X_train     – [N_tr × D] normalised training feature matrix
%   y_train     – [N_tr × 1] training labels  (0/1)
%   X_test      – [N_te × D] normalised test   feature matrix
%   hidden_size – number of hidden units in MLP
%   lr          – MLP learning rate
%   epochs      – number of MLP training epochs
%
% OUTPUT
%   y_pred  – [N_te × 1] predicted labels (0/1)
%   scores  – [N_te × 1] posterior probability of class 1 (female)
%   info    – struct with fields:
%               .feature_dim  – dimension after LDA
%               .param_count  – MLP total trainable parameters
%               .train_time   – seconds for LDA fit + MLP fit
%               .infer_time   – seconds PER SAMPLE at inference

    %% Training phase
    t_train = tic;
    lda_model  = lda_train(X_train, y_train);
    X_tr_red   = lda_project(lda_model, X_train);
    clf_model  = mlp_train(X_tr_red, y_train, hidden_size, lr, epochs);
    info.train_time  = toc(t_train);
    info.feature_dim = size(X_tr_red, 2);
    info.param_count = clf_model.param_count;

    %% Inference phase
    t_infer = tic;
    X_te_red        = lda_project(lda_model, X_test);
    [y_pred, scores] = mlp_predict(clf_model, X_te_red);
    info.infer_time = toc(t_infer) / max(size(X_test,1), 1);
end
