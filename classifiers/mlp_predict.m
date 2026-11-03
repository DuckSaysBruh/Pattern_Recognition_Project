function [y_pred, scores] = mlp_predict(model, X)
% MLP_PREDICT  Run inference with a trained MLP model.
%   Member 5 (MLP Classifier)
%
% ALGORITHM OUTLINE
%   Forward pass only:
%     z1     = X * model.W1 + model.b1    [N × H]
%     a1     = sigmoid(z1)                [N × H]
%     z2     = a1 * model.W2 + model.b2   [N × 1]
%     scores = sigmoid(z2)                [N × 1]  ← P(y=1 | x)
%
%   Threshold at 0.5 for hard prediction:
%     y_pred = double(scores >= 0.5)
%
% INPUT
%   model  – struct returned by mlp_train
%   X      – [N × D] feature matrix (after dim-reduction)
%
% OUTPUT
%   y_pred  – [N × 1] predicted labels  (0 or 1)
%   scores  – [N × 1] P(y=1 | x)  — used for ROC / AUC
%
% CALLED BY
%   pipelines/pipeline_pca_mlp.m
%   pipelines/pipeline_lda_mlp.m

    % ── TODO: Implement MLP inference ────────────────────────────────────────
    %
    % z1     = X * model.W1 + model.b1;
    % a1     = 1 ./ (1 + exp(-z1));
    % z2     = a1 * model.W2 + model.b2;
    % scores = 1 ./ (1 + exp(-z2));       % [N × 1]
    % y_pred = double(scores >= 0.5);     % [N × 1]

    error('mlp_predict: TODO — Member 5 must implement this function.');
end
