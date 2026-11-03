function X_red = pca_project(pca_model, X)
% PCA_PROJECT  Project data onto the PCA subspace learned by pca_train.
%   Member 2 (PCA Specialist)
%
% ALGORITHM OUTLINE
%   1. Centre X using the mean vector stored in pca_model.mean_vec.
%   2. Multiply by the projection matrix W:
%        X_red = X_centred * W
%
% INPUT
%   pca_model – struct returned by pca_train (must contain .W, .mean_vec)
%   X         – [N × D] feature matrix (training or test)
%
% OUTPUT
%   X_red     – [N × n_dims] projected (reduced) feature matrix
%
% CALLED BY
%   pipelines/pipeline_pca_bayes.m
%   pipelines/pipeline_pca_mlp.m

    % ── TODO: Implement PCA projection ───────────────────────────────────────
    %
    % X_c   = X - pca_model.mean_vec;   % centre with TRAINING mean
    % X_red = X_c * pca_model.W;        % project: [N×D] * [D×n_dims] = [N×n_dims]

    error('pca_project: TODO — Member 2 must implement this function.');
end
