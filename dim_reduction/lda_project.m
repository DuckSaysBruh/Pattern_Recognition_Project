function X_red = lda_project(lda_model, X)
% LDA_PROJECT  Project data onto the LDA subspace learned by lda_train.
%   Member 3 (LDA Specialist)
%
% ALGORITHM OUTLINE
%   Multiply X by the projection matrix W:
%     X_red = X * W
%   (No mean subtraction needed here — the scatter matrices already encode
%    class structure; mean subtraction was handled during normalise_data.)
%
% INPUT
%   lda_model – struct returned by lda_train (must contain .W)
%   X         – [N × D] feature matrix (training or test)
%
% OUTPUT
%   X_red     – [N × n_dims] discriminant-projected feature matrix
%               For binary classification, n_dims = 1 (scalar projection).
%
% CALLED BY
%   pipelines/pipeline_lda_bayes.m
%   pipelines/pipeline_lda_mlp.m

    % ── TODO: Implement LDA projection ───────────────────────────────────────
    %
    % X_red = X * lda_model.W;   % [N×D] * [D×n_dims] = [N×n_dims]

    error('lda_project: TODO — Member 3 must implement this function.');
end
