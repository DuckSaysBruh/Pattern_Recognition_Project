function lda_model = lda_train(X, y)
% LDA_TRAIN  Compute an LDA projection from labelled training data.
%   Member 3 (LDA Specialist)
%
% ALGORITHM OUTLINE
%   1. Identify the unique classes and compute per-class statistics
%      (means and counts).
%   2. Compute the WITHIN-CLASS scatter matrix S_W:
%        S_W = sum_c  sum_{x in class c} (x - mu_c)(x - mu_c)'
%   3. Compute the BETWEEN-CLASS scatter matrix S_B:
%        S_B = sum_c  N_c * (mu_c - mu_total)(mu_c - mu_total)'
%   4. Solve the generalised eigenvalue problem:
%        S_B * w = lambda * S_W * w
%        Equivalently: inv(S_W) * S_B * w = lambda * w
%        Use eig(S_B, S_W) or  eig(S_W \ S_B)  in MATLAB.
%   5. Sort eigenvectors by descending eigenvalue.
%   6. For binary classification (K=2) only 1 discriminant direction exists
%      (rank of S_B = K-1 = 1).  Retain all K-1 directions.
%
% INPUT
%   X  – [N × D] normalised training feature matrix
%   y  – [N × 1] integer class labels  (0/1 or any two distinct values)
%
% OUTPUT
%   lda_model – struct with fields:
%     .W           – [D × K-1]  LDA projection matrix
%     .eigenvalues – [K-1 × 1]  sorted generalised eigenvalues
%     .classes     – [K × 1]    unique class labels
%     .n_dims      – scalar     number of discriminant directions (= K-1)
%
% CALLED BY
%   pipelines/pipeline_lda_bayes.m
%   pipelines/pipeline_lda_mlp.m
%
% TIP — Regularisation
%   S_W is often singular or near-singular for high-D data.
%   Add a small ridge: S_W = S_W + 1e-6 * trace(S_W)/D * eye(D)

    % ── TODO: Implement LDA ───────────────────────────────────────────────────
    %
    % classes   = unique(y);
    % K         = numel(classes);
    % [N, D]    = size(X);
    % mu_total  = mean(X, 1);   % [1 × D]
    %
    % S_W = zeros(D, D);
    % S_B = zeros(D, D);
    %
    % for i = 1:K
    %     Xi       = X(y == classes(i), :);          % samples of class i
    %     Ni       = size(Xi, 1);
    %     mu_i     = mean(Xi, 1);                     % [1 × D]
    %     diff_W   = Xi - mu_i;                       % [Ni × D]
    %     S_W      = S_W + diff_W' * diff_W;          % within-class
    %     diff_B   = (mu_i - mu_total)';              % [D × 1]
    %     S_B      = S_B + Ni * (diff_B * diff_B');   % between-class
    % end
    %
    % % Optional regularisation of S_W
    % S_W = S_W + 1e-6 * trace(S_W) / D * eye(D);
    %
    % % Generalised eigendecomposition
    % [V, Lam] = eig(S_B, S_W);
    % lam      = real(diag(Lam));
    % [lam_s, idx] = sort(lam, 'descend');
    % V_sorted = real(V(:, idx));
    %
    % n_dims = K - 1;   % = 1 for binary
    % lda_model.W           = V_sorted(:, 1:n_dims);
    % lda_model.eigenvalues = lam_s(1:n_dims);
    % lda_model.classes     = classes;
    % lda_model.n_dims      = n_dims;

    error('lda_train: TODO — Member 3 must implement this function.');
end
