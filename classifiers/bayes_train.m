function model = bayes_train(X, y)
% BAYES_TRAIN  Fit a Gaussian Bayes classifier on labelled training data.
%   Member 4 (Bayes Classifier)
%
% ALGORITHM OUTLINE
%   For each class c in {0, 1}:
%     1. Compute the class prior:
%          P(c) = N_c / N
%     2. Compute the class mean (maximum-likelihood estimate):
%          mu_c = mean of all rows where y == c
%     3. Compute the class covariance (MLE):
%          Sigma_c = cov of all rows where y == c
%          (use MATLAB's cov(), which divides by N-1)
%
%   DECISION RULE at inference:
%     For each test sample x, compute the log-posterior for each class:
%       log P(c | x) ∝ log P(x | c) + log P(c)
%       log P(x | c) = log N(x ; mu_c, Sigma_c)
%                    = -0.5*(x-mu_c)' * inv(Sigma_c) * (x-mu_c)
%                      - 0.5*log|Sigma_c| - (D/2)*log(2*pi)
%     Assign x to the class with the higher log-posterior.
%
% INPUT
%   X  – [N × D] feature matrix (after dimensionality reduction)
%   y  – [N × 1] binary labels  (0 = male, 1 = female)
%
% OUTPUT
%   model – struct with fields:
%     .classes      – [2 × 1]   unique sorted class labels
%     .priors       – [2 × 1]   class priors  P(c)
%     .means        – {2×1}     cell of [1×D] class mean vectors
%     .covs         – {2×1}     cell of [D×D] class covariance matrices
%     .param_count  – scalar    total number of stored parameters
%                               = 2*(D + D^2) + 2  [means + covs + priors]
%
% NOTE — Regularisation
%   Covariance matrices can be singular when D > N_c.
%   Add a small diagonal term:  Sigma_c = Sigma_c + 1e-5 * eye(D)
%
% CALLED BY
%   pipelines/pipeline_pca_bayes.m
%   pipelines/pipeline_lda_bayes.m

    D       = size(X, 2);
    classes = unique(y);
    K       = numel(classes);

    % ── TODO: Implement Gaussian Bayes training ───────────────────────────────
    %
    % model.classes = classes;
    % model.priors  = zeros(K, 1);
    % model.means   = cell(K, 1);
    % model.covs    = cell(K, 1);
    %
    % for i = 1:K
    %     Xi             = X(y == classes(i), :);
    %     Ni             = size(Xi, 1);
    %     model.priors(i) = Ni / size(X, 1);
    %     model.means{i}  = mean(Xi, 1);                  % [1 × D]
    %     Sigma           = cov(Xi);                       % [D × D]
    %     Sigma           = Sigma + 1e-5 * eye(D);         % regularise
    %     model.covs{i}   = Sigma;
    % end
    %
    % model.param_count = K * (D + D^2) + K;   % means + covs + priors

    error('bayes_train: TODO — Member 4 must implement this function.');
end
