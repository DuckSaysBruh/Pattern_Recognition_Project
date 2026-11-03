function [y_pred, scores] = bayes_predict(model, X)
% BAYES_PREDICT  Classify samples using a trained Gaussian Bayes model.
%   Member 4 (Bayes Classifier)
%
% ALGORITHM OUTLINE
%   For each test sample x (row of X):
%     1. For each class c, compute the log-likelihood:
%          log_lik_c = log N(x ; mu_c, Sigma_c)
%                    = -0.5 * (x - mu_c) * inv(Sigma_c) * (x - mu_c)'
%                      - 0.5 * log(det(Sigma_c))
%                      - (D/2) * log(2*pi)
%        TIP: use  log(det(A)) = sum(log(diag(chol(A)))) * 2  for stability.
%        TIP: use  A \ b  instead of  inv(A) * b  for numerical stability.
%     2. Log-posterior:
%          log_post_c = log_lik_c + log(prior_c)
%     3. Hard prediction:
%          y_pred = argmax_c  log_post_c
%     4. Soft score for ROC (posterior probability of class 1):
%          Convert log-posteriors to probabilities via softmax:
%          p = exp(log_post) / sum(exp(log_post))   [handle underflow!]
%          scores = p(:, index_of_class_1)
%
% INPUT
%   model   – struct returned by bayes_train
%   X       – [N × D] feature matrix (after dimensionality reduction)
%
% OUTPUT
%   y_pred  – [N × 1] predicted class labels (values from model.classes)
%   scores  – [N × 1] posterior probability of the POSITIVE class (label=1)
%             (used by compute_roc / compute_auc)
%
% CALLED BY
%   pipelines/pipeline_pca_bayes.m
%   pipelines/pipeline_lda_bayes.m

    % ── TODO: Implement Gaussian Bayes inference ──────────────────────────────
    %
    % [N, D] = size(X);
    % K      = numel(model.classes);
    % log_post = zeros(N, K);
    %
    % for i = 1:K
    %     mu_i    = model.means{i};          % [1 × D]
    %     Sig_i   = model.covs{i};           % [D × D]
    %     diff    = X - mu_i;                % [N × D]
    %
    %     % Log-determinant (stable)
    %     L       = chol(Sig_i);
    %     log_det = 2 * sum(log(diag(L)));
    %
    %     % Mahalanobis distance squared for each sample
    %     mah2 = sum((diff / Sig_i) .* diff, 2);   % [N × 1]
    %
    %     log_post(:,i) = -0.5 * mah2 ...
    %                     - 0.5 * log_det ...
    %                     - (D/2) * log(2*pi) ...
    %                     + log(model.priors(i));
    % end
    %
    % % Hard labels
    % [~, idx] = max(log_post, [], 2);
    % y_pred   = model.classes(idx);
    %
    % % Soft scores: normalised posterior probability of class 1 (index 2)
    % log_post_shifted = log_post - max(log_post, [], 2);   % numerical stability
    % post  = exp(log_post_shifted) ./ sum(exp(log_post_shifted), 2);
    % pos_idx = find(model.classes == 1);
    % scores  = post(:, pos_idx);

    error('bayes_predict: TODO — Member 4 must implement this function.');
end
