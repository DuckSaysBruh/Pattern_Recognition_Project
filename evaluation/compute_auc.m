function auc = compute_auc(fpr, tpr)
% COMPUTE_AUC  Compute the Area Under the ROC Curve (AUC) via the
%              trapezoidal rule.
%   Member 6 (Evaluation Framework)
%
% ALGORITHM OUTLINE
%   AUC = integral of TPR w.r.t. FPR
%       ≈ sum_i  0.5 * |fpr(i+1) - fpr(i)| * (tpr(i+1) + tpr(i))
%       = trapz(fpr, tpr)   in MATLAB
%
%   Ensure fpr is sorted ascending before calling trapz.
%   AUC = 0.5 means random classifier; AUC = 1.0 means perfect.
%
% INPUT
%   fpr  – [M × 1] false positive rates  (from compute_roc)
%   tpr  – [M × 1] true  positive rates  (from compute_roc)
%
% OUTPUT
%   auc  – scalar in [0, 1]
%
% CALLED BY
%   evaluation/cross_validate.m
%   main_test.m

    % ── TODO: Implement AUC computation ──────────────────────────────────────
    %
    % % Ensure sorted by ascending FPR
    % [fpr_s, idx] = sort(fpr(:));
    % tpr_s        = tpr(idx);
    % auc          = trapz(fpr_s, tpr_s);

    error('compute_auc: TODO — Member 6 must implement this function.');
end
