function [fpr, tpr] = compute_roc(y_true, scores)
% COMPUTE_ROC  Compute the ROC curve (FPR, TPR) from binary scores.
%   Member 6 (Evaluation Framework)
%
% ALGORITHM OUTLINE
%   1. Obtain all unique score values as classification thresholds.
%      Sort thresholds in DESCENDING order (start from classifying everything
%      as positive, then progressively increase threshold).
%   2. For each threshold t:
%        y_hat = (scores >= t)
%        TP    = sum( (y_hat==1) & (y_true==1) )
%        FP    = sum( (y_hat==1) & (y_true==0) )
%        TN    = sum( (y_hat==0) & (y_true==0) )
%        FN    = sum( (y_hat==0) & (y_true==1) )
%        TPR(t) = TP / (TP + FN)   ← sensitivity / recall
%        FPR(t) = FP / (FP + TN)   ← 1 - specificity
%   3. Return fpr and tpr as column vectors, including (0,0) and (1,1)
%      anchor points.
%
%   TIP: MATLAB's perfcurve() can do this if you prefer:
%     [fpr, tpr] = perfcurve(y_true, scores, 1);
%     But implement it from scratch for the report methodology section.
%
% INPUT
%   y_true  – [N × 1] ground-truth binary labels  (0/1)
%   scores  – [N × 1] classifier scores / probabilities (higher → class 1)
%
% OUTPUT
%   fpr  – [M × 1] false positive rates (sorted ascending)
%   tpr  – [M × 1] true  positive rates (sorted ascending, same order as fpr)
%
% CALLED BY
%   evaluation/cross_validate.m
%   main_test.m

    % ── TODO: Implement ROC curve computation ─────────────────────────────────
    %
    % thresholds = sort(unique(scores), 'descend');
    % thresholds = [thresholds(1)+1; thresholds; thresholds(end)-1]; % anchors
    %
    % P = sum(y_true == 1);
    % N = sum(y_true == 0);
    %
    % fpr = zeros(numel(thresholds), 1);
    % tpr = zeros(numel(thresholds), 1);
    %
    % for i = 1:numel(thresholds)
    %     y_hat  = double(scores >= thresholds(i));
    %     TP = sum((y_hat==1) & (y_true==1));
    %     FP = sum((y_hat==1) & (y_true==0));
    %     tpr(i) = TP / P;
    %     fpr(i) = FP / N;
    % end
    %
    % [fpr, order] = sort(fpr);
    % tpr = tpr(order);

    error('compute_roc: TODO — Member 6 must implement this function.');
end
