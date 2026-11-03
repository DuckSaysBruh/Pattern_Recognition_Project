function plot_roc(results_array, title_str, save_path)
% PLOT_ROC  Plot ROC curves for multiple methods on one figure.
%   Member 6 (Evaluation Framework)
%
% ALGORITHM OUTLINE
%   1. Create a new figure.
%   2. For each entry in results_array, plot (roc_fpr, roc_tpr) with a
%      distinct colour and include the AUC in the legend label.
%   3. Draw the random-classifier diagonal (dashed black line).
%   4. Label axes, add title, legend, and grid.
%   5. Save to save_path if provided.
%
% INPUT
%   results_array  – struct array, each element must have:
%                      .name     (string)
%                      .roc_fpr  ([M×1] FPR)
%                      .roc_tpr  ([M×1] TPR)
%                      .auc      (scalar)
%   title_str      – figure title string (e.g. 'ROC – 5-fold CV')
%   save_path      – (optional) file path to save the figure as PNG
%
% CALLED BY
%   main_train.m  (indirectly – main_train.m contains its own inline plot;
%                  use this function if you prefer modular plotting)

    if nargin < 2, title_str = 'ROC Curves';  end
    if nargin < 3, save_path = '';            end

    % ── TODO: Implement ROC plotting ──────────────────────────────────────────
    %
    % figure('Position',[100,100,700,550]);
    % hold on;
    % clrs = lines(numel(results_array));
    % for i = 1:numel(results_array)
    %     r = results_array(i);
    %     plot(r.roc_fpr, r.roc_tpr, 'Color', clrs(i,:), 'LineWidth', 2, ...
    %          'DisplayName', sprintf('%s (AUC=%.3f)', r.name, r.auc));
    % end
    % plot([0 1],[0 1],'k--','LineWidth',1,'DisplayName','Random');
    % xlabel('False Positive Rate');
    % ylabel('True Positive Rate');
    % title(title_str);
    % legend('Location','southeast');
    % grid on;
    % if ~isempty(save_path)
    %     saveas(gcf, save_path);
    %     fprintf('ROC plot saved → %s\n', save_path);
    % end

    error('plot_roc: TODO — Member 6 must implement this function.');
end
