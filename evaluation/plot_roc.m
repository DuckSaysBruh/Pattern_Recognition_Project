function plot_roc(results,path)
% Plot pooled OOF ROC. Legend AUC corresponds to the plotted curve.
f=figure('Visible','off','Color','w'); hold on; grid on;
labels=cell(numel(results),1);
for i=1:numel(results)
    r=results{i}; plot(r.roc_fpr,r.roc_tpr,'LineWidth',1.8);
    labels{i}=sprintf('%s: pooled AUC=%.4f',strrep(r.method,'_',' + '),r.pooled_auc);
end
plot([0 1],[0 1],'k--','HandleVisibility','off');
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC (OOF for cross-validation; held-out scores for test)');
legend(labels,'Location','southeast','Interpreter','none');
axis square; xlim([0 1]); ylim([0 1]); saveas(f,path); close(f);
end
