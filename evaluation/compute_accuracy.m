function acc=compute_accuracy(y,pred)
assert(numel(y)==numel(pred) && ~isempty(y),'Invalid prediction lengths.');
acc=mean(y(:)==pred(:));
end
