function [Z,state] = normalize_data(X,state)
% Fit: normalize_data(X,cfg.normalize). Apply: normalize_data(X,model.normalization).
if ~isfield(state,'mu')
    if strcmp(state.mode,'zscore')
        state.mu=mean(X,1); state.sigma=std(X,0,1);
        state.sigma(state.sigma<state.min_std)=1;
    elseif strcmp(state.mode,'none')
        state.mu=zeros(1,size(X,2)); state.sigma=ones(1,size(X,2));
    else
        error('Unknown normalization mode.');
    end
end
Z=bsxfun(@rdivide,bsxfun(@minus,X,state.mu),state.sigma);
assert(all(isfinite(Z(:))),'Nonfinite normalized data.');
end
