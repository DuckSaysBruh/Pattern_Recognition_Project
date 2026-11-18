function Z=pca_project(model,X)
Z=bsxfun(@minus,X,model.mean_vec)*model.W;
end
