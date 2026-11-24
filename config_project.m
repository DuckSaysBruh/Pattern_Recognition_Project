function cfg = config_project()
% Central configuration. Training/test images are never committed.
cfg.image_size = [640 640];
cfg.resize = false;
cfg.train_path = '';
cfg.output_root = fullfile(fileparts(mfilename('fullpath')), 'results');
cfg.methods = {'pca_bayes'};
cfg.cv.num_folds = 5;
cfg.cv.seed = 42;
cfg.cv.mode = 'person_grouped'; % or 'image_stratified'
cfg.normalize.mode = 'zscore'; % or 'none'; [0,1] scaling still applies
cfg.normalize.min_std = 1e-8;
cfg.pca.n_dims = 20;
cfg.pca.relative_tolerance = 1e-10;
cfg.lda.ridge_relative = 0.1; % x trace(Sw)/min(D,N-2); tuned by person-grouped CV
cfg.lda.ridge_absolute = 1e-10;
cfg.bayes.ridge_absolute = 1e-5;
cfg.mlp.hidden_size = 64;
cfg.mlp.learning_rate = 0.01;
cfg.mlp.epochs = 1000;
cfg.mlp.seed = 42;
cfg.mlp.l2 = 0;
cfg.timing.repeats = 5;
end
