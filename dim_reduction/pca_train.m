function pca_model = pca_train(X, n_dims)
% PCA_TRAIN  Compute a PCA model from training data.
%   Member 2 (PCA Specialist)
%
% ALGORITHM OUTLINE
%   1. Compute the mean face vector (mean of all rows in X).
%   2. Centre the data by subtracting the mean.
%   3. Compute the covariance matrix  C = (X_centred' * X_centred) / (N-1)
%      TIP: for D >> N use the "compact trick":
%           eigen-decompose X_centred * X_centred' (N×N) instead of (D×D),
%           then recover the D-dimensional eigenvectors.
%   4. Perform eigendecomposition of C to get eigenvalues and eigenvectors.
%   5. Sort eigenvectors by eigenvalue in DESCENDING order.
%   6. Retain the top n_dims eigenvectors as the projection matrix W.
%   7. (Optional) Record the explained variance ratio for analysis.
%
% INPUT
%   X       – [N × D] centred (normalised) training feature matrix
%   n_dims  – number of principal components to retain
%
% OUTPUT
%   pca_model – struct with fields:
%     .W          – [D × n_dims]  projection matrix (principal components)
%     .mean_vec   – [1 × D]       mean of training data (for centring at test)
%     .eigenvalues– [n_dims × 1]  retained eigenvalues (descending)
%     .var_ratio  – [n_dims × 1]  explained variance ratio per component
%     .n_dims     – scalar        number of retained components
%
% CALLED BY
%   pipelines/pipeline_pca_bayes.m
%   pipelines/pipeline_pca_mlp.m

    % ── Validate inputs ──────────────────────────────────────────────────────
    [N, D] = size(X);
    if n_dims > min(N-1, D)
        warning('pca_train: n_dims (%d) exceeds max (%d). Clamping.', ...
            n_dims, min(N-1,D));
        n_dims = min(N-1, D);
    end

    % ── TODO: Implement PCA ───────────────────────────────────────────────────
    %
    % Step 1 – Compute mean vector
    %   mean_vec = mean(X, 1);   % [1 × D]
    %
    % Step 2 – Centre the data
    %   X_c = X - mean_vec;      % [N × D]
    %
    % Step 3 – Covariance / scatter matrix
    %   Option A (D×D, feasible when D is manageable):
    %     C = (X_c' * X_c) / (N - 1);
    %     [V, Lam] = eig(C);
    %
    %   Option B (compact trick, recommended when D > N):
    %     M   = (X_c * X_c') / (N - 1);   % [N × N]
    %     [U, Lam] = eig(M);
    %     V   = X_c' * U;                  % recover D-dim eigenvectors
    %     % Normalise each column of V to unit length (R2016b-compatible):
    %     V   = V ./ sqrt(sum(V.^2, 1));   % do NOT use vecnorm — requires R2017b+
    %
    % Step 4 – Sort by descending eigenvalue
    %   lam       = diag(Lam);
    %   [lam_s, idx] = sort(lam, 'descend');
    %   V_sorted  = V(:, idx);
    %
    % Step 5 – Retain top n_dims
    %   W         = V_sorted(:, 1:n_dims);
    %   var_ratio = lam_s(1:n_dims) / sum(abs(lam));
    %
    % Step 6 – Package model
    %   pca_model.W           = W;
    %   pca_model.mean_vec    = mean_vec;
    %   pca_model.eigenvalues = lam_s(1:n_dims);
    %   pca_model.var_ratio   = var_ratio;
    %   pca_model.n_dims      = n_dims;

    error('pca_train: TODO — Member 2 must implement this function.');
end
