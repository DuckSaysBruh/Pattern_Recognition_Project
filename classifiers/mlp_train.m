function model = mlp_train(X, y, hidden_size, lr, epochs)
% MLP_TRAIN  Train a single-hidden-layer MLP via mini-batch backpropagation.
%   Member 5 (MLP Classifier)
%
% NETWORK ARCHITECTURE
%   Input layer  : D units  (one per feature after dim-reduction)
%   Hidden layer : H units  (hidden_size), activation = sigmoid or ReLU
%   Output layer : 1 unit,  activation = sigmoid  → P(y=1 | x)
%
%   Forward pass:
%     z1     = X * W1 + b1         [N × H]
%     a1     = sigmoid(z1)         [N × H]
%     z2     = a1 * W2 + b2        [N × 1]
%     output = sigmoid(z2)         [N × 1]   ← P(y=1)
%
%   Loss: Binary Cross-Entropy
%     L = -mean( y.*log(output) + (1-y).*log(1-output) )
%
%   Backpropagation:
%     delta2 = output - y_col                     [N × 1]
%     dW2    = a1' * delta2 / N                   [H × 1]
%     db2    = mean(delta2, 1)                    [1 × 1]
%     delta1 = (delta2 * W2') .* sigmoid_grad(z1) [N × H]
%     dW1    = X' * delta1 / N                    [D × H]
%     db1    = mean(delta1, 1)                    [1 × H]
%
%   Weight update (SGD):
%     W1 = W1 - lr * dW1;   b1 = b1 - lr * db1
%     W2 = W2 - lr * dW2;   b2 = b2 - lr * db2
%
% INPUT
%   X           – [N × D] training feature matrix (after dim-reduction)
%   y           – [N × 1] binary labels  (0/1)
%   hidden_size – number of hidden units H
%   lr          – learning rate (e.g. 0.01)
%   epochs      – number of full-data passes
%
% OUTPUT
%   model – struct with fields:
%     .W1          – [D × H]   input→hidden weights
%     .b1          – [1 × H]   hidden biases
%     .W2          – [H × 1]   hidden→output weights
%     .b2          – [1 × 1]   output bias
%     .hidden_size – scalar    H
%     .param_count – scalar    D*H + H + H*1 + 1
%     .loss_history– [epochs×1] training loss per epoch (for diagnostics)
%
% TIP — Weight initialisation
%   Use Xavier / Glorot initialisation:
%     W1 = randn(D, H) * sqrt(2 / (D + H));
%     W2 = randn(H, 1) * sqrt(2 / (H + 1));
%
% CALLED BY
%   pipelines/pipeline_pca_mlp.m
%   pipelines/pipeline_lda_mlp.m

    [N, D] = size(X);
    H      = hidden_size;

    % ── TODO: Implement MLP training ─────────────────────────────────────────
    %
    % % Initialise weights (Xavier)
    % W1 = randn(D, H) * sqrt(2 / (D + H));
    % b1 = zeros(1, H);
    % W2 = randn(H, 1) * sqrt(2 / (H + 1));
    % b2 = 0;
    %
    % y_col       = y(:);           % ensure column vector
    % loss_history = zeros(epochs, 1);
    %
    % for ep = 1:epochs
    %     % Forward pass
    %     z1     = X * W1 + b1;              % [N × H]
    %     a1     = sigmoid_fn(z1);           % [N × H]
    %     z2     = a1 * W2 + b2;             % [N × 1]
    %     output = sigmoid_fn(z2);           % [N × 1]
    %
    %     % Loss (binary cross-entropy)
    %     eps_clip = 1e-12;
    %     loss = -mean( y_col.*log(output+eps_clip) + (1-y_col).*log(1-output+eps_clip) );
    %     loss_history(ep) = loss;
    %
    %     % Backward pass
    %     delta2 = output - y_col;                              % [N × 1]
    %     dW2    = (a1' * delta2) / N;                          % [H × 1]
    %     db2    = mean(delta2);
    %     delta1 = (delta2 * W2') .* sigmoid_grad_fn(z1);      % [N × H]
    %     dW1    = (X' * delta1) / N;                           % [D × H]
    %     db1    = mean(delta1, 1);
    %
    %     % Update
    %     W1 = W1 - lr * dW1;   b1 = b1 - lr * db1;
    %     W2 = W2 - lr * dW2;   b2 = b2 - lr * db2;
    %
    %     if mod(ep, max(1, floor(epochs/10))) == 0
    %         fprintf('  MLP epoch %d/%d  loss=%.5f\n', ep, epochs, loss);
    %     end
    % end
    %
    % model.W1           = W1;
    % model.b1           = b1;
    % model.W2           = W2;
    % model.b2           = b2;
    % model.hidden_size  = H;
    % model.param_count  = D*H + H + H + 1;
    % model.loss_history = loss_history;

    error('mlp_train: TODO — Member 5 must implement this function.');
end

% ── Helper functions (define in same file or as separate .m files) ────────────

function a = sigmoid_fn(z)
    a = 1 ./ (1 + exp(-z));
end

function g = sigmoid_grad_fn(z)
    s = sigmoid_fn(z);
    g = s .* (1 - s);
end
