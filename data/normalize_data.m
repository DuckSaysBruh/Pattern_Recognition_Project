function [X_norm, mu, sigma] = normalize_data(X, mu, sigma)
% NORMALIZE_DATA  Zero-mean, unit-variance feature normalisation.
%   Member 1 (Data Pipeline)
%
% TRAINING CALL (compute stats from data)
%   [X_norm, mu, sigma] = normalize_data(X)
%
% INFERENCE CALL (apply saved training stats)
%   [X_norm, ~, ~] = normalize_data(X, mu, sigma)
%
% INPUT
%   X      – [N × D] raw feature matrix
%   mu     – (optional) [1 × D] mean vector from training set
%   sigma  – (optional) [1 × D] std  vector from training set
%
% OUTPUT
%   X_norm – [N × D] normalised feature matrix
%   mu     – [1 × D] column means  (equals input mu when supplied)
%   sigma  – [1 × D] column stds   (equals input sigma when supplied)

    if nargin == 1
        mu    = mean(X, 1);
        sigma = std(X,  0, 1);
    end

    % Avoid division by zero for constant (zero-variance) features
    sigma(sigma == 0) = 1;

    X_norm = (X - mu) ./ sigma;
end
