function [X, y] = load_data(data_path, img_size)
% LOAD_DATA  Load face images from a dataset folder into a feature matrix.
%   Member 1 (Data Pipeline)
%
% Naming convention expected:
%   woman_<id>_<n>.png  →  label 1 (female)
%   man_<id>_<n>.png    →  label 0 (male)
%
% INPUT
%   data_path  – path to folder containing .png images (string)
%   img_size   – [H W] target resize dimensions, default [64 64]
%
% OUTPUT
%   X  – [N × D] double matrix, each row is a flattened, greyscale image
%   y  – [N × 1] double label vector  (0 = male, 1 = female)
%
% EXAMPLE
%   [X, y] = load_data('train', [64 64]);

    if nargin < 2
        img_size = [64, 64];
    end

    H = img_size(1);
    W = img_size(2);
    D = H * W;

    files = dir(fullfile(data_path, '*.png'));
    if isempty(files)
        error('load_data: no PNG files found in "%s".', data_path);
    end

    N = numel(files);
    X = zeros(N, D, 'double');
    y = zeros(N, 1, 'double');

    for i = 1:N
        fpath = fullfile(data_path, files(i).name);
        img   = imread(fpath);

        % Convert to greyscale if RGB
        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        % Resize to target dimensions
        img = imresize(img, [H, W]);

        % Flatten to row vector and cast to double
        X(i, :) = double(img(:))';

        % Label from filename
        if contains(files(i).name, 'woman')
            y(i) = 1;
        else
            y(i) = 0;
        end
    end

    fprintf('[load_data] %d images loaded from "%s" | males: %d | females: %d\n', ...
        N, data_path, sum(y==0), sum(y==1));
end
