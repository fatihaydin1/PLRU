%% Demo for the proposed PLRU algorithm
% Run the demo below section:
%

clc;
clear;

% cnae-9, connectionist_bench, gait_classification, har_aal, hill_valley,
% libras_movement, lsvt_voice_rehabilitation, madelon, musk,
% optical_recognition, semeion_handwritten_digit, tuandromd
S = load('cnae-9');
X = S.X;
Y = categorical(S.Y);

% Get indices of features according to parameters (alpha and optimization parameters)
%
% The options for the optimization algorithm are as follows:
% [~, I] = PLRU(X, 0.5); ===> for default settings
% [~, I] = PLRU(X, 0.5, 'ds-bfgs'); ===> for a detailed search with BFGS
% [~, I] = PLRU(X, 0.5, 'ds-lbfgs'); ===> for a detailed search with L-BFGS
% [~, I] = PLRU(X, 0.5, 'gh'); ===> for the fastest solution with precomputed gradient and Hessian matrices.
% 
tic;
[B, I] = PLRU(X, 0.5);
toc;

% for i = 1 : length(B)
%     fprintf("%1.4f\t", B(i));
% end


d = size(X, 2);
acc_results = zeros(d, 1);
nmi_results = zeros(d, 1);

% Clustering analysis
cats = categories(Y);
num_Clusters = numel(cats);
disp("Clustering results:");
for i = 1 : numel(I)
    idx = kmeans(X(:, I(1:i)), num_Clusters);
    idx(isnan(idx)) = round(mean(idx,"omitnan"));
    YY = grp2idx(Y);
    G = grp2idx(cats(idx));
    nmi_results(i) = NMI(YY, G);
end
fprintf("max: %1.8f\t mean: %1.8f\t min: %1.8f\n", max(nmi_results), mean(nmi_results), min(nmi_results));


% Classification analysis
disp("Classification results:");
for i = 1 : numel(I)
    acc_results(i) = classify(X(:, I(1:i)), Y, 'KNN');
end
fprintf("max: %1.8f\t mean: %1.8f\t min: %1.8f\n", max(acc_results), mean(acc_results), min(acc_results));



%% Classification
function [ ACC ] = classify( X, Y, classifier )

    predictions = repmat(Y, 1, 2);
    indices = crossvalind('Kfold', Y, 10);
        
    for i = 1:10
        %fprintf('%d',i);
        test = (indices == i);
        train = ~test;
                
        trainY = Y(train,:);
        trainX = X(train,:);
        testX = X(test,:);
        
        switch classifier
            case 'KNN'
                Mdl = fitcknn(trainX, trainY, 'NumNeighbors', 1);
            case 'CART'
                Mdl = fitctree(trainX, trainY);
                predictions(test, 2) = predict(Mdl, testX);
            case 'NB'
                % "normal", "mn", "kernel", "mvmn".
                Mdl = fitcnb(trainX, trainY, 'DistributionNames', 'normal');
                predictions(test, 2) = predict(Mdl, testX);
            case 'SVM'
                % "linear", "gaussian", "rbf", "polynomial"
                t = templateSVM('Standardize', true, 'KernelFunction', 'linear');
                Mdl = fitcecoc(trainX, trainY, 'Learners', t);
                predictions(test, 2) = predict(Mdl, testX);
        end
        predictions(test, 2) = predict(Mdl, testX);
    end
    ACC = sum(predictions(:,1) == predictions(:,2))*1/length(Y);
end



%% Normalized Mutual Information
function [ NMI_yc ] = NMI(Y, C)
    % Ensure column vectors
    Y = Y(:);
    C = C(:);

    % Check length consistency
    assert(length(Y) == length(C), 'Inputs Y and C must be of the same length');

    n = length(Y);

    % Unique labels and clusters
    unique_Y = unique(Y);
    unique_C = unique(C);
    c1 = length(unique_Y);
    c2 = length(unique_C);

    % One-hot encoding matrices
    My = double(repmat(Y, 1, c1) == repmat(unique_Y', n, 1));  % n x c1
    Mc = double(repmat(C, 1, c2) == repmat(unique_C', n, 1));  % n x c2

    % Marginal distributions P(Y) and P(C)
    P_y = sum(My, 1) / n;   % 1 x c1
    P_c = sum(Mc, 1) / n;   % 1 x c2

    % Entropy function: handles 0 * log(0) = 0 convention
    entropy = @(P) -sum(P(P > 0) .* log2(P(P > 0)));

    % Entropies H(Y) and H(C)
    H_y = entropy(P_y);
    H_c = entropy(P_c);

    % Joint distribution P(Y,C)
    P_joint = (My' * Mc) / n;   % c1 x c2
    P_outer = P_y' * P_c;       % outer product of marginals

    % Mask to avoid log2(0)
    mask = P_joint > 0;

    % Mutual Information I(Y;C)
    MI = sum(P_joint(mask) .* log2(P_joint(mask) ./ P_outer(mask)));

    % Normalized Mutual Information (geometric mean version)
    NMI_yc = MI / sqrt(H_y * H_c);
end


