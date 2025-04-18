%% Demo for the proposed PLRU algorithm
% Run the demo below section:
%

clc;
clear;

% cnae-9, connectionist_bench, gait_classification, har_aal, hill_valley,
% libras_movement, lsvt_voice_rehabilitation, madelon, musk,
% optical_recognition, semeion_handwritten_digit, tuandromd
S = load('connectionist_bench');
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
for i = 1 : numel(I)
    idx = kmeans(X(:, I(1:i)), num_Clusters);
    idx(isnan(idx)) = round(mean(idx,"omitnan"));
    YY = grp2idx(Y);
    G = grp2idx(cats(idx));
    nmi_results(i) = NMI(YY, G);
end
disp("Clustering results:");
fprintf("max: %1.8f\t mean: %1.8f\t min: %1.8f\n", max(nmi_results), mean(nmi_results), min(nmi_results));


% Classification analysis
for i = 1 : numel(I)
    acc_results(i) = classify(X(:, I(1:i)), Y, 'KNN');
end
disp("Classification results:");
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

    M = confusionmat(Y, C);
    %confusionchart(M);

    % Total number of instances
    N = sum(M(:));
    
    % Calculate the probabilities of the class labels
    P_y = sum(M, 2) / N;

    % Calculate the entropy of the class labels
    H_y = -sum(P_y .* log2_(P_y));
    
    % Calculate the probabilities of the cluster labels
    P_c = sum(M, 1) / N;

    % Calculate the entropy of the cluster labels
    H_c = -sum(P_c .* log2_(P_c));

    % Calculate the probabilities of class labels within each cluster
    P_yc = M ./ sum(M, 1);
    P_yc(isnan(P_yc)) = 0;

    % Calculate the entropy of class labels within each cluster
    H_yc = sum(P_c .* -sum(P_yc .* log2_(P_yc)));

    % Calculate the mutual information
    I_yc = H_y - H_yc;

    % Calculate the normalized mutual information
    NMI_yc = 2 * I_yc / (H_y + H_c);
end



%% Calculate log2(X) by turning Inf's into 0
function Y = log2_(X)
    Y = log2(X);
    Y(isinf(Y)&~isinf(X)) = 0;
end


