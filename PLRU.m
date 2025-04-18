%% PLRU (the Preservation of Locality Relation for Unsupervised feature selection)
%
% INPUTS:
% X denotes a data matrix consisting of m rows and d columns
% alpha denotes a slack variable to avoid a trivial solution and it is learnable by the optimization method
% optimization_option (optional) denotes optimization preferences: searching by default settings, searching in detail using BFGS, searching in detail using L-BFGS, by specifying gradients and Hessians, 
% 
%
% OUTPUT:
% B denotes a vector that represents the weighted values of features
% I denotes a vector that represents the indices of features ordered from largest to smallest according to their importance level (weight value)

function [ B, I ] = PLRU( X, alpha, optimization_option )

    if nargin == 2
        optimization_option = 'default';
    end

    if ~(alpha >= 0 && alpha <= 1)
        error("alpha must be between 0 and 1")
    end

    d = size(X, 2);
    X(isnan(X)) = 0;

    % Grammian matrix
    G = X'*X;

    % Normalization process
    N = diag(1./diag(G));
    N(isinf(N)) = 1;
    G = N*G;

    % variables
    x0 = [0.5*ones(1,d) alpha];
    
    epsilon = 1.0e-7;
    % lower bounds
    lb = zeros(1, d+1) + epsilon;
    % upper bounds
    ub = ones(1, d+1) - epsilon;

    solution = solve_by_interiorpoint(G, x0, lb, ub, optimization_option);

    [B, I] = sort(solution(1:end-1), 'descend');
end



%% Solve the optimization problem by the interior point algorithm
function [solution] = solve_by_interiorpoint(G, x0, lb, ub, optimization_option)
    
    function stop = outfun(x, optimValues, state)
        fitness_history = [];
        stop = false;
        if strcmp(state, 'iter')
            fitness_history(end+1) = optimValues.fval;
            %fprintf('İterasyon %d: fval = %.6f\n', optimValues.iteration, optimValues.fval);
            fprintf('%.6f, ', optimValues.fval);
        end
    end

    % The objective function
    if strcmpi(optimization_option, 'gh') % Gradient + Hessian
        objectiveFcn = @(x) objective_func_with_derivatives(x, G);
    else
        objectiveFcn = @(x) norm(diag(x(1:end-1)).^2 * G * diag(x(1:end-1)) - x(end) * G, 'fro');
    end

    % Setting hyperparameters for optimization process
    options = optimoptions(@fmincon, ...
        'Algorithm', 'interior-point', ...
        'BarrierParamUpdate', 'monotone', ...
        'EnableFeasibilityMode', false(), ...
        'HessianApproximation', 'bfgs', ... 
        'SpecifyObjectiveGradient', false, ... 
        'HessianFcn', [], ... 
        'HessianMultiplyFcn', [], ...
        'SubproblemAlgorithm', 'factorization', ...
        'Display', 'off', ...
        'OptimalityTolerance', 1e-6, ...
        'StepTolerance',       1e-10, ...
        'MaxIterations', 1000, ...
        'MaxFunctionEvaluations', 3000);
        %'OutputFcn', @outfun);
        %'Display', 'iter-detailed');
        %'PlotFcn', @optimplotfval);

    if strcmpi(optimization_option, 'gh') % Gradient + Hessian
        disp('Gradient + Hessian')
        options.SpecifyObjectiveGradient = true;
        options.HessianFcn = @(a, lambda) objective_func_with_derivatives(a, G);
        options.OptimalityTolerance = 1e-10;
        options.StepTolerance = 1e-12;
        options.MaxIterations = 3000;
        options.MaxFunctionEvaluations = 10000;
        options.HessianApproximation = 'bfgs';
    elseif strcmpi(optimization_option, 'ds-bfgs') % Detailed Search
        disp('Detailed search with BFGS')
        options.OptimalityTolerance = 1e-10;
        options.StepTolerance = 1e-12;
        options.MaxIterations = 1e+4;
        options.MaxFunctionEvaluations = 1e+5;
        options.HessianApproximation = 'bfgs';
    elseif strcmpi(optimization_option, 'ds-lbfgs') % Detailed Search
        disp('Detailed search with L-BFGS')
        options.OptimalityTolerance = 1e-10;
        options.StepTolerance = 1e-12;
        options.MaxIterations = 1e+4;
        options.MaxFunctionEvaluations = 1e+5;
        options.HessianApproximation = 'lbfgs';
    else
        disp('Default settings')
    end

    % Solve the minimization problem
    [solution, ~] = fmincon(objectiveFcn, x0, [], [], [], [], lb, ub, [], options);
end



%% Objective function with derivatives
% Use this when you pass gradients and Hessians to the interior-point's hyperparameters, calculating them manually
%
 % INPUTS:
 %   a: [d1, ..., dd, alpha]
 %   G: Gram matrix (d x d)

 % OUTPUTS:
 %   fval: scalar objective function value
 %   grad: gradient vector (d+1) x (1)
 %   hess: Hessian matrix (d+1) x (d+1)
function [fval, grad, hess] = objective_func_with_derivatives(a, G)

    d = length(a) - 1;
    di = a(1:d);
    alpha = a(end);

    % Construct diagonal matrices
    D = diag(di);
    D2 = diag(di.^2);

    % Compute M = D^2 G D - alpha G
    M = D2 * G * D - alpha * G;
    fval = norm(M, 'fro')^2;

    % Gradient computation
    grad = zeros(d+1,1);
    for k = 1 : d
        % Partial derivative w.r.t. d_k
        term1 = 0;
        term2 = 0;
        for j = 1 : d
            gij = G(k, j);
            term1 = term1 + 4 * di(k) * di(j) * (di(k)^2 * di(j) - alpha) * gij^2;
        end
        for i = 1 : d
            gij = G(i, k);
            term2 = term2 + 2 * di(i)^2 * (di(i)^2 * di(k) - alpha) * gij^2;
        end
        grad(k) = term1 + term2;
    end

    % Partial derivative w.r.t. alpha
    grad(end) = -2 * sum(sum((di'.^2 .* G .* di') - alpha * G).^2);

    % Hessian computation
    if nargout > 2
        hess = zeros(d+1, d+1);
        for k = 1 : d
            for l = 1 : d
                if k == l
                    % Diagonal element ∂²f/∂d_k²
                    sum1 = 0;
                    sum2 = 0;
                    for j = 1 : d
                        gij = G(k, j);
                        sum1 = sum1 + (4 * di(j) * ((3 * di(k)^2 * di(j) - alpha))) * gij^2;
                    end
                    for i = 1 : d
                        gij = G(i, k);
                        sum2 = sum2 + 2 * di(i)^2 * (3 * di(i)^2 * gij^2);
                    end
                    hess(k, k) = sum1 + sum2;
                else
                    % Cross term ∂²f/∂d_k∂d_l
                    hess(k, l) = 12 * di(k)^2 * di(l) * G(k, l)^2;
                end
            end
            % Cross term ∂²f/∂d_k∂alpha
            hess(k, end) = -4 * di(k)^2 * sum(G(k,:).^2 .* di');
            hess(end, k) = hess(k, end); % symmetry
        end
        % ∂²f/∂alpha²
        hess(end, end) = 2 * sum(sum(G.^2));
    end
end

