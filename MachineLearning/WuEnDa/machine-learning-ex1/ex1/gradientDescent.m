function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % theta_0 = theta(1);
    % theta_1 = theta(2);

    % J = computeCost(X, y, theta);
    % fprintf('\nbefore: J = %f', J);

    % theta_0 = theta_0 - alpha * sum(X * theta - y) / m;
    % size(X * theta - y)
    % size(X)
    % size(X(:, 2)')
    % theta_1 = theta_1 - alpha * (X(:, 2)' * (X * theta - y)) / m;
    % theta_1 = theta_1 - alpha * sum(X(:, 2) .* (X * theta - y)) / m;

    % theta = [theta_0; theta_1];

    beta = X' * (X * theta - y);
    theta = theta - (alpha / m) * beta;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    % fprintf('\nafter: J = %f\n', J_history(iter));

end

end
