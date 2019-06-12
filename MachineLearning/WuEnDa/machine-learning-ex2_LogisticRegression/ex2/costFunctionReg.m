function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X * theta);

% J = -1 / m * sum(y .* log(h) + (1 .- y) .* log(1 .- h)); 
% 
% theta_from_2 = zeros(size(theta) - 1, 1);
% for i = 2:size(theta),
%     theta_from_2(i - 1) = theta(i);
% end;
% 
% fprintf('size(theta): %f\n', size(theta));
% fprintf('size(theta_from_2): %f\n', size(theta_from_2));
% 
% J = J + lambda / 2 / m * sum(theta_from_2.^2);

% 需要去除theta(1)的值。
J = -1 / m * (y' * log(h) + (1 - y)' * log(1 - h)) ...
        + lambda / 2 / m * sum(theta(2 : size(theta)) .^ 2);

% grad = X' * (h .- y) ./ m;
% 
% for i = 2:length(theta),
%     grad(i) = grad(i) + lambda / m * theta(i);
% end;

% theta(1)和其他值的计算方法是不一样的。两个需要分开计算。
grad(1, :) = X(:, 1)' * (h - y) ./ m;
grad(2 : size(theta), :) = X(:, 2 : size(theta))' * (h - y) ./ m ...
                + lambda / m * theta(2 : size(theta), :);



% =============================================================

end
