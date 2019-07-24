function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
MyTheta1 = [zeros(size(Theta1,1),1),Theta1(:,2:end)];
MyTheta2 = [zeros(size(Theta2,1),1),Theta2(:,2:end)];
% Setup some useful variables
m = size(X, 1);
X = [ones(1,m);X'];
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% a1 a2 a3
Y=zeros(num_labels,m);
my_a=[1:num_labels]';
for i=1:m
  Y(:,i)= (y(i)==my_a);
endfor

a2 = sigmoid(Theta1 * X);
a3 = sigmoid(Theta2 * [ones(1,m);a2]);
%fprintf('in order sizes X y Y a2 a3')
%disp(size(X))
%disp(size(y))
%disp(size(Y))
%disp(size(a2))
%Sdisp(size(a3))
J = (((-1)*sum(sum(Y.*log(a3) + (1-Y).*log(1-a3)))) + ...
    (0.5 *lambda * (sum(sum(MyTheta1.^2))+sum(sum(MyTheta2.^2)))))*(1/m);

Delta3 = a3 - Y;
Delta2 = (Theta2(:,2:end)' * Delta3).*(a2.*(1-a2));
a2 = [ones(1,m);a2];
%for i=1:m
% Theta1_grad = Theta1_grad + Delta2(:,i)*(X(:,i)');
%  Theta2_grad = Theta2_grad + Delta3(:,i)*(a2(:,i)');
%endfor
Theta1_grad = Theta1_grad + Delta2*X';
Theta2_grad = Theta2_grad + Delta3*a2';

Theta1_grad = (Theta1_grad + lambda*MyTheta1 )*(1/m);
Theta2_grad = (Theta2_grad + lambda*MyTheta2)*(1/m);





  
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
