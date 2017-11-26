function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); %5000
num_labels = size(Theta2, 1); %10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); %5000x1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1) X]; %5000x401
for i=1:m
    a1 = X(i,:); %1x401
    z2 = Theta1*a1'; %25x1
    a2 = sigmoid(z2);
    a2 = [1; a2]; %26x1
    z3 = Theta2*a2; %10x1
    h = sigmoid(z3);
   [~,p(i)] = max(h);
end
% =========================================================================
end
