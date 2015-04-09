function J = log_likelihood(w, X, y)
%Calculate the cost for the given weight vector and some input/output pairs

m = length(y); % number of training examples

theta_x = sigmoid(X*w);

%calculate the cost
J = sum(y.*(X*w) - log(1+exp(X*w)));
end