function p = predict(w, X)
%predict labels for given input X and weight vector w

m = size(X, 1); % Number of training examples
p = round(sigmoid(X*w));