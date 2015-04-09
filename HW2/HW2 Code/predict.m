function p = predict(w, X)
%predict labels for given input X and weight vector w (using LR)
p = round(sigmoid(X*w));