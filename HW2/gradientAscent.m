function [w, LL_history] = gradientAscent(X, y, w, eta, min_change, max_iters)
%run gradient descent with the given parameters

change = 1;
LL_history = zeros(max_iters,1);

for iter = 1:max_iters
    gradient = X.'*(y-sigmoid(X*w));
    change = eta*gradient;
    w = w + change;
    LL_history(iter) = log_likelihood(w, X, y);
    if change < min_change
        break
    end
end

%disp(iter)

