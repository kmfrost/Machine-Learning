function [rmse] = loocvreg_slow(xTr, yTr)
%this function calculates the normalized LOOCVerror (given training x and
%y) using the 'slow'/naive method

%inputs: xTr = NxD feature matrix, yTr is Nx1 label vector

xTr = xTr.';
yTr = yTr.';

[m, n] = size(xTr);

LOOCVerror = 0;

for ii=1:n  % loop through each x to leave it out
    X_minus_i = xTr(:, [1:ii-1, ii+1:end]).';
    Y_minus_i = yTr([1:ii-1, ii+1:end]).';
    w = pinv(X_minus_i.'*X_minus_i)*X_minus_i.'*Y_minus_i;
    y_hat_minus_i = (w.'*xTr);
    LOOCVerror = LOOCVerror + (yTr(ii)-y_hat_minus_i(ii))^2;
end

rmse = sqrt(LOOCVerror/n);

