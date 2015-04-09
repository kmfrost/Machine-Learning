function [rmse] = loocvreg_fast(xTr, yTr)
%this function calculates the normalized LOOCVerror (given training x and
%y) using the 'fast' method derived from the homework problems

%inputs: xTr = NxD feature matrix, yTr is Nx1 label vector


xTr = xTr.';
yTr = yTr.';

[m, n] = size(xTr);

H = xTr.'*pinv(xTr*xTr.')*xTr;
YhatTr = (H*yTr.').';

LOOCVerror = sum(((YhatTr-yTr)./(1-diag(H).')).^2);

rmse = sqrt(LOOCVerror/n);