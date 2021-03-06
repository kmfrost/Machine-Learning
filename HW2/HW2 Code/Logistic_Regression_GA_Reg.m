%this script runs LR using regularized gradient ascent on the training and
%test data supplied from 'SenatorVoting.mat'
%It also plots the log likelihood function to ensure that gradient asecnet
%converges properly

%the output is saved as 'LR_preds.csv'


load('SenatorVoting.mat')

[m, n] = size(TrainData);

% Add intercept term to x and X_test
X = [ones(m, 1) TrainData];
y = TrainLabel;

w = zeros(n + 1, 1);

min_change = 0.0001;
eta = 0.005;
max_iters = 500;
lambda = 0.15;

[w, LL] = gradientAscentReg(X, y, w, eta, min_change, max_iters, lambda);

[m1, n1] = size(TestData);
predlabels = predict(w, [ones(m1, 1) TestData]);

csv_data = [(1:m1)', predlabels];
dlmwrite('LR_preds.csv','Senator_ID,Party', 'delimiter', '', 'coffset', 1);
dlmwrite('LR_preds.csv', csv_data, '-append');

figure;
plot(LL)
