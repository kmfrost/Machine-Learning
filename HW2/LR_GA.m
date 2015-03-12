function predlabels = LR_GA(TrainData, TrainLabel, TestData)
%return prediction labels for a logistic regression classifier 
%using gradient ascent given Training Data, Traning
%Lables, and Test Data

[m, n] = size(TrainData);

% Add intercept term to x and X_test
X = [ones(m, 1) TrainData];
y = TrainLabel;

w = zeros(n + 1, 1);

min_change = 0.0001;
eta = 0.01;
max_iters = 500;

[w, LL] = gradientAscent(X, y, w, eta, min_change, max_iters);

[m1, n1] = size(TestData);
predlabels = predict(w, [ones(m1, 1) TestData]);

