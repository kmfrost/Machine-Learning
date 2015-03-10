load('SenatorVoting.mat')

[m, n] = size(TrainData);

% Add intercept term to x and X_test
X = [ones(m, 1) TrainData];
y = TrainLabel;

initial_theta = zeros(n + 1, 1);

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

[m1, n1] = size(TestData);
predlabels = predict(theta, [ones(m1, 1) TestData]);

csv_data = [(1:m1)', predlabels];
dlmwrite('predlabels.csv','Senator_ID,Party', 'delimiter', '', 'coffset', 1);
dlmwrite('predlabels.csv', csv_data, '-append');
