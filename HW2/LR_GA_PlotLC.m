load('SenatorVoting.mat')

[m, n] = size(TrainData);
runs = 50;
cv_error = zeros(runs, 1);
num_train = round(m*(2/3));
num_test = m-num_train;

%use_train = [2 4 8 16 33];
use_train = [1:num_train];

for i = 1:runs
    %randomly divide the data into train (2/3) and test (1/3) sets
    rand_ints = randperm(m);
    train_ints = rand_ints(1:num_train);
    test_ints = rand_ints(num_train+1:end);
    
    for j=1:length(use_train)
        % Add intercept term to x and X_test
        subset = train_ints(1:use_train(j));
        X = [ones(use_train(j), 1) TrainData(subset, :)];
        y = TrainLabel(subset);

        %initialize weight vector
        w = zeros(n + 1, 1);

        min_change = 0.0001;
        eta = 0.005;
        max_iters = 500;
        lambda = 0.015;

        [w, LL] = gradientAscentReg(X, y, w, eta, min_change, max_iters, lambda);

        predlabels = predict(w, [ones(num_test, 1) TrainData(test_ints, :)]);

        cv_error(i, j) = sum(abs(TrainLabel(test_ints)-predlabels));
    end
end

figure;
plot(use_train, mean(cv_error), '*-');
title(sprintf('Average error vs. number of training points over %d runs', runs));
xlabel('Size of training data');
ylabel('Average Error');


