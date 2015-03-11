load('SenatorVoting.mat')

[m, n] = size(TrainData);
runs = 10;
cv_error = zeros(runs, 1);
num_train = round(m*(2/3));
num_test = m-num_train;

for i = 1:runs
    %randomly divide the data into train (2/3) and test (1/3) sets
    rand_ints = randperm(m);
    train_ints = rand_ints(1:num_train);
    test_ints = rand_ints(num_train+1:end);
    
    % Add intercept term to x and X_test
    X = [ones(num_train, 1) TrainData(train_ints, :)];
    y = TrainLabel(train_ints);

    %initialize weight vector
    w = zeros(n + 1, 1);

    min_change = 0.0001;
    eta = 0.005;
    max_iters = 500;
    lambda = 0.1

    [w, LL] = gradientAscentReg(X, y, w, eta, min_change, max_iters, lambda);

    predlabels = predict(w, [ones(num_test, 1) TrainData(test_ints, :)]);
    
    cv_error(i) = sum(abs(TrainLabel(test_ints)-predlabels));
end

avg_cv_error = mean(cv_error)


