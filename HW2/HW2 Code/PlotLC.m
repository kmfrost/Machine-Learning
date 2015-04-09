%this script plots the learning curves for a NB and a LR classifier using
%the training and test data from 'SenatorVoting.mat'

load('SenatorVoting.mat')

[m, n] = size(TrainData);
runs = 20;
NB_cv_error = zeros(runs, 1);
LR_cv_error = zeros(runs, 1);
num_train = round(m*(2/3));
num_test = m-num_train;

%use_train = [2 4 8 16 33]; % only test using these specific numbers of
                            %training points
use_train = [1:num_train]; %test using 1 through 2/3 of the train data
%use_train = [33]; %just test a full 2/3 of the train data

for ii = 1:runs
    %randomly divide the data into train (2/3) and test (1/3) sets
    rand_ints = randperm(m);
    train_ints = rand_ints(1:num_train);
    test_ints = rand_ints(num_train+1:end);
    
    for jj=1:length(use_train)
        % Add intercept term to x and X_test
        subset = train_ints(1:use_train(jj));
        X = [ones(use_train(jj), 1) TrainData(subset, :)];
        y = TrainLabel(subset);
        test_data = [ones(num_test, 1) TrainData(test_ints, :)];
        
        lr_predlabels = LR_GA_Reg(X, y, test_data);
        nb_predlabels = NB_Classifier(TrainData(subset, :), y, TrainData(test_ints, :));

        LR_cv_error(ii, jj) = sum(abs(TrainLabel(test_ints)-lr_predlabels));
        NB_cv_error(ii, jj) = sum(abs(TrainLabel(test_ints)-nb_predlabels));
    end
end

figure;
plot(use_train, mean(LR_cv_error), '*-');
hold on;
grid on;
plot(use_train, mean(NB_cv_error), '*-');
legend('Logistic Regression', 'Naive Bayes');
title(sprintf('Average error vs. number of training points over %d runs', runs));
xlabel('Size of training data');
ylabel('Average Error');

mean(LR_cv_error);
mean(NB_cv_error);

